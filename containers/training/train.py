# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Training script for FSI Fraud Propensity Scoring model.
Compatible with: local testing, SageMaker AI Training, and AWS Clean Rooms ML.
"""

import argparse, os, sys, json, glob, traceback, logging
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Columns as they arrive from Clean Rooms ML (pre-joined, no customer_id)
CLEANROOMS_COLUMNS = [
    "account_id", "login_count", "failed_auth_attempts",
    "account_age_days", "linked_devices", "avg_transaction_value",
    "geo_spread_score", "night_activity_ratio", "avg_session_duration_min",
    "ip_change_frequency", "dormant_reactivation", "observation_date",
    "card_type", "chargeback_count", "declined_transactions",
    "transaction_velocity", "merchant_category_diversity",
    "cross_border_ratio", "days_since_last_dispute", "last_dispute_date",
    "avg_txn_amount", "txn_amount_stddev", "weekend_txn_ratio",
    "rapid_succession_count", "unique_country_count",
    "is_suspicious"
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--output_dir", type=str, default=os.environ.get("SM_OUTPUT_DIR", "/opt/ml/output/data"))
    parser.add_argument("--train_dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAIN",
                                os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")))
    parser.add_argument("--train_file_format", type=str, default=os.environ.get("FILE_FORMAT", "csv"))
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    return parser.parse_args()


def load_data(train_dir, file_format):
    logger.info(f"Loading data from {train_dir} (format: {file_format})")
    if os.path.exists(train_dir):
        for root, dirs, files in os.walk(train_dir):
            logger.info(f"  Dir: {root}, subdirs: {dirs}, files: {files}")
    else:
        alternatives = ["/opt/ml/input/data/training", "/opt/ml/input/data/train", "/opt/ml/input/data"]
        for alt in alternatives:
            if os.path.exists(alt):
                train_dir = alt
                break
        else:
            raise FileNotFoundError(f"No training data directory found. Tried: {train_dir}, {alternatives}")

    all_files = []
    for root, dirs, files in os.walk(train_dir):
        for f in files:
            if f.endswith(f".{file_format}") or not os.path.splitext(f)[1]:
                all_files.append(os.path.join(root, f))
    if not all_files:
        all_files = [f for f in glob.glob(os.path.join(train_dir, "**/*"), recursive=True) if os.path.isfile(f)]
    if not all_files:
        raise FileNotFoundError(f"No data files found in {train_dir}")

    logger.info(f"Found {len(all_files)} files: {all_files}")
    dataframes = {}
    for filepath in all_files:
        name = os.path.basename(filepath).replace(f".{file_format}", "")
        try:
            if file_format == "csv":
                df = pd.read_csv(filepath)
                first_col = str(df.columns[0])
                # Detect headerless Clean Rooms ML output
                is_headerless = (
                    first_col not in ["customer_id", "account_id", "card_type",
                                      "login_count", "chargeback_count"]
                    and len(df.columns) == len(CLEANROOMS_COLUMNS)
                )
                if is_headerless:
                    df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
                elif len(df.columns) == len(CLEANROOMS_COLUMNS) - 1:
                    df = pd.read_csv(filepath, header=None, names=CLEANROOMS_COLUMNS)
            else:
                df = pd.read_parquet(filepath)
        except Exception as e:
            logger.error(f"Failed to read {filepath}: {e}")
            raise ValueError(f"Could not parse data file {filepath} as {file_format}: {e}")
        dataframes[name] = df
        logger.info(f"  Loaded {name}: {df.shape}")
    return dataframes


def engineer_features(dataframes):
    """Route to pre-joined or separate-file feature engineering."""
    pre_joined_df = None
    for name, df in dataframes.items():
        has_bank = "login_count" in df.columns or "failed_auth_attempts" in df.columns
        has_proc = "chargeback_count" in df.columns or "declined_transactions" in df.columns
        if has_bank and has_proc:
            pre_joined_df = df
            break

    if pre_joined_df is not None:
        return _engineer_features_prejoined(pre_joined_df)
    return _engineer_features_separate(dataframes)


def _derive_features(df):
    """Compute derived features common to both code paths."""
    df["auth_failure_rate"] = df["failed_auth_attempts"] / df["login_count"].clip(lower=1)
    df["decline_rate"] = df["declined_transactions"] / (
        df["declined_transactions"] + df["transaction_velocity"].clip(lower=0.1) * 30
    ).clip(lower=1)
    return df


def _engineer_features_prejoined(df):
    """Feature engineering when data arrives pre-joined (Clean Rooms ML path)."""
    df = _derive_features(df)
    feature_cols = [
        "login_count", "failed_auth_attempts", "account_age_days", "linked_devices",
        "avg_transaction_value", "geo_spread_score", "night_activity_ratio",
        "avg_session_duration_min", "ip_change_frequency", "dormant_reactivation",
        "chargeback_count", "declined_transactions", "transaction_velocity",
        "merchant_category_diversity", "cross_border_ratio", "days_since_last_dispute",
        "avg_txn_amount", "txn_amount_stddev", "weekend_txn_ratio",
        "rapid_succession_count", "unique_country_count",
        "auth_failure_rate", "decline_rate",
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols].fillna(0)
    y = df["is_suspicious"] if "is_suspicious" in df.columns else pd.Series(0, index=df.index)
    logger.info(f"Features shape: {X.shape}, Target dist:\n{y.value_counts().to_string()}")
    return X, y, feature_cols, None


def _engineer_features_separate(dataframes):
    """Feature engineering when bank and processor CSVs are separate files (local test path)."""
    bank_df = proc_df = None
    for name, df in dataframes.items():
        if "login_count" in df.columns or "failed_auth_attempts" in df.columns:
            bank_df = df
        elif "chargeback_count" in df.columns or "declined_transactions" in df.columns:
            proc_df = df
    if bank_df is None or proc_df is None:
        raise ValueError(f"Could not identify both datasets. Columns: {[list(df.columns) for df in dataframes.values()]}")

    # Aggregate bank data per customer
    bank_agg = bank_df.groupby("customer_id").agg(
        total_login_count=("login_count", "sum"),
        total_failed_auth=("failed_auth_attempts", "sum"),
        avg_account_age_days=("account_age_days", "mean"),
        total_linked_devices=("linked_devices", "sum"),
        avg_transaction_value=("avg_transaction_value", "mean"),
        avg_geo_spread_score=("geo_spread_score", "mean"),
        avg_night_activity_ratio=("night_activity_ratio", "mean"),
        avg_session_duration_min=("avg_session_duration_min", "mean"),
        avg_ip_change_frequency=("ip_change_frequency", "mean"),
        max_dormant_reactivation=("dormant_reactivation", "max"),
        num_accounts=("account_id", "nunique"),
    ).reset_index()

    # Aggregate processor data per customer
    proc_agg = proc_df.groupby("customer_id").agg(
        total_chargeback_count=("chargeback_count", "sum"),
        total_declined_transactions=("declined_transactions", "sum"),
        avg_transaction_velocity=("transaction_velocity", "mean"),
        avg_merchant_category_diversity=("merchant_category_diversity", "mean"),
        avg_cross_border_ratio=("cross_border_ratio", "mean"),
        min_days_since_last_dispute=("days_since_last_dispute", "min"),
        avg_txn_amount=("avg_txn_amount", "mean"),
        avg_txn_amount_stddev=("txn_amount_stddev", "mean"),
        avg_weekend_txn_ratio=("weekend_txn_ratio", "mean"),
        total_rapid_succession_count=("rapid_succession_count", "sum"),
        max_unique_country_count=("unique_country_count", "max"),
        num_cards=("card_type", "nunique"),
    ).reset_index()

    # Extract target from processor data
    if "is_suspicious" in proc_df.columns:
        target = proc_df.groupby("customer_id")["is_suspicious"].max().reset_index()
        proc_agg = proc_agg.merge(target, on="customer_id")
    else:
        proc_agg["is_suspicious"] = 0

    merged = bank_agg.merge(proc_agg, on="customer_id", how="inner")

    # Derived features
    merged["auth_failure_rate"] = merged["total_failed_auth"] / merged["total_login_count"].clip(lower=1)
    merged["decline_rate"] = merged["total_declined_transactions"] / (
        merged["total_declined_transactions"] + merged["avg_transaction_velocity"].clip(lower=0.1) * 30
    ).clip(lower=1)

    feature_cols = [
        "total_login_count", "total_failed_auth", "avg_account_age_days",
        "total_linked_devices", "avg_transaction_value", "avg_geo_spread_score",
        "avg_night_activity_ratio", "avg_session_duration_min", "avg_ip_change_frequency",
        "max_dormant_reactivation", "num_accounts",
        "total_chargeback_count", "total_declined_transactions", "avg_transaction_velocity",
        "avg_merchant_category_diversity", "avg_cross_border_ratio",
        "min_days_since_last_dispute", "avg_txn_amount", "avg_txn_amount_stddev",
        "avg_weekend_txn_ratio", "total_rapid_succession_count", "max_unique_country_count",
        "num_cards", "auth_failure_rate", "decline_rate",
    ]
    X = merged[feature_cols].fillna(0)
    y = merged["is_suspicious"]
    logger.info(f"Merged: {merged.shape[0]} customers, Features: {X.shape}")
    return X, y, feature_cols, merged["customer_id"]


def train_model(X, y, args):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)
    logger.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    model = GradientBoostingClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                       learning_rate=args.learning_rate, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "train_samples": X_train.shape[0], "test_samples": X_test.shape[0], "n_features": X_train.shape[1],
    }
    importance = dict(zip(X.columns, [round(float(v), 4) for v in model.feature_importances_]))
    metrics["feature_importance"] = dict(sorted(importance.items(), key=lambda x: -x[1]))
    logger.info(f"Metrics: {json.dumps({k: v for k, v in metrics.items() if k != 'feature_importance'}, indent=2)}")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    return model, metrics


def save_artifacts(model, metrics, feature_cols, model_dir, output_dir):
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))
    with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
        json.dump(feature_cols, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Artifacts saved to {model_dir} and {output_dir}")


def main():
    args = parse_args()
    logger.info(f"Arguments: {vars(args)}")
    try:
        sm_vars = {k: v for k, v in os.environ.items() if k.startswith("SM_") or k.startswith("FILE_")}
        logger.info(f"SageMaker env vars: {json.dumps(sm_vars, indent=2)}")
        dataframes = load_data(args.train_dir, args.train_file_format)
        X, y, feature_cols, customer_ids = engineer_features(dataframes)
        model, metrics = train_model(X, y, args)
        save_artifacts(model, metrics, feature_cols, args.model_dir, args.output_dir)
        logger.info("Training completed successfully.")
    except Exception as e:
        failure_path = "/opt/ml/output/failure"
        os.makedirs(os.path.dirname(failure_path), exist_ok=True)
        with open(failure_path, "w") as f:
            f.write(str(e)[:1024])
        logger.error(f"Training failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
