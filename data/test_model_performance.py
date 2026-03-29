"""
Quick local test: join bank + payment processor data, train GradientBoosting,
and evaluate fraud propensity scoring performance.

Tests three scenarios:
1. Clean Rooms mode (pre-joined, row-level)
2. Local/SageMaker mode (aggregated per-customer)
3. Single-party only (to prove both parties are needed)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)


def train_and_evaluate(X, y, label=""):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"  Confusion: TN={cm[0][0]}  FP={cm[0][1]}  FN={cm[1][0]}  TP={cm[1][1]}")

    importance = sorted(
        zip(X.columns, model.feature_importances_), key=lambda x: -x[1])
    print("\n  Feature Importance:")
    for feat, imp in importance:
        bar = "█" * int(imp * 80)
        print(f"    {feat:35s} {imp:.4f}  {bar}")

    return model


# Load data
bank = pd.read_csv("data/bank_account_behavior.csv")
proc = pd.read_csv("data/payment_processor_transactions.csv")

print(f"Bank rows: {len(bank)}, columns: {list(bank.columns)}")
print(f"Processor rows: {len(proc)}, columns: {list(proc.columns)}")
print(f"Bank customers: {bank['customer_id'].nunique()}")
print(f"Processor customers: {proc['customer_id'].nunique()}")

# === MODE 1: Clean Rooms (pre-joined, row-level) ===
print("\n" + "=" * 70)
print("MODE 1: Clean Rooms (pre-joined, row-level) — BOTH PARTIES")
print("=" * 70)

merged = bank.merge(proc, on="customer_id", how="inner")
print(f"Joined rows: {len(merged)}")
print(f"Target distribution:\n{merged['is_suspicious'].value_counts().to_string()}")

# Derived features
merged["auth_failure_rate"] = merged["failed_auth_attempts"] / merged["login_count"].clip(lower=1)
merged["decline_rate"] = merged["declined_transactions"] / (
    merged["declined_transactions"] + merged["transaction_velocity"] * 30).clip(lower=1)

feature_cols_both = [
    # Bank features
    "login_count", "failed_auth_attempts", "account_age_days", "linked_devices",
    "avg_transaction_value", "geo_spread_score", "night_activity_ratio",
    "avg_session_duration_min", "ip_change_frequency", "dormant_reactivation",
    # Derived
    "auth_failure_rate",
    # Processor features
    "chargeback_count", "declined_transactions", "transaction_velocity",
    "merchant_category_diversity", "cross_border_ratio", "days_since_last_dispute",
    "avg_txn_amount", "txn_amount_stddev", "weekend_txn_ratio",
    "rapid_succession_count", "unique_country_count",
    # Derived
    "decline_rate",
]

X1 = merged[feature_cols_both].fillna(0)
y1 = merged["is_suspicious"]
train_and_evaluate(X1, y1, "Both parties")

# === MODE 2: Bank-only features ===
print("\n" + "=" * 70)
print("MODE 2: BANK-ONLY features (to show bank alone is insufficient)")
print("=" * 70)

bank_features = [
    "login_count", "failed_auth_attempts", "account_age_days", "linked_devices",
    "avg_transaction_value", "geo_spread_score", "night_activity_ratio",
    "avg_session_duration_min", "ip_change_frequency", "dormant_reactivation",
    "auth_failure_rate",
]
X_bank = merged[bank_features].fillna(0)
train_and_evaluate(X_bank, y1, "Bank only")

# === MODE 3: Processor-only features ===
print("\n" + "=" * 70)
print("MODE 3: PROCESSOR-ONLY features (to show processor alone is insufficient)")
print("=" * 70)

proc_features = [
    "chargeback_count", "declined_transactions", "transaction_velocity",
    "merchant_category_diversity", "cross_border_ratio", "days_since_last_dispute",
    "avg_txn_amount", "txn_amount_stddev", "weekend_txn_ratio",
    "rapid_succession_count", "unique_country_count", "decline_rate",
]
X_proc = merged[proc_features].fillna(0)
train_and_evaluate(X_proc, y1, "Processor only")

# === MODE 4: Aggregated per-customer ===
print("\n" + "=" * 70)
print("MODE 4: Local/SageMaker (aggregated per-customer) — BOTH PARTIES")
print("=" * 70)

bank_agg = bank.groupby("customer_id").agg(
    total_logins=("login_count", "sum"),
    total_failed_auth=("failed_auth_attempts", "sum"),
    num_accounts=("account_id", "nunique"),
    max_linked_devices=("linked_devices", "max"),
    avg_transaction_value=("avg_transaction_value", "mean"),
    max_geo_spread=("geo_spread_score", "max"),
    avg_night_ratio=("night_activity_ratio", "mean"),
    min_session_duration=("avg_session_duration_min", "min"),
    max_ip_change_freq=("ip_change_frequency", "max"),
    any_dormant=("dormant_reactivation", "max"),
    min_account_age=("account_age_days", "min"),
).reset_index()
bank_agg["auth_failure_rate"] = bank_agg["total_failed_auth"] / bank_agg["total_logins"].clip(lower=1)

proc_agg = proc.groupby("customer_id").agg(
    total_chargebacks=("chargeback_count", "sum"),
    total_declined=("declined_transactions", "sum"),
    avg_velocity=("transaction_velocity", "mean"),
    num_card_types=("card_type", "nunique"),
    max_merchant_diversity=("merchant_category_diversity", "max"),
    max_cross_border_ratio=("cross_border_ratio", "max"),
    min_days_since_dispute=("days_since_last_dispute", "min"),
    avg_txn_amount=("avg_txn_amount", "mean"),
    max_txn_stddev=("txn_amount_stddev", "max"),
    avg_weekend_ratio=("weekend_txn_ratio", "mean"),
    total_rapid_succession=("rapid_succession_count", "sum"),
    max_unique_countries=("unique_country_count", "max"),
).reset_index()

target = proc.groupby("customer_id")["is_suspicious"].max().reset_index()
proc_agg = proc_agg.merge(target, on="customer_id")

merged_agg = bank_agg.merge(proc_agg, on="customer_id", how="inner")
print(f"Merged customers: {len(merged_agg)}")
print(f"Target distribution:\n{merged_agg['is_suspicious'].value_counts().to_string()}")

agg_features = [
    "total_logins", "total_failed_auth", "num_accounts", "max_linked_devices",
    "avg_transaction_value", "max_geo_spread", "avg_night_ratio",
    "min_session_duration", "max_ip_change_freq", "any_dormant",
    "min_account_age", "auth_failure_rate",
    "total_chargebacks", "total_declined", "avg_velocity", "num_card_types",
    "max_merchant_diversity", "max_cross_border_ratio", "min_days_since_dispute",
    "avg_txn_amount", "max_txn_stddev", "avg_weekend_ratio",
    "total_rapid_succession", "max_unique_countries",
]

X4 = merged_agg[agg_features].fillna(0)
y4 = merged_agg["is_suspicious"]
train_and_evaluate(X4, y4, "Aggregated both parties")

# === SUMMARY ===
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print("If both-party ROC-AUC is significantly higher than single-party,")
print("the data correctly requires collaboration for good predictions.")
