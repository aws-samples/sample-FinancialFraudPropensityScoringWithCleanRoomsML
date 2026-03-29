# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML FSI — Create QuickSight Dashboard (Step 6)
Reads all account/region config from config.py.

Run with: python scripts/create_dashboard.py  (from the project root folder)

Prerequisites:
  - run_cleanrooms_ml.py must have completed successfully
  - Inference output must exist at s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/

What this script does (idempotent — safe to re-run):
  1. Register QuickSight account (skip if already exists)
  2. Register QuickSight admin user (skip if already exists)
  3. Create Glue table for inference output
  4. Create Athena data source in QuickSight
  5. Create QuickSight dataset
  6. Create analysis + publish dashboard (4 sheets)
  7. Print dashboard URL
"""

import sys, os, json, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate(require_qs_email=True)

import boto3
from botocore.exceptions import ClientError

# ─── Clients ──────────────────────────────────────────────
session     = boto3.Session(region_name=AWS_REGION)
session_iam = boto3.Session(region_name="us-east-1")   # QS identity plane is global
qs          = session.client("quicksight")
qs_iam      = session_iam.client("quicksight")
glue        = session.client("glue")
s3          = session.client("s3")
iam         = session.client("iam")
sts         = session.client("sts")

# ─── Resource IDs (imported from config) ──────────────────
# QS_DATASOURCE_ID, QS_DS_INFERENCE, QS_ANALYSIS_ID, QS_DASHBOARD_ID
INFERENCE_TABLE  = "inference_output"


def log(msg):
    print(f"  → {msg}")


# ═══════════════════════════════════════════════════════════
# SECTION 1 — QuickSight account registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_account():
    """Register QuickSight ENTERPRISE account. No-op if already registered."""
    print("\n[1/6] Ensuring QuickSight account...")
    try:
        resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
        status = resp["AccountInfo"]["AccountSubscriptionStatus"]
        log(f"QuickSight account already exists (status: {status})")
        if status not in ("ACCOUNT_CREATED", "ACTIVE"):
            print(f"  WARNING: QuickSight account status is '{status}'. Waiting up to 60s...")
            _wait_for_qs_account()
        return
    except ClientError as e:
        if e.response["Error"]["Code"] not in ("ResourceNotFoundException", "AccessDeniedException"):
            raise

    log(f"Registering QuickSight ENTERPRISE account (email: {QS_NOTIFICATION_EMAIL})")
    try:
        qs_iam.create_account_subscription(
            AwsAccountId=AWS_ACCOUNT_ID,
            AccountName=f"{PREFIX}-{AWS_ACCOUNT_ID}",
            Edition="ENTERPRISE",
            AuthenticationMethod="IAM_AND_QUICKSIGHT",
            NotificationEmail=QS_NOTIFICATION_EMAIL,
        )
        log("QuickSight account registration submitted — waiting for activation...")
        _wait_for_qs_account()
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("ResourceExistsException", "ConflictException"):
            log("QuickSight account already exists (race condition) — continuing")
        else:
            raise


def _wait_for_qs_account(max_wait=120):
    for _ in range(max_wait // 10):
        try:
            resp = qs_iam.describe_account_subscription(AwsAccountId=AWS_ACCOUNT_ID)
            status = resp["AccountInfo"]["AccountSubscriptionStatus"]
            if status == "ACCOUNT_CREATED":
                log("QuickSight account is ACTIVE")
                return
            log(f"  Account status: {status} — waiting...")
        except ClientError:
            pass
        time.sleep(10)
    print("  WARNING: QuickSight account did not reach ACCOUNT_CREATED within timeout. Continuing anyway.")


# ═══════════════════════════════════════════════════════════
# SECTION 2 — QuickSight admin user registration
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_user():
    """Register the current IAM caller as a QuickSight ADMIN user. No-op if exists."""
    print("\n[2/6] Ensuring QuickSight admin user...")

    identity = sts.get_caller_identity()
    caller_arn = identity["Arn"]
    arn_parts = caller_arn.split(":")
    raw_name  = arn_parts[-1]
    if raw_name.startswith("assumed-role/"):
        username = "/".join(raw_name.split("/")[1:])
    else:
        username = raw_name.split("/")[-1]

    try:
        qs_iam.describe_user(AwsAccountId=AWS_ACCOUNT_ID, Namespace="default", UserName=username)
        log(f"QuickSight user already exists: {username}")
        return username
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    log(f"Registering QuickSight ADMIN user: {username}")
    is_assumed_role = "assumed-role" in caller_arn
    if is_assumed_role:
        parts = caller_arn.split(":")
        role_parts = parts[-1].split("/")
        role_name    = role_parts[1]
        session_name = role_parts[2]
        role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{role_name}"
    else:
        role_arn     = caller_arn
        session_name = None

    qs_iam.register_user(
        AwsAccountId=AWS_ACCOUNT_ID,
        Namespace="default",
        IdentityType="IAM",
        IamArn=role_arn,
        UserRole="ADMIN",
        Email=QS_NOTIFICATION_EMAIL,
        **({"SessionName": session_name} if session_name else {}),
    )
    log(f"Registered user: {username}")
    return username


def _qs_user_arn(username):
    return f"arn:aws:quicksight:us-east-1:{AWS_ACCOUNT_ID}:user/default/{username}"


# ═══════════════════════════════════════════════════════════
# SECTION 3 — Glue table for inference output
# ═══════════════════════════════════════════════════════════

def prepare_glue_tables():
    """Register the inference output CSV as a Glue table (Athena-queryable)."""
    print("\n[3/6] Preparing Glue tables for dashboard data...")
    _ensure_glue_db()
    _register_inference_table()


def _ensure_glue_db():
    try:
        glue.create_database(DatabaseInput={"Name": GLUE_DB, "Description": "AWS Clean Rooms ML FSI Fraud demo"})
        log(f"Created Glue database: {GLUE_DB}")
    except glue.exceptions.AlreadyExistsException:
        log(f"Glue database already exists: {GLUE_DB}")


def _register_inference_table():
    """Register inference output CSV location as a Glue external table."""
    # Columns match the FSI inference handler output:
    # fraud_propensity_score, predicted_suspicious + passthrough contextual columns
    columns = [
        {"Name": "fraud_propensity_score",   "Type": "double"},
        {"Name": "predicted_suspicious",     "Type": "int"},
        {"Name": "account_id",               "Type": "string"},
        {"Name": "card_type",                "Type": "string"},
        {"Name": "chargeback_count",         "Type": "int"},
        {"Name": "declined_transactions",    "Type": "int"},
        {"Name": "transaction_velocity",     "Type": "double"},
        {"Name": "cross_border_ratio",       "Type": "double"},
        {"Name": "rapid_succession_count",   "Type": "int"},
        {"Name": "unique_country_count",     "Type": "int"},
    ]
    table_input = {
        "Name": INFERENCE_TABLE,
        "StorageDescriptor": {
            "Columns": columns,
            "Location": f"s3://{OUTPUT_BUCKET}/cleanrooms-ml-output/",
            "InputFormat":  "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.serde2.OpenCSVSerde",
                "Parameters": {
                    "separatorChar": ",",
                    "quoteChar": '"',
                    "skip.header.line.count": "1",
                },
            },
        },
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": "csv"},
    }
    try:
        glue.create_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Created Glue table: {INFERENCE_TABLE}")
    except glue.exceptions.AlreadyExistsException:
        glue.update_table(DatabaseName=GLUE_DB, TableInput=table_input)
        log(f"Updated Glue table: {INFERENCE_TABLE}")


# ═══════════════════════════════════════════════════════════
# SECTION 3b — Grant QuickSight access to S3 + Athena
# ═══════════════════════════════════════════════════════════

def ensure_quicksight_s3_access():
    """Grant QuickSight permission to read the output S3 bucket via managed service roles."""
    qs_service_role = "aws-quicksight-service-role-v0"
    qs_s3_role      = "aws-quicksight-s3-consumers-role-v0"

    policy_doc = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "FSIFraudOutputBucketAccess",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:ListBucket", "s3:GetBucketLocation"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
            {
                "Sid": "AthenaAccess",
                "Effect": "Allow",
                "Action": [
                    "athena:BatchGetQueryExecution", "athena:GetQueryExecution",
                    "athena:GetQueryResults", "athena:GetQueryResultsStream",
                    "athena:ListQueryExecutions", "athena:StartQueryExecution",
                    "athena:StopQueryExecution", "athena:GetWorkGroup",
                ],
                "Resource": "*",
            },
            {
                "Sid": "LakeFormationDataAccess",
                "Effect": "Allow",
                "Action": ["lakeformation:GetDataAccess"],
                "Resource": "*",
            },
            {
                "Sid": "GlueAccess",
                "Effect": "Allow",
                "Action": [
                    "glue:GetDatabase", "glue:GetDatabases",
                    "glue:GetTable", "glue:GetTables",
                    "glue:GetPartition", "glue:GetPartitions",
                    "glue:BatchGetPartition",
                ],
                "Resource": [
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:catalog",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:database/{GLUE_DB}",
                    f"arn:aws:glue:{AWS_REGION}:{AWS_ACCOUNT_ID}:table/{GLUE_DB}/*",
                ],
            },
            {
                "Sid": "AthenaResultsBucket",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket",
                           "s3:GetBucketLocation", "s3:AbortMultipartUpload"],
                "Resource": [
                    f"arn:aws:s3:::{OUTPUT_BUCKET}",
                    f"arn:aws:s3:::{OUTPUT_BUCKET}/*",
                ],
            },
        ],
    }

    for role_name in [qs_service_role, qs_s3_role]:
        try:
            iam.put_role_policy(
                RoleName=role_name,
                PolicyName="cleanrooms-ml-fsi-fraud-qs-access",
                PolicyDocument=json.dumps(policy_doc),
            )
            log(f"Granted S3/Athena/Glue access to QuickSight role: {role_name}")
        except iam.exceptions.NoSuchEntityException:
            log(f"QuickSight role not found (skipping): {role_name}")
        except Exception as e:
            log(f"Could not update role {role_name} (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════
# SECTION 3c — Lake Formation permissions for QuickSight
# ═══════════════════════════════════════════════════════════

def ensure_lakeformation_permissions():
    """Grant least-privilege Lake Formation permissions for QuickSight/Athena access.

    AWS best practice (per Lake Formation docs) is to grant specific LF permissions
    to specific principals rather than using IAMAllowedPrincipals (which is a legacy
    backward-compatibility mode that bypasses LF fine-grained access control).

    The correct principals for QuickSight+Athena access are:
      1. The QuickSight user ARN (arn:aws:quicksight:...:user/default/<username>)
         — for catalog metadata access (DESCRIBE on DB, SELECT+DESCRIBE on table)
      2. The IAM role used by Athena to vend credentials (lakeformation:GetDataAccess)
         — already covered by the IAM inline policy in ensure_quicksight_s3_access()

    The QuickSight service-linked role (aws-quicksight-service-role-v0) cannot be
    used as a LF principal directly — LF rejects service-linked role paths.

    This function is idempotent: granting an already-existing permission is a no-op.
    Falls back gracefully if LF is in IAM-passthrough mode (legacy accounts).
    """
    lf = session.client("lakeformation")

    # Check if Lake Formation is in legacy IAM-passthrough mode.
    # Non-empty CreateDatabaseDefaultPermissions means LF defers to IAM — no LF grants needed.
    try:
        settings = lf.get_data_lake_settings()
        default_db_perms = settings.get("DataLakeSettings", {}).get("CreateDatabaseDefaultPermissions", [])
        if default_db_perms:
            log("Lake Formation is in IAM-passthrough mode — skipping LF grants")
            return
    except Exception as e:
        log(f"Could not check Lake Formation settings (skipping LF grants): {e}")
        return

    # Resolve the QuickSight user ARN for the current caller.
    # LF requires the QuickSight user ARN (arn:aws:quicksight:...:user/default/<name>)
    # as the principal — not the IAM role ARN — for catalog-level permissions.
    try:
        identity = sts.get_caller_identity()
        caller_arn = identity["Arn"]
        raw_name = caller_arn.split(":")[-1]
        if raw_name.startswith("assumed-role/"):
            username = "/".join(raw_name.split("/")[1:])
        else:
            username = raw_name.split("/")[-1]
        qs_user_arn = f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:user/default/{username}"
    except Exception as e:
        log(f"Could not resolve QuickSight user ARN (skipping LF grants): {e}")
        return

    # Least-privilege grants per AWS docs:
    #   - DESCRIBE on database: allows QuickSight to see the DB in catalog
    #   - SELECT + DESCRIBE on table: allows Athena to query the inference output
    grants = [
        {
            "Id": "fsi-db-describe",
            "Principal": {"DataLakePrincipalIdentifier": qs_user_arn},
            "Resource": {"Database": {"Name": GLUE_DB}},
            "Permissions": ["DESCRIBE"],
            "PermissionsWithGrantOption": [],
        },
        {
            "Id": "fsi-tbl-select",
            "Principal": {"DataLakePrincipalIdentifier": qs_user_arn},
            "Resource": {"Table": {"DatabaseName": GLUE_DB, "Name": "inference_output"}},
            "Permissions": ["SELECT", "DESCRIBE"],
            "PermissionsWithGrantOption": [],
        },
    ]

    try:
        resp = lf.batch_grant_permissions(Entries=grants)
        failures = resp.get("Failures", [])
        already_exists = [
            f for f in failures
            if "AlreadyExistsException" in f.get("Error", {}).get("ErrorCode", "")
        ]
        real_failures = [f for f in failures if f not in already_exists]

        if already_exists:
            log(f"Lake Formation permissions already granted to QuickSight user")
        if real_failures:
            for f in real_failures:
                log(f"LF grant warning [{f['RequestEntry']['Id']}]: {f['Error']['ErrorMessage']}")
        if not failures:
            log(f"Granted Lake Formation SELECT+DESCRIBE on {GLUE_DB}.inference_output to {qs_user_arn}")
    except Exception as e:
        log(f"Could not apply Lake Formation grants (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════
# SECTION 4 — Athena data source
# ═══════════════════════════════════════════════════════════

def ensure_datasource(user_arn):
    """Create (or verify) the Athena data source in QuickSight."""
    print("\n[4/6] Ensuring QuickSight Athena data source...")
    try:
        qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        log(f"Data source already exists: {QS_DATASOURCE_ID}")
        return
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise

    qs.create_data_source(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSourceId=QS_DATASOURCE_ID,
        Name="CleanRoomsML FSI — Athena",
        Type="ATHENA",
        DataSourceParameters={"AthenaParameters": {"WorkGroup": "primary"}},
        Permissions=[{
            "Principal": user_arn,
            "Actions": [
                "quicksight:DescribeDataSource",
                "quicksight:DescribeDataSourcePermissions",
                "quicksight:PassDataSource",
                "quicksight:UpdateDataSource",
                "quicksight:DeleteDataSource",
                "quicksight:UpdateDataSourcePermissions",
            ],
        }],
    )
    log(f"Created Athena data source: {QS_DATASOURCE_ID}")
    _wait_for_datasource()


def _wait_for_datasource(max_wait=60):
    for _ in range(max_wait // 5):
        resp = qs.describe_data_source(AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)
        status = resp["DataSource"]["Status"]
        if status == "CREATION_SUCCESSFUL":
            log("Data source is ready")
            return
        if "FAILED" in status:
            raise RuntimeError(f"Data source creation failed: {status} — "
                               f"{resp['DataSource'].get('DataSourceErrorInfo', {})}")
        time.sleep(5)
    print("  WARNING: Data source did not reach CREATION_SUCCESSFUL within timeout.")


# ═══════════════════════════════════════════════════════════
# SECTION 5 — QuickSight dataset
# ═══════════════════════════════════════════════════════════

def _dataset_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDataSet", "quicksight:DescribeDataSetPermissions",
            "quicksight:PassDataSet", "quicksight:DescribeIngestion",
            "quicksight:ListIngestions", "quicksight:UpdateDataSet",
            "quicksight:DeleteDataSet", "quicksight:CreateIngestion",
            "quicksight:CancelIngestion", "quicksight:UpdateDataSetPermissions",
        ],
    }]


def ensure_datasets(user_arn):
    """Create (or update) the fraud inference SPICE dataset."""
    print("\n[5/6] Ensuring QuickSight datasets...")
    datasource_arn = f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:datasource/{QS_DATASOURCE_ID}"

    # Derived fields:
    #   risk_segment  — High/Medium/Low based on fraud_propensity_score
    #   score_decile  — 1–10 decile bucket
    #   risk_exposure — chargeback_count * fraud_propensity_score (estimated exposure weight)
    sql = (
        f"SELECT fraud_propensity_score, predicted_suspicious, "
        f"account_id, card_type, chargeback_count, declined_transactions, "
        f"transaction_velocity, cross_border_ratio, rapid_succession_count, unique_country_count, "
        f"CASE WHEN fraud_propensity_score > 0.7 THEN 'High' "
        f"     WHEN fraud_propensity_score >= 0.3 THEN 'Medium' "
        f"     ELSE 'Low' END AS risk_segment, "
        f"CAST(CEIL(PERCENT_RANK() OVER (ORDER BY fraud_propensity_score) * 10) AS INTEGER) AS score_decile, "
        f"chargeback_count * fraud_propensity_score AS risk_exposure "
        f"FROM {GLUE_DB}.{INFERENCE_TABLE}"
    )

    columns = [
        {"Name": "fraud_propensity_score",  "Type": "DECIMAL"},
        {"Name": "predicted_suspicious",    "Type": "INTEGER"},
        {"Name": "account_id",              "Type": "STRING"},
        {"Name": "card_type",               "Type": "STRING"},
        {"Name": "chargeback_count",        "Type": "INTEGER"},
        {"Name": "declined_transactions",   "Type": "INTEGER"},
        {"Name": "transaction_velocity",    "Type": "DECIMAL"},
        {"Name": "cross_border_ratio",      "Type": "DECIMAL"},
        {"Name": "rapid_succession_count",  "Type": "INTEGER"},
        {"Name": "unique_country_count",    "Type": "INTEGER"},
        {"Name": "risk_segment",            "Type": "STRING"},
        {"Name": "score_decile",            "Type": "INTEGER"},
        {"Name": "risk_exposure",           "Type": "DECIMAL"},
    ]

    physical_id = f"{QS_DS_INFERENCE}-physical"
    logical_id  = f"{QS_DS_INFERENCE}-logical"
    ds_name     = "Fraud Propensity Inference Output"

    physical_table_map = {
        physical_id: {
            "CustomSql": {
                "DataSourceArn": datasource_arn,
                "Name": ds_name,
                "SqlQuery": sql,
                "Columns": columns,
            }
        }
    }
    logical_table_map = {
        logical_id: {
            "Alias": ds_name,
            "Source": {"PhysicalTableId": physical_id},
        }
    }

    kwargs = dict(
        AwsAccountId=AWS_ACCOUNT_ID,
        DataSetId=QS_DS_INFERENCE,
        Name=ds_name,
        PhysicalTableMap=physical_table_map,
        LogicalTableMap=logical_table_map,
        ImportMode="DIRECT_QUERY",
        Permissions=_dataset_permissions(user_arn),
    )

    try:
        qs.describe_data_set(AwsAccountId=AWS_ACCOUNT_ID, DataSetId=QS_DS_INFERENCE)
        update_kwargs = {k: v for k, v in kwargs.items() if k != "Permissions"}
        qs.update_data_set(**update_kwargs)
        log(f"Updated dataset: {ds_name}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_data_set(**kwargs)
        log(f"Created dataset: {ds_name}")


# ═══════════════════════════════════════════════════════════
# SECTION 6 — Dashboard visual helpers
# ═══════════════════════════════════════════════════════════

_DS = "fraud"   # dataset alias used in DataSetIdentifierDeclarations


def _dataset_declarations():
    return [
        {"Identifier": _DS, "DataSetArn": f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:dataset/{QS_DS_INFERENCE}"},
    ]


def _col(col_name):
    return {"DataSetIdentifier": _DS, "ColumnName": col_name}


def _num_measure(field_id, col_name, agg="AVERAGE"):
    return {"NumericalMeasureField": {
        "FieldId": field_id,
        "Column": _col(col_name),
        "AggregationFunction": {"SimpleNumericalAggregation": agg},
    }}


def _num_dim(field_id, col_name):
    return {"NumericalDimensionField": {"FieldId": field_id, "Column": _col(col_name)}}


def _cat_dim(field_id, col_name):
    return {"CategoricalDimensionField": {"FieldId": field_id, "Column": _col(col_name)}}


def _title(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


def _subtitle(text):
    return {"Visibility": "VISIBLE", "FormatText": {"PlainText": text}}


# ── Sheet 1: Score Distribution ───────────────────────────

def _sheet1():
    histogram = {"BarChartVisual": {
        "VisualId": "bar-score-dist",
        "Title": _title("Fraud Score Distribution by Decile"),
        "Subtitle": _subtitle("Record count per fraud propensity score decile (1=lowest risk, 10=highest risk). A right-skewed distribution concentrates most accounts in low-risk deciles — expected in a healthy portfolio. High counts in decile 9–10 warrant immediate review."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("decile-dim", "score_decile")],
                    "Values":   [_num_measure("decile-cnt", "fraud_propensity_score", "COUNT")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "decile-dim", "Direction": "ASC"}}],
            },
        },
    }}

    donut = {"PieChartVisual": {
        "VisualId": "donut-risk-segments",
        "Title": _title("Accounts by Risk Segment"),
        "Subtitle": _subtitle("Proportion of accounts in High (score >0.7), Medium (0.3–0.7), and Low (<0.3) risk segments. High-segment accounts are candidates for enhanced due diligence or transaction monitoring."),
        "ChartConfiguration": {
            "FieldWells": {
                "PieChartAggregatedFieldWells": {
                    "Category": [_cat_dim("seg-dim", "risk_segment")],
                    "Values":   [_num_measure("seg-cnt", "fraud_propensity_score", "COUNT")],
                }
            },
            "DonutOptions": {"ArcOptions": {"ArcThickness": "MEDIUM"}},
        },
    }}

    decile_table = {"TableVisual": {
        "VisualId": "tbl-decile-lift",
        "Title": _title("Fraud Score Decile Lift Table"),
        "Subtitle": _subtitle("For each score decile: record count, average fraud propensity score, and suspicious flag rate. Higher deciles should show higher suspicious rates — this is the model's lift over random account selection."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_num_dim("lift-decile", "score_decile")],
                    "Values":  [
                        _num_measure("lift-cnt",   "fraud_propensity_score", "COUNT"),
                        _num_measure("lift-score", "fraud_propensity_score", "AVERAGE"),
                        _num_measure("lift-susp",  "predicted_suspicious",   "AVERAGE"),
                    ],
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "lift-decile", "Direction": "ASC"}}],
            },
        },
    }}

    susp_bar = {"BarChartVisual": {
        "VisualId": "bar-susp-vs-clean",
        "Title": _title("Avg Fraud Score: Suspicious vs Clean Accounts"),
        "Subtitle": _subtitle("Validates model discrimination: accounts flagged as suspicious (label=1) should have a materially higher average fraud propensity score than clean accounts (label=0). A clear gap confirms the model is separating risk correctly."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("susp-dim", "predicted_suspicious")],
                    "Values":   [_num_measure("susp-score", "fraud_propensity_score", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    return {
        "SheetId": "sheet-1",
        "Name": "Score Distribution",
        "Visuals": [histogram, donut, decile_table, susp_bar],
    }


# ── Sheet 2: Risk Breakdown ───────────────────────────────

def _sheet2():
    def _avg_score_bar(vid, title, subtitle, dim_field_id, dim_col):
        return {"BarChartVisual": {
            "VisualId": vid,
            "Title": _title(title),
            "Subtitle": _subtitle(subtitle),
            "ChartConfiguration": {
                "FieldWells": {
                    "BarChartAggregatedFieldWells": {
                        "Category": [_cat_dim(dim_field_id, dim_col)],
                        "Values":   [_num_measure(f"{vid}-val", "fraud_propensity_score", "AVERAGE")],
                    }
                },
                "Orientation": "HORIZONTAL",
                "SortConfiguration": {
                    "CategorySort": [{"FieldSort": {"FieldId": f"{vid}-val", "Direction": "DESC"}}],
                },
            },
        }}

    card_bar = _avg_score_bar(
        "bar-card-type",
        "Avg Fraud Score by Card Type",
        "Which card types are associated with the highest average fraud propensity. Prepaid and virtual cards typically show elevated risk — use this to calibrate card-type-specific monitoring thresholds.",
        "card-dim", "card_type",
    )

    chargeback_scatter = {"ScatterPlotVisual": {
        "VisualId": "scatter-chargeback-score",
        "Title": _title("Chargeback Count vs Fraud Propensity Score"),
        "Subtitle": _subtitle("Each point represents a group of accounts with similar chargeback counts. Accounts with more chargebacks should cluster toward higher fraud scores — this validates that the model correctly weights chargeback history as a fraud signal."),
        "ChartConfiguration": {
            "FieldWells": {
                "ScatterPlotCategoricallyAggregatedFieldWells": {
                    "XAxis":    [_num_measure("sc-x",    "chargeback_count",        "AVERAGE")],
                    "YAxis":    [_num_measure("sc-y",    "fraud_propensity_score",  "AVERAGE")],
                    "Category": [_cat_dim("sc-seg",      "risk_segment")],
                    "Size":     [_num_measure("sc-sz",   "fraud_propensity_score",  "COUNT")],
                }
            },
        },
    }}

    velocity_bar = {"BarChartVisual": {
        "VisualId": "bar-velocity-segment",
        "Title": _title("Avg Transaction Velocity by Risk Segment"),
        "Subtitle": _subtitle("Average daily transaction velocity (rolling 30-day) for each risk segment. High-risk accounts typically show elevated velocity — rapid transaction bursts are a key fraud indicator captured by the payment processor data."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("vel-seg", "risk_segment")],
                    "Values":   [_num_measure("vel-val", "transaction_velocity", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    crossborder_bar = {"BarChartVisual": {
        "VisualId": "bar-crossborder-segment",
        "Title": _title("Avg Cross-Border Ratio by Risk Segment"),
        "Subtitle": _subtitle("Average ratio of cross-border to domestic transactions per risk segment. High cross-border activity is a strong fraud signal — this chart shows whether the model correctly associates cross-border exposure with elevated risk scores."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("cb-seg", "risk_segment")],
                    "Values":   [_num_measure("cb-val", "cross_border_ratio", "AVERAGE")],
                }
            },
            "Orientation": "VERTICAL",
        },
    }}

    return {
        "SheetId": "sheet-2",
        "Name": "Risk Breakdown",
        "Visuals": [card_bar, chargeback_scatter, velocity_bar, crossborder_bar],
    }


# ── Sheet 3: Account & Card Analysis ─────────────────────

def _sheet3():
    segment_table = {"TableVisual": {
        "VisualId": "tbl-segment-summary",
        "Title": _title("Risk Segment Summary"),
        "Subtitle": _subtitle("Aggregated view of High, Medium, and Low risk segments. Shows record count, average fraud score, suspicious flag rate, average chargeback count, and average rapid succession events. Use filters to drill into specific card types or risk segments."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableAggregatedFieldWells": {
                    "GroupBy": [_cat_dim("seg-grp", "risk_segment")],
                    "Values":  [
                        _num_measure("seg-cnt",    "fraud_propensity_score",  "COUNT"),
                        _num_measure("seg-score",  "fraud_propensity_score",  "AVERAGE"),
                        _num_measure("seg-susp",   "predicted_suspicious",    "AVERAGE"),
                        _num_measure("seg-cb",     "chargeback_count",        "AVERAGE"),
                        _num_measure("seg-rapid",  "rapid_succession_count",  "AVERAGE"),
                    ],
                }
            },
        },
    }}

    top_records = {"TableVisual": {
        "VisualId": "tbl-top-risk-accounts",
        "Title": _title("Highest Risk Accounts"),
        "Subtitle": _subtitle("Individual account records ranked by fraud propensity score (highest first). Each row is one account-card combination. Use this list to prioritise accounts for manual review, enhanced monitoring, or transaction holds."),
        "ChartConfiguration": {
            "FieldWells": {
                "TableUnaggregatedFieldWells": {
                    "Values": [
                        {"FieldId": "tr-score",   "Column": _col("fraud_propensity_score")},
                        {"FieldId": "tr-susp",    "Column": _col("predicted_suspicious")},
                        {"FieldId": "tr-acct",    "Column": _col("account_id")},
                        {"FieldId": "tr-card",    "Column": _col("card_type")},
                        {"FieldId": "tr-cb",      "Column": _col("chargeback_count")},
                        {"FieldId": "tr-decl",    "Column": _col("declined_transactions")},
                        {"FieldId": "tr-vel",     "Column": _col("transaction_velocity")},
                        {"FieldId": "tr-rapid",   "Column": _col("rapid_succession_count")},
                        {"FieldId": "tr-country", "Column": _col("unique_country_count")},
                    ]
                }
            },
            "SortConfiguration": {
                "RowSort": [{"FieldSort": {"FieldId": "tr-score", "Direction": "DESC"}}],
            },
            "PaginatedReportOptions": {"VerticalOverflowVisibility": "VISIBLE"},
        },
    }}

    pivot = {"PivotTableVisual": {
        "VisualId": "pivot-segment-card",
        "Title": _title("Suspicious Rate: Risk Segment × Card Type"),
        "Subtitle": _subtitle("Cross-tab of risk segment (rows) vs card type (columns), showing average suspicious flag rate per cell. Identifies which segment + card type combinations carry the highest fraud density — useful for targeted intervention policies."),
        "ChartConfiguration": {
            "FieldWells": {
                "PivotTableAggregatedFieldWells": {
                    "Rows":    [_cat_dim("pt-seg",  "risk_segment")],
                    "Columns": [_cat_dim("pt-card", "card_type")],
                    "Values":  [_num_measure("pt-susp", "predicted_suspicious", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-3",
        "Name": "Account & Card Analysis",
        "Visuals": [segment_table, top_records, pivot],
    }

# ── Sheet 4: Business Impact ──────────────────────────────

def _sheet4():
    gains_bar = {"BarChartVisual": {
        "VisualId": "bar-suspicious-by-decile",
        "Title": _title("Suspicious Accounts Captured by Score Decile"),
        "Subtitle": _subtitle("How many flagged accounts (predicted_suspicious=1) fall in each score decile. A well-calibrated model concentrates suspicious accounts in the top deciles (9–10). This shows what fraction of all fraud you capture if you only review the top N deciles."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_num_dim("cg-decile", "score_decile")],
                    "Values":   [_num_measure("cg-susp", "predicted_suspicious", "SUM")],
                }
            },
            "Orientation": "VERTICAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "cg-decile", "Direction": "ASC"}}],
            },
        },
    }}

    exposure_bar = {"BarChartVisual": {
        "VisualId": "bar-risk-exposure-segment",
        "Title": _title("Estimated Risk Exposure by Segment"),
        "Subtitle": _subtitle("Estimated fraud exposure per segment, calculated as chargeback_count × fraud_propensity_score. High-segment accounts represent the largest potential loss — use this to size the business case for enhanced monitoring investment."),
        "ChartConfiguration": {
            "FieldWells": {
                "BarChartAggregatedFieldWells": {
                    "Category": [_cat_dim("exp-seg", "risk_segment")],
                    "Values":   [_num_measure("exp-val", "risk_exposure", "SUM")],
                }
            },
            "Orientation": "HORIZONTAL",
            "SortConfiguration": {
                "CategorySort": [{"FieldSort": {"FieldId": "exp-val", "Direction": "DESC"}}],
            },
        },
    }}

    heatmap = {"HeatMapVisual": {
        "VisualId": "heatmap-card-country",
        "Title": _title("Avg Fraud Score: Card Type × Unique Country Count"),
        "Subtitle": _subtitle("Heatmap of average fraud propensity score across card types and number of distinct countries transacted from. Darker cells indicate higher fraud risk. Use this to identify the most dangerous card-type + geographic-spread combinations for policy intervention."),
        "ChartConfiguration": {
            "FieldWells": {
                "HeatMapAggregatedFieldWells": {
                    "Rows":    [_cat_dim("hm-card",    "card_type")],
                    "Columns": [_num_dim("hm-country", "unique_country_count")],
                    "Values":  [_num_measure("hm-val", "fraud_propensity_score", "AVERAGE")],
                }
            },
        },
    }}

    return {
        "SheetId": "sheet-4",
        "Name": "Business Impact",
        "Visuals": [gains_bar, exposure_bar, heatmap],
    }


# ═══════════════════════════════════════════════════════════
# SECTION 7 — Build dashboard definition + filters
# ═══════════════════════════════════════════════════════════

def _build_definition():
    return {
        "DataSetIdentifierDeclarations": _dataset_declarations(),
        "Sheets": [_sheet1(), _sheet2(), _sheet3(), _sheet4()],
        "FilterGroups": [
            _filter_group("fg-card",    "card_type",    _DS),
            _filter_group("fg-segment", "risk_segment", _DS),
        ],
    }


def _filter_group(fg_id, col_name, ds_alias):
    """CategoryFilter group scoped to ALL_VISUALS on sheets 2–4."""
    return {
        "FilterGroupId": fg_id,
        "Filters": [{
            "CategoryFilter": {
                "FilterId": f"{fg_id}-filter",
                "Column": {"DataSetIdentifier": ds_alias, "ColumnName": col_name},
                "Configuration": {
                    "FilterListConfiguration": {
                        "MatchOperator": "CONTAINS",
                        "SelectAllOptions": "FILTER_ALL_VALUES",
                    }
                },
            }
        }],
        "ScopeConfiguration": {
            "SelectedSheets": {
                "SheetVisualScopingConfigurations": [
                    {"SheetId": sid, "Scope": "ALL_VISUALS"}
                    for sid in ["sheet-2", "sheet-3", "sheet-4"]
                ]
            }
        },
        "Status": "ENABLED",
        "CrossDataset": "SINGLE_DATASET",
    }


def _analysis_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeAnalysis",
            "quicksight:DescribeAnalysisPermissions",
            "quicksight:UpdateAnalysis",
            "quicksight:UpdateAnalysisPermissions",
            "quicksight:DeleteAnalysis",
            "quicksight:RestoreAnalysis",
            "quicksight:QueryAnalysis",
        ],
    }]


def _dashboard_permissions(user_arn):
    return [{
        "Principal": user_arn,
        "Actions": [
            "quicksight:DescribeDashboard",
            "quicksight:ListDashboardVersions",
            "quicksight:UpdateDashboardPermissions",
            "quicksight:QueryDashboard",
            "quicksight:UpdateDashboard",
            "quicksight:DeleteDashboard",
            "quicksight:DescribeDashboardPermissions",
            "quicksight:UpdateDashboardPublishedVersion",
        ],
    }]


# ═══════════════════════════════════════════════════════════
# SECTION 8 — Create analysis + publish dashboard
# ═══════════════════════════════════════════════════════════

def ensure_dashboard(user_arn):
    """Create or update the QuickSight analysis and dashboard."""
    print("\n[6/6] Creating QuickSight analysis and dashboard...")

    definition = _build_definition()
    dashboard_name = "Cross-Institution Fraud Propensity Scoring"

    # ── Analysis ──
    try:
        qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        qs.update_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name=dashboard_name,
            Definition=definition,
        )
        log(f"Updated analysis: {QS_ANALYSIS_ID}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_analysis(
            AwsAccountId=AWS_ACCOUNT_ID,
            AnalysisId=QS_ANALYSIS_ID,
            Name=dashboard_name,
            Definition=definition,
            Permissions=_analysis_permissions(user_arn),
        )
        log(f"Created analysis: {QS_ANALYSIS_ID}")

    _wait_for_analysis()

    # ── Dashboard ──
    publish_opts = {
        "AdHocFilteringOption":  {"AvailabilityStatus": "ENABLED"},
        "ExportToCSVOption":     {"AvailabilityStatus": "ENABLED"},
        "VisualPublishOptions":  {"ExportHiddenFieldsOption": {"AvailabilityStatus": "DISABLED"}},
    }
    try:
        qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        qs.update_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name=dashboard_name,
            Definition=definition,
            DashboardPublishOptions=publish_opts,
        )
        log(f"Updated dashboard: {QS_DASHBOARD_ID}")
        resp = qs.describe_dashboard(AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)
        latest = resp["Dashboard"]["Version"]["VersionNumber"]
        qs.update_dashboard_published_version(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            VersionNumber=latest,
        )
        log(f"Published dashboard version: {latest}")
    except ClientError as e:
        if e.response["Error"]["Code"] != "ResourceNotFoundException":
            raise
        qs.create_dashboard(
            AwsAccountId=AWS_ACCOUNT_ID,
            DashboardId=QS_DASHBOARD_ID,
            Name=dashboard_name,
            Definition=definition,
            Permissions=_dashboard_permissions(user_arn),
            DashboardPublishOptions=publish_opts,
        )
        log(f"Created dashboard: {QS_DASHBOARD_ID}")


def _wait_for_analysis(max_wait=120):
    for _ in range(max_wait // 5):
        resp = qs.describe_analysis(AwsAccountId=AWS_ACCOUNT_ID, AnalysisId=QS_ANALYSIS_ID)
        status = resp["Analysis"]["Status"]
        if status in ("CREATION_SUCCESSFUL", "UPDATE_SUCCESSFUL"):
            log(f"Analysis status: {status}")
            return
        if "FAILED" in status:
            errors = resp["Analysis"].get("Errors", [])
            raise RuntimeError(f"Analysis failed ({status}): {errors}")
        time.sleep(5)
    print("  WARNING: Analysis did not reach SUCCESSFUL status within timeout.")


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("AWS Clean Rooms ML FSI — Create QuickSight Dashboard")
    print("=" * 60)
    print(f"Account: {AWS_ACCOUNT_ID}  Region: {AWS_REGION}")
    print(f"Output bucket: {OUTPUT_BUCKET}")

    identity = sts.get_caller_identity()
    log(f"Authenticated as: {identity['Arn']}")

    ensure_quicksight_account()
    username = ensure_quicksight_user()
    user_arn = _qs_user_arn(username)

    prepare_glue_tables()
    ensure_quicksight_s3_access()
    ensure_lakeformation_permissions()
    ensure_datasource(user_arn)
    ensure_datasets(user_arn)
    ensure_dashboard(user_arn)

    dashboard_url = (
        f"https://{AWS_REGION}.quicksight.aws.amazon.com"
        f"/sn/dashboards/{QS_DASHBOARD_ID}"
    )
    print("\n" + "=" * 60)
    print("Dashboard ready!")
    print("=" * 60)
    print(f"\n  {dashboard_url}")
    print(f"\n  Note: If visuals show 'No data', wait ~2 min and refresh.")
    print(f"\nNext: open the URL above in your browser.")


if __name__ == "__main__":
    main()
