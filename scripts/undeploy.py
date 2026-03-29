# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
AWS Clean Rooms ML FSI Fraud Propensity Scoring — Undeploy / Teardown Script
Deletes all resources created by the demo scripts, in reverse dependency order.

Run with: python scripts/undeploy.py  (from the project root folder)

Add --dry-run to preview what would be deleted without actually deleting.
Add --skip-confirmation to skip the interactive confirmation prompt.
"""

import sys, os, argparse
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import *
validate()

import boto3, time

session     = boto3.Session(region_name=AWS_REGION)
iam         = session.client("iam")
glue        = session.client("glue")
cr          = session.client("cleanrooms")
crml        = session.client("cleanroomsml")
s3          = session.client("s3")
ecr         = session.client("ecr")
cb          = session.client("codebuild")
logs        = session.client("logs")
qs          = session.client("quicksight")
lf          = session.client("lakeformation")
sts_client  = session.client("sts")

DRY_RUN = False

IAM_ROLES = [
    ROLE_DATA_PROVIDER,
    ROLE_MODEL_PROVIDER,
    ROLE_ML_CONFIG,
    ROLE_QUERY_RUNNER,
    f"{PREFIX}-codebuild-role",
    f"{PREFIX}-sagemaker-role",
]

ECR_REPOS       = [f"{PREFIX}-training", f"{PREFIX}-inference"]
CODEBUILD_PROJECT = f"{PREFIX}-build"


def log(msg):
    prefix = "[DRY RUN] " if DRY_RUN else "  → "
    print(f"{prefix}{msg}")


def safe(fn, *args, **kwargs):
    """Call fn, swallowing NotFound-style exceptions."""
    try:
        if not DRY_RUN:
            return fn(*args, **kwargs)
    except Exception as e:
        err = str(e)
        benign = ["NoSuchEntity", "NoSuchBucket", "NotFoundException",
                  "ResourceNotFoundException", "RepositoryNotFoundException",
                  "EntityNotFoundException", "not found", "does not exist",
                  "NoSuchKey", "404"]
        if any(b.lower() in err.lower() for b in benign):
            log(f"  (already gone: {e.__class__.__name__})")
        else:
            log(f"  ERROR: {e}")


# ═══ 1. CLEAN ROOMS ML RESOURCES ═══
def delete_cleanrooms_ml():
    """Delete ML input channels, trained models, inference jobs, algorithm associations, and algorithms."""
    print("\n[1/8] Deleting Clean Rooms ML resources...")

    membership_id = None
    memberships = cr.list_memberships(status="ACTIVE")
    for m in memberships.get("membershipSummaries", []):
        if PREFIX in m.get("collaborationName", ""):
            membership_id = m["id"]
            break

    if not membership_id:
        log("No active membership found — skipping ML resource cleanup")
        return

    # Cancel/delete inference jobs
    try:
        jobs = crml.list_trained_model_inference_jobs(membershipIdentifier=membership_id)
        for job in jobs.get("trainedModelInferenceJobs", []):
            arn    = job["trainedModelInferenceJobArn"]
            status = job.get("status", "")
            log(f"Inference job: {job.get('name', arn)} ({status})")
            if status in ("CREATE_PENDING", "CREATE_IN_PROGRESS"):
                safe(crml.cancel_trained_model_inference_job,
                     membershipIdentifier=membership_id,
                     trainedModelInferenceJobArn=arn)
    except Exception as e:
        log(f"  Error listing inference jobs: {e}")

    # Delete trained models
    try:
        models = crml.list_trained_models(membershipIdentifier=membership_id)
        for m in models.get("trainedModels", []):
            arn = m["trainedModelArn"]
            log(f"Deleting trained model: {m.get('name', arn)}")
            safe(crml.delete_trained_model_output,
                 membershipIdentifier=membership_id,
                 trainedModelArn=arn)
    except Exception as e:
        log(f"  Error cleaning trained models: {e}")

    # Delete ML input channels
    try:
        channels = crml.list_ml_input_channels(membershipIdentifier=membership_id)
        for ch in channels.get("mlInputChannelsList", []):
            arn = ch["mlInputChannelArn"]
            log(f"Deleting ML input channel: {ch.get('name', arn)}")
            safe(crml.delete_ml_input_channel_data,
                 mlInputChannelArn=arn,
                 membershipIdentifier=membership_id)
    except Exception as e:
        log(f"  Error cleaning ML input channels: {e}")

    # Delete configured model algorithm associations
    try:
        assocs = crml.list_configured_model_algorithm_associations(membershipIdentifier=membership_id)
        for a in assocs.get("configuredModelAlgorithmAssociations", []):
            arn = a["configuredModelAlgorithmAssociationArn"]
            log(f"Deleting algorithm association: {a.get('name', arn)}")
            safe(crml.delete_configured_model_algorithm_association,
                 configuredModelAlgorithmAssociationArn=arn,
                 membershipIdentifier=membership_id)
    except Exception as e:
        log(f"  Error cleaning algorithm associations: {e}")

    # Delete configured model algorithms
    try:
        algos = crml.list_configured_model_algorithms()
        for algo in algos.get("configuredModelAlgorithms", []):
            if PREFIX in algo.get("name", ""):
                arn = algo["configuredModelAlgorithmArn"]
                log(f"Deleting configured model algorithm: {algo['name']}")
                safe(crml.delete_configured_model_algorithm,
                     configuredModelAlgorithmArn=arn)
    except Exception as e:
        log(f"  Error cleaning model algorithms: {e}")


# ═══ 2. CLEAN ROOMS COLLABORATION ═══
def delete_cleanrooms():
    """Delete configured table associations, configured tables, membership, and collaboration."""
    print("\n[2/8] Deleting Clean Rooms collaboration resources...")

    membership_id = None
    collab_id     = None
    memberships   = cr.list_memberships(status="ACTIVE")
    for m in memberships.get("membershipSummaries", []):
        if PREFIX in m.get("collaborationName", ""):
            membership_id = m["id"]
            collab_id     = m["collaborationId"]
            break

    if not membership_id:
        log("No active membership found — skipping collaboration cleanup")
        return

    log("Deleting ML configuration...")
    safe(crml.delete_ml_configuration, membershipIdentifier=membership_id)

    try:
        assocs = cr.list_configured_table_associations(membershipIdentifier=membership_id)
        for a in assocs.get("configuredTableAssociationSummaries", []):
            aid  = a["id"]
            name = a["name"]
            log(f"Deleting association analysis rule for: {name}")
            safe(cr.delete_configured_table_association_analysis_rule,
                 membershipIdentifier=membership_id,
                 configuredTableAssociationIdentifier=aid,
                 analysisRuleType="LIST")
            log(f"Deleting table association: {name}")
            safe(cr.delete_configured_table_association,
                 membershipIdentifier=membership_id,
                 configuredTableAssociationIdentifier=aid)
    except Exception as e:
        log(f"  Error cleaning table associations: {e}")

    try:
        tables = cr.list_configured_tables()
        for ct in tables.get("configuredTableSummaries", []):
            if PREFIX in ct.get("name", ""):
                ct_id = ct["id"]
                log(f"Deleting analysis rule for configured table: {ct['name']}")
                safe(cr.delete_configured_table_analysis_rule,
                     configuredTableIdentifier=ct_id,
                     analysisRuleType="LIST")
                log(f"Deleting configured table: {ct['name']}")
                safe(cr.delete_configured_table, configuredTableIdentifier=ct_id)
    except Exception as e:
        log(f"  Error cleaning configured tables: {e}")

    log(f"Deleting collaboration: {collab_id}")
    safe(cr.delete_collaboration, collaborationIdentifier=collab_id)


# ═══ 3. GLUE DATA CATALOG ═══
def delete_glue():
    """Delete Glue tables and database."""
    print("\n[3/8] Deleting Glue Data Catalog...")
    for tbl in [BANK_TABLE, PAYMENT_PROCESSOR_TABLE, "inference_output"]:
        log(f"Deleting table: {GLUE_DB}.{tbl}")
        safe(glue.delete_table, DatabaseName=GLUE_DB, Name=tbl)
    log(f"Deleting database: {GLUE_DB}")
    safe(glue.delete_database, Name=GLUE_DB)


# ═══ 4. LAKE FORMATION PERMISSIONS ═══
def delete_lake_formation_permissions():
    """Revoke Lake Formation permissions granted during setup and dashboard creation."""
    print("\n[4/8] Revoking Lake Formation permissions...")

    # Revoke data provider role permissions (granted by setup_cleanrooms.py)
    role_arn = f"arn:aws:iam::{AWS_ACCOUNT_ID}:role/{ROLE_DATA_PROVIDER}"
    safe(lf.revoke_permissions,
         Principal={"DataLakePrincipalIdentifier": role_arn},
         Resource={"Database": {"Name": GLUE_DB}},
         Permissions=["DESCRIBE"])
    log(f"Revoked DESCRIBE on database {GLUE_DB} from data provider role")

    for tbl in [BANK_TABLE, PAYMENT_PROCESSOR_TABLE]:
        safe(lf.revoke_permissions,
             Principal={"DataLakePrincipalIdentifier": role_arn},
             Resource={"Table": {"DatabaseName": GLUE_DB, "Name": tbl}},
             Permissions=["SELECT", "DESCRIBE"])
        log(f"Revoked SELECT+DESCRIBE on {tbl}")

    # Revoke QuickSight user permissions (granted by create_dashboard.py)
    try:
        identity   = sts_client.get_caller_identity()
        caller_arn = identity["Arn"]
        raw_name   = caller_arn.split(":")[-1]
        username   = "/".join(raw_name.split("/")[1:]) if raw_name.startswith("assumed-role/") else raw_name.split("/")[-1]
        qs_user_arn = f"arn:aws:quicksight:{AWS_REGION}:{AWS_ACCOUNT_ID}:user/default/{username}"

        safe(lf.revoke_permissions,
             Principal={"DataLakePrincipalIdentifier": qs_user_arn},
             Resource={"Database": {"Name": GLUE_DB}},
             Permissions=["DESCRIBE"])
        safe(lf.revoke_permissions,
             Principal={"DataLakePrincipalIdentifier": qs_user_arn},
             Resource={"Table": {"DatabaseName": GLUE_DB, "Name": "inference_output"}},
             Permissions=["SELECT", "DESCRIBE"])
        log(f"Revoked QuickSight user LF permissions")
    except Exception as e:
        log(f"  Could not revoke QuickSight LF permissions (non-fatal): {e}")


# ═══ 5. S3 BUCKETS ═══
def delete_s3_bucket(bucket_name):
    """Empty and delete an S3 bucket (handles versioned objects)."""
    log(f"Emptying bucket: {bucket_name}")
    try:
        paginator = s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket_name):
            objects = []
            for v in page.get("Versions", []):
                objects.append({"Key": v["Key"], "VersionId": v["VersionId"]})
            for dm in page.get("DeleteMarkers", []):
                objects.append({"Key": dm["Key"], "VersionId": dm["VersionId"]})
            if objects and not DRY_RUN:
                for i in range(0, len(objects), 1000):
                    s3.delete_objects(Bucket=bucket_name,
                                      Delete={"Objects": objects[i:i+1000], "Quiet": True})
        log(f"Deleting bucket: {bucket_name}")
        safe(s3.delete_bucket, Bucket=bucket_name)
    except Exception as e:
        if "NoSuchBucket" in str(e):
            log(f"  Bucket already gone: {bucket_name}")
        else:
            log(f"  Error deleting bucket {bucket_name}: {e}")


def delete_s3():
    print("\n[5/8] Deleting S3 buckets...")
    delete_s3_bucket(BUCKET)
    delete_s3_bucket(OUTPUT_BUCKET)


# ═══ 6. ECR REPOSITORIES ═══
def delete_ecr():
    print("\n[6/8] Deleting ECR repositories...")
    for repo in ECR_REPOS:
        log(f"Deleting ECR repo: {repo} (force — includes all images)")
        safe(ecr.delete_repository, repositoryName=repo, force=True)


# ═══ 7. IAM ROLES ═══
def delete_iam_role(role_name):
    """Delete inline policies then the role itself."""
    try:
        policies = iam.list_role_policies(RoleName=role_name)
        for pname in policies.get("PolicyNames", []):
            log(f"  Deleting inline policy: {pname}")
            safe(iam.delete_role_policy, RoleName=role_name, PolicyName=pname)
        attached = iam.list_attached_role_policies(RoleName=role_name)
        for p in attached.get("AttachedPolicies", []):
            log(f"  Detaching managed policy: {p['PolicyName']}")
            safe(iam.detach_role_policy, RoleName=role_name, PolicyArn=p["PolicyArn"])
        log(f"Deleting role: {role_name}")
        safe(iam.delete_role, RoleName=role_name)
    except iam.exceptions.NoSuchEntityException:
        log(f"  Role already gone: {role_name}")
    except Exception as e:
        log(f"  Error deleting role {role_name}: {e}")


def delete_iam():
    print("\n[7/8] Deleting IAM roles...")
    for role in IAM_ROLES:
        delete_iam_role(role)


# ═══ 8. CODEBUILD + CLOUDWATCH LOGS ═══
def delete_codebuild():
    print("\n[8/8] Deleting CodeBuild project and CloudWatch log groups...")
    log(f"Deleting CodeBuild project: {CODEBUILD_PROJECT}")
    safe(cb.delete_project, name=CODEBUILD_PROJECT)

    log_prefixes = [
        f"/aws/codebuild/{CODEBUILD_PROJECT}",
        "/aws/cleanrooms",
    ]
    for prefix in log_prefixes:
        try:
            paginator = logs.get_paginator("describe_log_groups")
            for page in paginator.paginate(logGroupNamePrefix=prefix):
                for lg in page.get("logGroups", []):
                    name = lg["logGroupName"]
                    log(f"Deleting log group: {name}")
                    safe(logs.delete_log_group, logGroupName=name)
        except Exception as e:
            log(f"  Error cleaning log groups with prefix {prefix}: {e}")


# ═══ 9. QUICKSIGHT RESOURCES ═══
def delete_quicksight():
    """Delete QuickSight dashboard, analysis, dataset, and data source (best-effort)."""
    print("\n[9/9] Deleting QuickSight resources (if present)...")

    log(f"Deleting QuickSight dashboard: {QS_DASHBOARD_ID}")
    safe(qs.delete_dashboard, AwsAccountId=AWS_ACCOUNT_ID, DashboardId=QS_DASHBOARD_ID)

    log(f"Deleting QuickSight analysis: {QS_ANALYSIS_ID}")
    safe(qs.delete_analysis,
         AwsAccountId=AWS_ACCOUNT_ID,
         AnalysisId=QS_ANALYSIS_ID,
         ForceDeleteWithoutRecovery=True)

    log(f"Deleting QuickSight dataset: {QS_DS_INFERENCE}")
    safe(qs.delete_data_set, AwsAccountId=AWS_ACCOUNT_ID, DataSetId=QS_DS_INFERENCE)

    log(f"Deleting QuickSight data source: {QS_DATASOURCE_ID}")
    safe(qs.delete_data_source, AwsAccountId=AWS_ACCOUNT_ID, DataSourceId=QS_DATASOURCE_ID)

    log("Removing QuickSight inline policy from service role")
    safe(iam.delete_role_policy,
         RoleName="aws-quicksight-service-role-v0",
         PolicyName="cleanrooms-ml-fsi-fraud-qs-access")


# ═══ MAIN ═══
def main():
    global DRY_RUN

    parser = argparse.ArgumentParser(description="Undeploy FSI Fraud Propensity Scoring demo resources")
    parser.add_argument("--dry-run", action="store_true", help="Preview deletions without executing them")
    parser.add_argument("--skip-confirmation", action="store_true", help="Skip interactive confirmation prompt")
    args   = parser.parse_args()
    DRY_RUN = args.dry_run

    print("=" * 60)
    print("AWS Clean Rooms ML FSI — UNDEPLOY")
    print("=" * 60)
    print(f"Account:  {AWS_ACCOUNT_ID}")
    print(f"Region:   {AWS_REGION}")
    print(f"Run ID:   {RUN_ID}")
    print(f"Buckets:  {BUCKET}, {OUTPUT_BUCKET}")
    if DRY_RUN:
        print("\n*** DRY RUN MODE — nothing will be deleted ***")

    if not args.skip_confirmation and not DRY_RUN:
        print("\nThis will PERMANENTLY DELETE all demo resources listed above.")
        confirm = input("Type 'yes' to confirm: ").strip().lower()
        if confirm != "yes":
            print("Aborted.")
            sys.exit(0)

    try:
        identity = sts_client.get_caller_identity()
        assert identity["Account"] == AWS_ACCOUNT_ID
        log(f"Authenticated as: {identity['Arn']}")
    except Exception as e:
        print(f"ERROR: AWS credentials not valid: {e}")
        sys.exit(1)

    delete_cleanrooms_ml()
    delete_cleanrooms()
    delete_glue()
    delete_lake_formation_permissions()
    delete_s3()
    delete_ecr()
    delete_iam()
    delete_codebuild()
    delete_quicksight()

    print("\n" + "=" * 60)
    if DRY_RUN:
        print("DRY RUN complete — no resources were deleted.")
    else:
        print("Undeploy complete! All demo resources have been deleted.")
    print("=" * 60)


if __name__ == "__main__":
    main()
