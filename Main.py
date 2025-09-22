import os, time, uuid, io, csv, json, datetime as dt
import boto3
import pandas as pd

ATHENA_DB          = os.environ["ATHENA_DB"]
ATHENA_TABLE       = os.environ["ATHENA_TABLE"]
ATHENA_WORKGROUP   = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT      = os.environ["ATHENA_OUTPUT"]
RESULTS_BUCKET     = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX     = os.environ.get("RESULTS_PREFIX", "cost-agent")
ATHENA_RECS_TABLE  = os.environ.get("ATHENA_RECS_TABLE", "recommendations")

# Price knobs (can be overridden via env)
S3_STD   = float(os.environ.get("S3_STANDARD_PER_GB_MONTH", "0.023"))
S3_IA    = float(os.environ.get("S3_IA_PER_GB_MONTH", "0.0125"))
M5_LARGE = float(os.environ.get("EC2_M5_LARGE", "0.096"))
M7G_LARGE= float(os.environ.get("EC2_M7G_LARGE", "0.084"))
S3_RETR  = float(os.environ.get("S3_DEFAULT_RETRIEVAL_RATE", "0.10"))

athena = boto3.client("athena")
s3     = boto3.client("s3")
glue   = boto3.client("glue")

def handler(event=None, context=None):
    # 1) Pull usage
    hourly = hourly_usage_breakdown()

    # 2) Build recs
    recs = []
    recs += recommend_ec2_rightsize(hourly)
    recs += recommend_s3_tiering(hourly)
    recs += recommend_snapshot_hygiene(hourly)

    # 3) Write to S3 and (idempotently) ensure Athena table exists
    s3_prefix = write_recommendations_csv_to_s3(recs)
    build_recommendations_table_if_needed(s3_prefix)

    # 4) Return a small summary
    total = sum(r["est_monthly_saving_usd"] for r in recs)
    return {
        "count": len(recs),
        "est_total_monthly_saving_usd": round(total, 2),
        "athena_recs_table": ATHENA_RECS_TABLE,
        "s3_prefix": s3_prefix,
        "sample": recs[:5]
    }

if __name__ == "__main__":
    print(json.dumps(handler(), indent=2))
