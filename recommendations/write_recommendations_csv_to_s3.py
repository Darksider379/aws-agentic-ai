# recommendations/write_recommendations_csv_to_s3.py
import io
import csv
import time
import boto3
from typing import List, Dict, Optional
import os

s3 = boto3.client("s3")

def write_recommendations_csv_to_s3(recs: List[Dict], run_id: Optional[str] = None) -> str:
    """
    Write recommendations to S3 as CSV and return the S3 prefix URI.
    Adds 'run_id' and 'created_at' columns so Athena can filter the latest run.
    Keeps 'one_time_saving_usd' optional (defaults to 0.0).
    """
    if not recs:
        recs = []

    bucket = os.environ["RESULTS_BUCKET"]
    prefix = os.environ.get("RESULTS_PREFIX", "cost-agent")
    ts = int(time.time())
    # If run_id is provided, include it in the object key for easier browsing
    key = f"{prefix}/recommendations/{run_id or 'run'}/{ts}.csv"

    cols = [
        "run_id",
        "created_at",
        "category",
        "subtype",
        "region",
        "assumption",
        "metric",
        "est_monthly_saving_usd",
        "one_time_saving_usd",
        "action_sql_hint",
        "source_note",
    ]

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for r in recs:
        w.writerow([
            r.get("run_id", run_id or ""),
            r.get("created_at", ""),
            r.get("category", ""),
            r.get("subtype", ""),
            r.get("region", ""),
            r.get("assumption", ""),
            r.get("metric", ""),
            float(r.get("est_monthly_saving_usd", 0.0)),
            float(r.get("one_time_saving_usd", 0.0)),
            r.get("action_sql_hint", ""),
            r.get("source_note", ""),
        ])

    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"), ContentType="text/csv")
    return f"s3://{bucket}/{prefix}/recommendations/"
