# recommendations/write_recommendations_csv_to_s3.py
import os
import io
import csv
from typing import List, Dict
from datetime import datetime, timezone
import boto3

RESULTS_BUCKET = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")

s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", None))

# IMPORTANT: This order must match the Athena table columns exactly.
CSV_COLUMNS = [
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
    "rline_item_resource_id",  # <-- ensure this is written
]

def _coerce(v):
    if v is None:
        return ""
    if isinstance(v, (float, int)):
        # keep numbers compact
        return str(v)
    return str(v)

def write_recommendations_csv_to_s3(recs: List[Dict], run_id: str) -> str:
    """
    Write all recommendation rows to:
      s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/recommendations/run={run_id}/rows.csv
    with the header matching CSV_COLUMNS exactly.
    """
    # ensure required keys exist so DictWriter doesn't drop them
    rows = []
    for r in recs:
        row = {k: _coerce(r.get(k, "")) for k in CSV_COLUMNS}
        rows.append(row)

    # CSV in-memory
    buf = io.StringIO()
    w = csv.DictWriter(
        buf,
        fieldnames=CSV_COLUMNS,
        extrasaction="ignore",
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    w.writeheader()
    for row in rows:
        w.writerow(row)

    # put to S3 under a unique path per run
    key = f"{RESULTS_PREFIX.strip('/')}/recommendations/run={run_id}/rows.csv"
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=key,
        Body=buf.getvalue().encode("utf-8"),
        ContentType="text/csv",
        CacheControl="no-store, no-cache, must-revalidate",
    )
    return f"s3://{RESULTS_BUCKET}/{RESULTS_PREFIX.strip('/')}/recommendations/"
