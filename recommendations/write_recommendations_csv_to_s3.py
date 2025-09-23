# recommendations/write_recommendations_csv_to_s3.py
import os
import io
import csv
import uuid
import datetime as dt
import boto3

RESULTS_BUCKET = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX = os.environ.get("RESULTS_PREFIX", "cost-agent")

s3 = boto3.client("s3")


def write_recommendations_csv_to_s3(recs: list[dict]) -> str:
    """
    Writes recommendations list to:
      s3://RESULTS_BUCKET/RESULTS_PREFIX/recommendations/<run-id>.csv
    Returns the S3 prefix (folder) used, which you can use as the Athena LOCATION.
    """
    prefix = f"{RESULTS_PREFIX}/recommendations/"
    s3_prefix_uri = f"s3://{RESULTS_BUCKET}/{prefix}"

    if not recs:
        # still return the prefix so the external table LOCATION is stable
        return s3_prefix_uri

    run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    key = f"{prefix}{run_id}.csv"

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(
        [
            "run_ts",
            "category",
            "subtype",
            "region",
            "assumption",
            "metric",
            "est_monthly_saving_usd",
            "source_note",
        ]
    )
    now = dt.datetime.utcnow().isoformat()
    for r in recs:
        w.writerow(
            [
                now,
                r.get("category", ""),
                r.get("subtype", ""),
                r.get("region", ""),
                r.get("assumption", ""),
                r.get("metric", ""),
                float(r.get("est_monthly_saving_usd", 0.0)),
                r.get("source_note", "heuristics-v1"),
            ]
        )

    s3.put_object(Bucket=RESULTS_BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))
    return s3_prefix_uri
