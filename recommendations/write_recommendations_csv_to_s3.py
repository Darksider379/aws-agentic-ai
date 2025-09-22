import pandas as pd
import boto3
import time
import os

class write_recommendations_csv_to_s3():
    def write_recommendations_csv_to_s3(recs: list) -> str:
        """Write CSV to s3://RESULTS_BUCKET/RESULTS_PREFIX/recommendations/{run_id}.csv and return LOCATION prefix."""
        if not recs:
            return f"s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/recommendations/"

        run_id = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
        key = f"{RESULTS_PREFIX}/recommendations/{run_id}.csv"

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["run_ts","category","subtype","region","assumption","metric","est_monthly_saving_usd","source_note"])
        now = dt.datetime.utcnow().isoformat()
        for r in recs:
            writer.writerow([now, r["category"], r["subtype"], r.get("region",""),
                            r.get("assumption",""), r.get("metric",""),
                            r["est_monthly_saving_usd"], "heuristics-v1"])
        s3.put_object(Bucket=RESULTS_BUCKET, Key=key, Body=buf.getvalue().encode("utf-8"))
        return f"s3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/recommendations/"
