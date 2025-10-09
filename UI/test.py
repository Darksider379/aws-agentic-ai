import boto3
import time
import pandas as pd

sql = f"SELECT CAST(MAX(run_id) AS VARCHAR) AS run_id FROM  synthetic_cur.recommendations_v2 "
ath = boto3.client("athena", region_name="us-east-1")
start_kwargs = dict(
        QueryString=sql,
        QueryExecutionContext={"Database": "synthetic_cur"},
        WorkGroup="primary",
    )

start_kwargs["ResultConfiguration"] = {"OutputLocation": "s3://athena-query-results-agentic-ai/athena/"}


qid = ath.start_query_execution(**start_kwargs)["QueryExecutionId"]
# Poll
while True:
    time.sleep(0.35)
    meta = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
    state = meta["Status"]["State"]
    reason = meta["Status"].get("StateChangeReason", "")
    if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
        break

res = ath.get_query_results(QueryExecutionId=qid)
print(res)