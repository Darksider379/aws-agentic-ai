# functions/run_athena.py
# -------------------------------------------------
# Generic Athena runner used by mainagent.py and helpers
# -------------------------------------------------
import os
import time
from typing import Optional, List
import boto3
import pandas as pd

# ====== Environment Variables ======
ATHENA_DB        = os.environ["ATHENA_DB"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT    = os.environ["ATHENA_OUTPUT"]
AWS_REGION       = os.environ.get("AWS_REGION", None)

# ====== AWS Client ======
athena = boto3.client("athena", region_name=AWS_REGION)

def run_athena(
    sql: str,
    database: str = ATHENA_DB,
    workgroup: str = ATHENA_WORKGROUP,
    output_s3: Optional[str] = ATHENA_OUTPUT,
    poll_sec: Optional[float] = None,
    max_wait_sec: Optional[int] = None,
    expect_result: bool = True,   # allow DDL/CTAS to skip fetching results
) -> Optional[pd.DataFrame]:
    """
    Execute an Athena query and return a Pandas DataFrame when expect_result=True.
    For DDL/CTAS/etc., set expect_result=False to skip fetching results (returns None).
    Raises RuntimeError on query failure.
    """
    if not output_s3:
        raise ValueError("ATHENA_OUTPUT (S3 output location) must be set in env.")

    # Tuneable locally
    if poll_sec is None:
        poll_sec = float(os.getenv("ATHENA_POLL_SEC", "1.5"))
    if max_wait_sec is None:
        max_wait_sec = int(os.getenv("ATHENA_MAX_WAIT_SEC", "90"))

    # ---- Start Query ----
    start = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_s3},
        WorkGroup=workgroup,
    )
    qid = start["QueryExecutionId"]

    # ---- Poll for completion ----
    waited = 0.0
    while waited < max_wait_sec:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state = status["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_sec)
        waited += poll_sec

    # ---- Final status ----
    status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
    if status["State"] != "SUCCEEDED":
        raise RuntimeError(f"Athena failed: {status}")

    if not expect_result:
        return None

    # ---- Fetch results ----
    res = athena.get_query_results(QueryExecutionId=qid, MaxResults=1000)
    rows = res.get("ResultSet", {}).get("Rows", [])
    if not rows:
        return pd.DataFrame()

    headers = [c.get("VarCharValue") for c in rows[0].get("Data", [])]
    records: List[list] = []
    for r in rows[1:]:
        records.append([(c.get("VarCharValue") if c else None) for c in r.get("Data", [])])

    # ---- Pagination ----
    token = res.get("NextToken")
    while token:
        res = athena.get_query_results(QueryExecutionId=qid, NextToken=token, MaxResults=1000)
        for r in res.get("ResultSet", {}).get("Rows", []):
            records.append([(c.get("VarCharValue") if c else None) for c in r.get("Data", [])])
        token = res.get("NextToken")

    return pd.DataFrame(records, columns=headers)

if __name__ == "__main__":
    try:
        df = run_athena("SELECT 1 AS test_col")
        print(df)
    except Exception as e:
        print("[athena] Error:", e)
