# functions/run_athena.py
import os
import time
import boto3
import pandas as pd

ATHENA_DB        = os.environ["ATHENA_DB"]
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT    = os.environ["ATHENA_OUTPUT"]  # s3://.../athena/

athena = boto3.client("athena")


def run_athena(sql: str, expect_result: bool = True) -> pd.DataFrame | None:
    """
    Execute a SQL string in Athena.
    If expect_result=True, returns a pandas DataFrame.
    If expect_result=False (e.g. for DDL), waits for success and returns None.
    """
    qid = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )["QueryExecutionId"]

    # wait
    while True:
        q = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        s = q["Status"]["State"]
        if s in ("SUCCEEDED", "FAILED", "CANCELLED"):
            if s != "SUCCEEDED":
                raise RuntimeError(f"Athena failed: {q['Status']}")
            break
        time.sleep(0.5)

    if not expect_result:
        return None

    # first page
    res = athena.get_query_results(QueryExecutionId=qid, MaxResults=1000)
    if not res["ResultSet"]["Rows"]:
        return pd.DataFrame()

    cols = [c.get("VarCharValue", "") for c in res["ResultSet"]["Rows"][0]["Data"]]
    rows = []
    for row in res["ResultSet"]["Rows"][1:]:
        rows.append([d.get("VarCharValue") for d in row["Data"]])

    # paginate
    token = res.get("NextToken")
    while token:
        res = athena.get_query_results(QueryExecutionId=qid, NextToken=token, MaxResults=1000)
        for row in res["ResultSet"]["Rows"]:
            rows.append([d.get("VarCharValue") for d in row["Data"]])
        token = res.get("NextToken")

    return pd.DataFrame(rows, columns=cols)
