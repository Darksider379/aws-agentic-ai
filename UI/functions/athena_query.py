import boto3
import pandas as pd
from typing import Tuple
import boto3
import streamlit as st
from functions.athena_output_s3 import _resolve_athena_output_s3
from functions.set_config import set_config
import time

def athena_query(sql: str, quiet: bool = False) -> Tuple[pd.DataFrame, bool, str]:
    FORECAST_PROXY_ARN, REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

    ath = boto3.client("athena", region_name=REGION)

    output_loc = _resolve_athena_output_s3(ath)
    
    print(output_loc)
    start_kwargs = dict(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
        # QueryString=sql,
        # QueryExecutionContext={"Database": "synthetic_cur"},
        # WorkGroup="primary",
    )
    print(start_kwargs)
    if output_loc:
        start_kwargs["ResultConfiguration"] = {"OutputLocation": "s3://athena-query-results-agentic-ai/athena/"}
    # start_kwargs["ResultConfiguration"] = {"OutputLocation": "s3://athena-query-results-agentic-ai/athena/"}

    try:
        qid = ath.start_query_execution(**start_kwargs)["QueryExecutionId"]
    except Exception as e:
        if not quiet:
            st.error(
                "Athena error: "
                + str(e)
                + "\nTip: set ATHENA_OUTPUT_S3 to an S3 URI (e.g. s3://bucket/prefix/) or configure a Query result location in the workgroup."
            )
        return pd.DataFrame(), False, str(e)

    # Poll
    while True:
        time.sleep(0.35)
        meta = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = meta["Status"]["State"]
        reason = meta["Status"].get("StateChangeReason", "")
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break

    if state != "SUCCEEDED":
        if not quiet:
            st.error(f"Athena query {state}: {reason or '(no reason)'}")
        return pd.DataFrame(), False, reason or state

    # Collect results
    res = ath.get_query_results(QueryExecutionId=qid)
    cols = [c["Name"] for c in res["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows = []

    def _consume(page, skip_header=True):
        rs = page["ResultSet"]["Rows"]
        start = 1 if skip_header else 0
        for r in rs[start:]:
            rows.append([f.get("VarCharValue") for f in r["Data"]])

    _consume(res, skip_header=True)
    token = res.get("NextToken")
    while token:
        res = ath.get_query_results(QueryExecutionId=qid, NextToken=token)
        _consume(res, skip_header=False)
        token = res.get("NextToken")

    return pd.DataFrame(rows, columns=cols), True, ""

# sql = f"SELECT CAST(MAX(run_id) AS VARCHAR) AS run_id FROM synthetic_cur.recommendations_v2"
# df, bool1,var = athena_query(sql, True)
# print(df)