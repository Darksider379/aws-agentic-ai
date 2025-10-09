# functions/athena_strict.py
from __future__ import annotations
import time
import boto3
import pandas as pd
from dataclasses import dataclass


@dataclass
class AthenaMeta:
    query_id: str
    state: str
    reason: str
    sql_preview: str


def athena_client(region: str):
    return boto3.client("athena", region_name=region)


def athena_query_strict(
    *,
    sql: str,
    database: str,
    workgroup: str,
    output_s3: str,
    region: str,
    poll_sec: float = 0.6,
) -> tuple[pd.DataFrame, AthenaMeta]:
    """
    Run an Athena query and return (DataFrame, metadata).
    On failure raises RuntimeError **including the exact StateChangeReason**.
    """
    ath = athena_client(region)
    start = ath.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": database},
        WorkGroup=workgroup,
        ResultConfiguration={"OutputLocation": output_s3},
    )
    qid = start["QueryExecutionId"]

    # Poll until terminal state
    state = "RUNNING"
    reason = ""
    while True:
        exec_ = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = exec_["Status"]["State"]
        reason = exec_["Status"].get("StateChangeReason", "")
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll_sec)

    sql_preview = sql.strip().replace("\n", " ")
    if len(sql_preview) > 900:
        sql_preview = sql_preview[:900] + " …"

    if state != "SUCCEEDED":
        raise RuntimeError(
            f"Athena query failed (state={state}, qid={qid}). Reason: {reason or 'n/a'} | SQL: {sql_preview}"
        )

    # Fetch results (non-paginated path for small/medium result sets)
    res = ath.get_query_results(QueryExecutionId=qid)
    cols = [c["Label"] for c in res["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows = []
    for r in res["ResultSet"]["Rows"][1:]:
        rows.append([d.get(list(d.keys())[0], None) for d in r["Data"]])
    df = pd.DataFrame(rows, columns=cols)

    return df, AthenaMeta(query_id=qid, state=state, reason=reason, sql_preview=sql_preview)


def ts_expr_for_cur(column: str = "line_item_usage_start_date") -> str:
    """
    Robust timestamp parser for CUR 'line_item_usage_start_date' that can be:
    - TIMESTAMP already
    - ISO8601 text
    - 'YYYY-MM-DD HH:MM:SS'
    - 'YYYY-MM-DD HH:MM'
    - 'dd/MM/yy HH:MM'  (your sample)
    """
    return f"""
    COALESCE(
      TRY(CAST({column} AS TIMESTAMP)),
      TRY(from_iso8601_timestamp({column})),
      TRY(date_parse({column}, '%Y-%m-%d %H:%i:%s')),
      TRY(date_parse({column}, '%Y-%m-%d %H:%i')),
      TRY(date_parse({column}, '%d/%m/%y %H:%i'))
    )
    """.strip()
