import pandas as pd
import boto3
import time
import os

class run_athena():
    def run_athena(sql: str) -> pd.DataFrame:
        qid = athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": ATHENA_DB},
            WorkGroup=ATHENA_WORKGROUP,
            ResultConfiguration={"OutputLocation": ATHENA_OUTPUT}
        )["QueryExecutionId"]

        # Wait for completion
        while True:
            q = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
            state = q["Status"]["State"]
            if state in ("SUCCEEDED","FAILED","CANCELLED"):
                if state != "SUCCEEDED":
                    raise RuntimeError(f"Athena failed: {q['Status']}")
                break
            time.sleep(0.5)

        # Paginate results
        res = athena.get_query_results(QueryExecutionId=qid, MaxResults=1000)
        cols = [c["VarCharValue"] for c in res["ResultSet"]["Rows"][0]["Data"]]
        rows = []
        for row in res["ResultSet"]["Rows"][1:]:
            rows.append([d.get("VarCharValue") for d in row["Data"]])
        # more pages
        token = res.get("NextToken")
        while token:
            res = athena.get_query_results(QueryExecutionId=qid, NextToken=token, MaxResults=1000)
            for row in res["ResultSet"]["Rows"]:
                rows.append([d.get("VarCharValue") for d in row["Data"]])
            token = res.get("NextToken")

        return pd.DataFrame(rows, columns=cols)