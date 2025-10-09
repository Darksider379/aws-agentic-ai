import boto3
import json
import os
from functions.set_config import set_config

REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()


def lambda_invoke(function_arn: str, payload: dict, invocation_type: str = "RequestResponse"):

    lam = boto3.client("lambda", region_name=REGION)
    raw = lam.invoke(
        FunctionName=function_arn,
        InvocationType=invocation_type,
        Payload=json.dumps(payload).encode("utf-8"),
    )["Payload"].read()
    try:
        body = json.loads(raw)
        if isinstance(body, dict) and "body" in body:
            b = body["body"]
            return json.loads(b) if isinstance(b, str) else b
        return body
    except Exception:
        return {"_raw": raw.decode("utf-8", errors="ignore")}