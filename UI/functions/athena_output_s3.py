# ---- Resolve Athena output S3 location (workgroup or env var) ----
from typing import Optional
from functions.set_config import set_config
import boto3

FORECAST_PROXY_ARN, REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

def _resolve_athena_output_s3(ath) -> Optional[str]:
 
    global _ATHENA_RESOLVED_OUTPUT
    if ATHENA_OUTPUT_S3:
        _ATHENA_RESOLVED_OUTPUT = ATHENA_OUTPUT_S3
        print(ATHENA_OUTPUT_S3)
        return _ATHENA_RESOLVED_OUTPUT
    try:
        ath = boto3.client("athena", region_name=REGION)
        wg = ath.get_work_group(Name=ATHENA_WORKGROUP)
        cfg = wg.get("WorkGroup", {}).get("Configuration", {})
        out = cfg.get("ResultConfiguration", {}).get("OutputLocation")
        if out:
            _ATHENA_RESOLVED_OUTPUT = out
            return _ATHENA_RESOLVED_OUTPUT
    except Exception:
        pass
        return None
