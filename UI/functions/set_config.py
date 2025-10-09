import os

def set_config():
    REGION = os.getenv("AWS_REGION", "us-east-1")

    BEDROCK_AGENT_ID = "IO47D3HMWR"
    BEDROCK_ALIAS_ID  = "OLYTGC8D37"

    ARN_FINOPS_PROXY  = "arn:aws:lambda:us-east-1:784161806232:function:finops-agent-proxy"
    ARN_ANOMALY_PROXY = "arn:aws:lambda:us-east-1:784161806232:function:anomaly-http-proxy"

    ATHENA_DB         = os.getenv("ATHENA_DB", "synthetic_cur")
    ATHENA_TABLE_RECS = os.getenv("ATHENA_RECS_TABLE", "recommendations_v2")
    ATHENA_WORKGROUP  = os.getenv("ATHENA_WORKGROUP", "primary")
    ATHENA_OUTPUT_S3  = os.getenv("ATHENA_OUTPUT", "s3://athena-query-results-agentic-ai/athena/").strip()  # e.g. s3://athena-query-results-agentic-ai/cost-agent-v2/recommendations/
    TABLE_FQN = f"{ATHENA_DB}.{ATHENA_TABLE_RECS}"

    return REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN
