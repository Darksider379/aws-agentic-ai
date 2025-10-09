import boto3
import re
import streamlit as st
from functions.set_config import set_config

REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()


def call_bedrock_agent(prompt: str) -> str:
    try:
        import boto3
        brt = boto3.client("bedrock-agent-runtime", region_name=REGION)
        resp = brt.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_ALIAS_ID,
            sessionId=st.session_state.session_id,
            inputText=prompt,
        )
        parts = []
        if "completion" in resp:
            for ev in resp["completion"]:
                if "chunk" in ev and "bytes" in ev["chunk"]:
                    parts.append(ev["chunk"]["bytes"].decode("utf-8"))
        if not parts and "outputText" in resp:
            parts.append(resp["outputText"])
        text = "".join(parts) if parts else "[No content returned by agent]"
        text = re.sub(r"<[^>]*REDACTED[^>]*>|<REDACTED>|\[?REDACTED\]?", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()
    except Exception as e:
        return f"[Agent call error] {e}"
