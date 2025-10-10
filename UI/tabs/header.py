
from functions.set_config import set_config
import uuid
from tabs.anomalies import anomalies
from tabs.forecast import forecast
from tabs.chat import chat
from tabs.costs import cost
from tabs.recommendations import recommendations


FORECAST_PROXY_ARN, REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

def header(st):
    # ========= Session =========
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    for k, v in {
        "chat": [],
        "reco_df": None,
        "reco_last_run": None,
        "reco_render_seq": 0,
        "anom_df": None,
        "anom_findings": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

    lh, rh = st.columns([0.7, 0.3])
    with lh:
        st.title("Agentic FinOps Assistant")
        st.caption("Powered by AWS Bedrock Agents and Lambda backends.")
    with rh:
        st.metric("Session", st.session_state.session_id[:8])
        st.metric("Region", REGION)

    tab_chat, tab_reco, tab_fore, tab_cost = st.tabs(
        ["💬 Chat", "🛠️ Recommendations", "📈 Forecast", "💵 Costs/Anomalies"]
    )

    chat(st,tab_chat)
    #anomalies(st,tab_anom)
    cost(st,tab_cost)
    forecast(st,tab_fore)
    recommendations(st,tab_reco)
    
    