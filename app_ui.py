# app.py
# -----------------------------
# Agentic FinOps UI (Mock-first)
# -----------------------------
# Runs fully offline with mock data. Later, plug in Bedrock/Lambda by
# implementing the "REAL INTEGRATION" stubs at the bottom.

import os
import uuid
import json
import time
import random
from datetime import datetime, timedelta

import pandas as pd
import streamlit as st

# -----------------------------
# App Config
# -----------------------------
st.set_page_config(page_title="Agentic FinOps (Mock)", page_icon="💸", layout="wide")

# -----------------------------
# Sidebar: Settings
# -----------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    use_mock = st.toggle("Use mock data", value=True, help="Turn off to try real integrations when you wire them up.")
    st.divider()
    st.subheader("AWS (for later)")
    region = st.text_input("Region", value=os.getenv("AWS_REGION", "us-east-1"))
    agent_id = st.text_input("Bedrock Agent ID", value=os.getenv("BEDROCK_AGENT_ID", "AGENT_ID_HERE"))
    alias_id = st.text_input("Agent Alias ID", value=os.getenv("BEDROCK_AGENT_ALIAS_ID", "ALIAS_ID_HERE"))
    st.caption("These are placeholders until you integrate Bedrock.")
    st.divider()
    st.subheader("Filters")
    lookback_days = st.slider("Lookback days", 7, 90, 14)
    cost_threshold = st.slider("Anomaly z-score threshold", 2, 6, 3)

# -----------------------------
# Session state
# -----------------------------
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "chat" not in st.session_state:
    st.session_state.chat = []  # list of dicts: {role, text, ts}

# -----------------------------
# Mock Data Generators
# -----------------------------
@st.cache_data(show_spinner=False)
def mock_cost_series(days=60, seed=7):
    random.seed(seed)
    today = datetime.utcnow().date()
    base = 250.0
    rows = []
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        noise = random.uniform(-25, 25)
        # occasional spike
        spike = 0
        if random.random() < 0.07:
            spike = random.uniform(60, 150)
        rows.append({"date": d.isoformat(), "cost_usd": round(base + noise + spike, 2)})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def mock_rightsize(n=8, seed=11):
    random.seed(seed)
    inst_types = ["m5.4xlarge", "m6i.2xlarge", "c6g.xlarge", "r6i.2xlarge"]
    props = ["m6i.2xlarge", "m6i.xlarge", "c7i.large", "r6i.xlarge"]
    out = []
    for i in range(n):
        cur = random.choice(inst_types)
        prop = random.choice(props)
        cur_cost = round(random.uniform(120, 380), 2)
        prop_cost = round(cur_cost * random.uniform(0.4, 0.8), 2)
        out.append({
            "resource_id": f"i-{random.randrange(10**12, 10**13-1)}",
            "current_instance": cur,
            "proposed_instance": prop,
            "current_monthly_usd": cur_cost,
            "proposed_monthly_usd": prop_cost,
            "savings_usd_month": round(cur_cost - prop_cost, 2),
            "confidence": round(random.uniform(0.6, 0.95), 2),
            "region": random.choice(["us-east-1", "us-west-2", "eu-west-1"]),
        })
    df = pd.DataFrame(out).sort_values("savings_usd_month", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(show_spinner=False)
def mock_s3_tiering(n=6, seed=13):
    random.seed(seed)
    classes = ["STANDARD", "STANDARD_IA", "INTELLIGENT_TIERING", "GLACIER", "DEEP_ARCHIVE"]
    out = []
    for i in range(n):
        cur = random.choice(classes[:3])
        prop = random.choice(classes[2:])
        gb = round(random.uniform(500, 8000), 1)
        save = round(gb * random.uniform(0.002, 0.02), 2)
        out.append({
            "bucket": f"company-bucket-{random.randint(10, 999)}",
            "prefix": f"project-{random.randint(1,9)}/data/",
            "current_class": cur,
            "proposed_class": prop,
            "storage_gb": gb,
            "est_monthly_savings_usd": save,
        })
    return pd.DataFrame(out).sort_values("est_monthly_savings_usd", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def mock_snapshot_hygiene(n=7, seed=17):
    random.seed(seed)
    out = []
    for i in range(n):
        days_old = random.randint(35, 280)
        out.append({
            "snapshot_id": f"snap-{random.randrange(10**9, 10**10-1)}",
            "volume_id": f"vol-{random.randrange(10**9, 10**10-1)}",
            "age_days": days_old,
            "region": random.choice(["us-east-1", "us-west-2", "ap-south-1"]),
            "action": "delete" if days_old > 90 else "keep",
            "est_monthly_cost_usd": round(random.uniform(3, 40), 2),
        })
    return pd.DataFrame(out).sort_values("age_days", ascending=False).reset_index(drop=True)

@st.cache_data(show_spinner=False)
def mock_anomalies(cost_df: pd.DataFrame, z_threshold=3):
    # simple z-score on cost_usd
    x = cost_df["cost_usd"]
    mu = x.mean()
    sd = max(x.std(ddof=0), 1e-9)
    cost_df = cost_df.copy()
    cost_df["z"] = (x - mu) / sd
    hits = cost_df[cost_df["z"].abs() >= z_threshold].copy()
    hits["explain"] = hits.apply(
        lambda r: f"Cost {r.cost_usd:.2f} USD is {r.z:+.1f}σ from mean ({mu:.1f}).", axis=1
    )
    return cost_df, hits

@st.cache_data(show_spinner=False)
def mock_forecast(cost_df: pd.DataFrame):
    # naive linear extrapolation for EOM/EOQ
    df = cost_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    # daily average trend (very simple)
    days_so_far = (df["date"].max() - df["date"].min()).days + 1
    daily_avg = df["cost_usd"].mean()
    today = datetime.utcnow().date()
    # EOM
    end_of_month = (datetime(today.year, today.month, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
    remaining_days_eom = (end_of_month - today).days
    eom_proj = df["cost_usd"].sum() + daily_avg * max(0, remaining_days_eom)
    # EOQ
    q = (today.month - 1)//3 + 1
    q_end_month = q*3
    end_of_quarter = datetime(today.year, q_end_month, 28) + timedelta(days=4)
    end_of_quarter = end_of_quarter.date().replace(day=1) - timedelta(days=1)
    remaining_days_eoq = (end_of_quarter - today).days
    eoq_proj = df["cost_usd"].sum() + daily_avg * max(0, remaining_days_eoq)
    return {
        "daily_avg": round(daily_avg, 2),
        "eom_date": end_of_month.isoformat(),
        "eom_projection_usd": round(float(eom_proj), 2),
        "eoq_date": end_of_quarter.isoformat(),
        "eoq_projection_usd": round(float(eoq_proj), 2),
    }

# -----------------------------
# Header / Hero
# -----------------------------
left, right = st.columns([0.7, 0.3])
with left:
    st.title("Agentic FinOps Assistant (Mock)")
    st.caption("Runs with synthetic data. You can wire Bedrock/Lambda later without changing the UI.")
with right:
    st.metric("Session", st.session_state.session_id[:8])
    st.metric("Region (target)", region)

# -----------------------------
# Tabs
# -----------------------------
tab_chat, tab_reco, tab_anom, tab_fore, tab_cost = st.tabs(
    ["💬 Chat", "🛠️ Recommendations", "⚠️ Anomalies", "📈 Forecast", "💵 Costs"]
)

# -----------------------------
# CHAT TAB
# -----------------------------
with tab_chat:
    st.subheader("Chat with Agent (mock)")
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])

    user_msg = st.chat_input("Ask about costs, rightsizing, S3 tiering…")
    if user_msg:
        st.session_state.chat.append({"role": "user", "text": user_msg, "ts": time.time()})
        # MOCK reply
        with st.chat_message("assistant"):
            placeholder = st.empty()
            reply = mock_agent_reply(user_msg)  # defined below
            # simulate streaming
            buf = ""
            for ch in reply:
                buf += ch
                time.sleep(0.01)
                placeholder.markdown(buf)
            st.session_state.chat.append({"role": "assistant", "text": reply, "ts": time.time()})

# -----------------------------
# RECOMMENDATIONS TAB
# -----------------------------
with tab_reco:
    st.subheader("Heuristics Recommendations (mock)")
    col1, col2, col3 = st.columns(3)
    df_r = mock_rightsize()
    df_s3 = mock_s3_tiering()
    df_snap = mock_snapshot_hygiene()

    with col1:
        st.markdown("**EC2 Rightsizing**")
        st.dataframe(df_r, use_container_width=True, height=300)
        st.download_button("Download EC2 recs (CSV)", df_r.to_csv(index=False), "ec2_rightsize.csv")

    with col2:
        st.markdown("**S3 Tiering**")
        st.dataframe(df_s3, use_container_width=True, height=300)
        st.download_button("Download S3 recs (CSV)", df_s3.to_csv(index=False), "s3_tiering.csv")

    with col3:
        st.markdown("**Snapshot Hygiene**")
        st.dataframe(df_snap, use_container_width=True, height=300)
        st.download_button("Download Snapshots (CSV)", df_snap.to_csv(index=False), "snapshot_hygiene.csv")

# -----------------------------
# ANOMALIES TAB
# -----------------------------
with tab_anom:
    st.subheader("Cost Anomalies (mock z-score)")
    df_cost = mock_cost_series(days=max(lookback_days, 30))
    scored, hits = mock_anomalies(df_cost, z_threshold=cost_threshold)

    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.line_chart(df_cost.set_index("date")["cost_usd"])
    with c2:
        st.caption(f"Threshold: |z| ≥ {cost_threshold}")
        st.dataframe(hits[["date", "cost_usd", "z", "explain"]], use_container_width=True, height=320)

# -----------------------------
# FORECAST TAB
# -----------------------------
with tab_fore:
    st.subheader("EOM / EOQ Forecast (mock)")
    df_cost = mock_cost_series(days=max(lookback_days, 30), seed=9)
    fc = mock_forecast(df_cost)
    l, r = st.columns([0.5, 0.5])
    with l:
        st.metric("Daily Average (USD)", fc["daily_avg"])
        st.metric(f"EOM ({fc['eom_date']})", fc["eom_projection_usd"])
    with r:
        st.metric(f"EOQ ({fc['eoq_date']})", fc["eoq_projection_usd"])
    st.area_chart(df_cost.set_index("date")["cost_usd"])

# -----------------------------
# COSTS TAB
# -----------------------------
with tab_cost:
    st.subheader("Historical Cost (mock)")
    df_cost = mock_cost_series(days=90, seed=21)
    st.bar_chart(df_cost.set_index("date")["cost_usd"])
    with st.expander("Show raw data"):
        st.dataframe(df_cost, use_container_width=True, height=300)
    st.download_button("Download costs (CSV)", df_cost.to_csv(index=False), "cost_history.csv")

st.divider()
st.caption("© Agentic AI Hackathon • Mock UI • Swap in real backends when ready.")

# -----------------------------
# MOCK "AGENT" REPLIES
# -----------------------------
def mock_agent_reply(user_text: str) -> str:
    """Lightweight mock to keep the chat alive without Bedrock."""
    t = user_text.lower().strip()
    if "rightsize" in t or "ec2" in t:
        top = mock_rightsize().head(3)
        bullets = "\n".join(
            f"- `{r.resource_id}` {r.current_instance} → {r.proposed_instance} "
            f"(save ${r.savings_usd_month}/mo, conf {r.confidence})"
            for _, r in top.iterrows()
        )
        return f"Here are the top EC2 rightsizing candidates (mock):\n{bullets}\n\nWant me to export the full list?"
    if "s3" in t or "tier" in t:
        s3 = mock_s3_tiering().head(3)
        bullets = "\n".join(
            f"- `{r.bucket}`/{r.prefix}: {r.current_class} → {r.proposed_class} "
            f"(est save ${r.est_monthly_savings_usd}/mo)"
            for _, r in s3.iterrows()
        )
        return f"S3 tiering suggestions (mock):\n{bullets}"
    if "anoma" in t or "spike" in t:
        df = mock_cost_series()
        _, hits = mock_anomalies(df, z_threshold=3)
        if hits.empty:
            return "No significant anomalies detected in the last window (mock)."
        row = hits.iloc[-1]
        return f"Anomaly on {row['date']}: cost ${row['cost_usd']} (z={row['z']:+.1f}). (mock)"
    if "forecast" in t or "eom" in t or "eoq" in t:
        fc = mock_forecast(mock_cost_series())
        return (f"EOM projection: ${fc['eom_projection_usd']} by {fc['eom_date']}\n"
                f"EOQ projection: ${fc['eoq_projection_usd']} by {fc['eoq_date']} (mock).")
    # default
    return "I can show rightsizing, S3 tiering, snapshot hygiene, anomalies, and EOM/EOQ forecasts — all mock data right now."

# -----------------------------
# REAL INTEGRATION (STUBS)
# -----------------------------
# When ready:
# 1) Set use_mock = False (sidebar).
# 2) Implement the functions below to call your real backends (API Gateway + Lambda or Bedrock Agents).
# 3) In tabs above, branch on use_mock to use these instead of mocks.

def call_bedrock_agent(prompt: str, agent_id: str, alias_id: str, region: str, session_id: str) -> str:
    """Replace this stub with bedrock-agent-runtime.invoke_agent(...) and streaming."""
    # Example (commented):
    # import boto3
    # brt = boto3.client("bedrock-agent-runtime", region_name=region)
    # resp = brt.invoke_agent(agentId=agent_id, agentAliasId=alias_id, sessionId=session_id, inputText=prompt)
    # text = []
    # for ev in resp.get("completion", []):
    #     if "chunk" in ev:
    #         text.append(ev["chunk"]["bytes"].decode("utf-8"))
    # return "".join(text)
    return "[REAL agent call not wired yet]"

def call_api_rightsize(lookback_days: int) -> pd.DataFrame:
    """Replace with requests.post('.../recommendations/ec2', json={...})."""
    return mock_rightsize()

def call_api_s3_tiering() -> pd.DataFrame:
    return mock_s3_tiering()

def call_api_snapshot_hygiene() -> pd.DataFrame:
    return mock_snapshot_hygiene()

def call_api_anomalies(z_threshold: int) -> pd.DataFrame:
    df = mock_cost_series()
    _, hits = mock_anomalies(df, z_threshold=z_threshold)
    return hits
