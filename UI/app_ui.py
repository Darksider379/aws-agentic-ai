# app.py
# -------------------------------------------------
# Agentic FinOps UI — Bedrock Agent + Lambda + Athena (direct-from-Athena)
# -------------------------------------------------

import os
import uuid
import json
import time
import random
import re
from datetime import datetime, timedelta
from typing import Any, List, Tuple, Optional

import pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Agentic FinOps", page_icon="💸", layout="wide")

# ========= Config =========
REGION = os.getenv("AWS_REGION", "us-east-1")

BEDROCK_AGENT_ID = "IO47D3HMWR"
BEDROCK_ALIAS_ID  = "OLYTGC8D37"

ARN_FINOPS_PROXY  = "arn:aws:lambda:us-east-1:784161806232:function:finops-agent-proxy"
ARN_ANOMALY_PROXY = "arn:aws:lambda:us-east-1:784161806232:function:anomaly-http-proxy"

ATHENA_DB         = os.getenv("ATHENA_DB", "synthetic_cur")
ATHENA_TABLE_RECS = os.getenv("ATHENA_RECS_TABLE", "recommendations_v2")
ATHENA_WORKGROUP  = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT_S3  = os.getenv("ATHENA_OUTPUT_S3", "").strip()  # e.g. s3://athena-query-results-agentic-ai/cost-agent-v2/recommendations/
TABLE_FQN = f"{ATHENA_DB}.{ATHENA_TABLE_RECS}"

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

# ========= Utils =========
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

def lambda_invoke(function_arn: str, payload: dict, invocation_type: str = "RequestResponse"):
    import boto3
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

# ---- Resolve Athena output S3 location (workgroup or env var) ----
_ATHENA_RESOLVED_OUTPUT = None
def _resolve_athena_output_s3(ath) -> Optional[str]:
    global _ATHENA_RESOLVED_OUTPUT
    if _ATHENA_RESOLVED_OUTPUT:
        return _ATHENA_RESOLVED_OUTPUT
    if ATHENA_OUTPUT_S3:
        _ATHENA_RESOLVED_OUTPUT = ATHENA_OUTPUT_S3
        return _ATHENA_RESOLVED_OUTPUT
    try:
        wg = ath.get_work_group(Name=ATHENA_WORKGROUP)
        cfg = wg.get("WorkGroup", {}).get("Configuration", {})
        out = cfg.get("ResultConfiguration", {}).get("OutputLocation")
        if out:
            _ATHENA_RESOLVED_OUTPUT = out
            return _ATHENA_RESOLVED_OUTPUT
    except Exception:
        pass
    return None

def athena_query(sql: str, quiet: bool = False) -> Tuple[pd.DataFrame, bool, str]:
    import boto3
    ath = boto3.client("athena", region_name=REGION)

    output_loc = _resolve_athena_output_s3(ath)
    start_kwargs = dict(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
    )
    if output_loc:
        start_kwargs["ResultConfiguration"] = {"OutputLocation": output_loc}

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

# ========= Athena helpers (robust run_id + fetch) =========
@st.cache_data(show_spinner=False, ttl=30)
def get_latest_run_id_cached() -> Optional[str]:
    # CAST avoids type surprises; alias name is exactly 'run_id'
    sql = f"SELECT CAST(MAX(run_id) AS VARCHAR) AS run_id FROM {TABLE_FQN}"
    df, ok, err = athena_query(sql, quiet=True)
    if not ok or df.empty:
        return None
    val = df.iloc[0].get("run_id")
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return str(val).strip() or None

def get_latest_run_id_uncached() -> Optional[str]:
    get_latest_run_id_cached.clear()
    return get_latest_run_id_cached()

def fetch_recs_by_run(run_id: str) -> pd.DataFrame:
    """Fetch exactly one run's rows; show only stable columns."""
    rid = (run_id or "").replace("'", "''")
    sql = f"""
    SELECT
      category,
      subtype,
      assumption,
      action_sql_hint
    FROM {TABLE_FQN}
    WHERE run_id = '{rid}'
    ORDER BY category
    """
    df, ok, err = athena_query(sql, quiet=True)
    if not ok:
        st.warning(f"Athena error: {err}")
        return pd.DataFrame()
    if not df.empty:
        # make printable and fill empties for display
        for c in df.columns:
            df[c] = df[c].astype(str).replace({"None": "", "null": ""})
    return df

# ========= Header =========
lh, rh = st.columns([0.7, 0.3])
with lh:
    st.title("Agentic FinOps Assistant")
    st.caption("Powered by AWS Bedrock Agents and Lambda backends.")
with rh:
    st.metric("Session", st.session_state.session_id[:8])
    st.metric("Region", REGION)

tab_chat, tab_reco, tab_anom, tab_fore, tab_cost = st.tabs(
    ["💬 Chat", "🛠️ Recommendations", "⚠️ Anomalies", "📈 Forecast", "💵 Costs"]
)

# ========= Chat =========
with tab_chat:
    h1, h2 = st.columns([0.8, 0.2])
    with h1:
        st.subheader("Chat with Agent")
    with h2:
        if st.button("🧹 Clear chat"):
            st.session_state.chat = []
            st.rerun()
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])
    q = st.chat_input("Ask about cross-cloud pricing, anomalies, forecasting, rightsizing…")
    if q:
        st.session_state.chat.append({"role":"user","text":q,"ts":time.time()})
        with st.chat_message("assistant"):
            ph = st.empty(); buf = ""; ans = call_bedrock_agent(q)
            for ch in ans:
                buf += ch; time.sleep(0.002); ph.markdown(buf)
        st.session_state.chat.append({"role":"assistant","text":ans,"ts":time.time()})
        st.rerun()

# ========= Recommendations (Athena) =========
with tab_reco:
    st.subheader("Heuristics Recommendations")

    left, right = st.columns([0.7, 0.3])
    with left:
        refresh_clicked = st.button("⟳ Refresh recommendations", help="Invoke backend and load the newest run")
    with right:
        clear_cache_clicked = st.button("🧹 Clear cache", help="Clear cached run_id and table")

    status_slot = st.empty()
    render_slot = st.empty()

    if clear_cache_clicked:
        get_latest_run_id_cached.clear()
        st.session_state["reco_df"] = None
        st.session_state["reco_last_run"] = None
        st.session_state["reco_render_seq"] = 0
        st.rerun()

    def render_by_category(df: pd.DataFrame, rid: str):
        st.session_state["reco_df"] = df.copy()
        st.session_state["reco_last_run"] = rid
        st.session_state["reco_render_seq"] = st.session_state.get("reco_render_seq", 0) + 1

        status_slot.info(f"Using run_id = `{rid}` • Rows = {len(df)}")
        with render_slot.container():
            cats = sorted(df["category"].fillna("Uncategorized").unique()) if "category" in df.columns else ["All"]
            cols_to_show = [c for c in ["subtype", "assumption", "action_sql_hint"] if c in df.columns]
            for i in range(0, len(cats), 2):
                c1, c2 = st.columns(2)
                for j, col in enumerate((c1, c2)):
                    if i + j >= len(cats):
                        continue
                    cat = cats[i + j]
                    with col:
                        st.markdown(f"**{cat}**")
                        sdf = df[df["category"] == cat] if "category" in df.columns else df
                        # nice display without literal 'None'
                        disp = sdf[cols_to_show].fillna("")
                        st.dataframe(disp, use_container_width=True, height=320,
                                     key=f"recs-{_slug(cat)}-{st.session_state['reco_render_seq']}")
                        st.download_button(
                            f"Download {cat} (CSV)",
                            disp.to_csv(index=False),
                            f"{_slug(cat)}.csv",
                            key=f"dl-{_slug(cat)}-{st.session_state['reco_render_seq']}",
                        )

    if refresh_clicked:
        # 1) record current latest run (may be None)
        original = get_latest_run_id_uncached() or ""

        # 2) invoke backend Lambda to generate a new run
        with st.spinner("Invoking backend and waiting for a new run…"):
            try:
                lambda_invoke(ARN_FINOPS_PROXY, {"action": "recommendations", "use_llm": True})
            except Exception as e:
                status_slot.error(f"Lambda invoke failed: {e}")

            # 3) poll Athena for a *new* run_id
            deadline = time.time() + 120  # up to 2 minutes
            new_id = None
            while time.time() < deadline:
                cand = get_latest_run_id_uncached()
                if cand and cand != original:
                    new_id = cand
                    break
                time.sleep(3)

            # 4) fall back to whatever the latest is if none detected
            if not new_id:
                new_id = get_latest_run_id_uncached() or original

            if new_id:
                df = fetch_recs_by_run(new_id)
                if not df.empty:
                    render_by_category(df, new_id)
                else:
                    status_slot.warning(f"Backend invoked, but Athena returned no rows for run_id `{new_id}` yet. Try Refresh again in a moment.")
            else:
                status_slot.error("Could not determine a run_id from Athena after invoking backend.")
    else:
        # Initial/normal load: just show the latest available run
        rid = get_latest_run_id_cached()
        if not rid:
            status_slot.error("Could not determine a run_id from Athena.")
        else:
            df = fetch_recs_by_run(rid)
            if df.empty:
                status_slot.error(f"No recommendations found for run_id `{rid}`.")
            else:
                render_by_category(df, rid)

# ========= Anomalies =========
with tab_anom:
    st.subheader("Cost Anomalies")
    if st.button("🔎 Fetch anomalies"):
        with st.spinner("Fetching anomalies…"):
            res = lambda_invoke(ARN_ANOMALY_PROXY, {"action":"anomalies"})
            try:
                series = None; findings = None
                if isinstance(res, dict) and "series" in res: series = res["series"]
                elif isinstance(res, dict) and "items" in res:
                    it = res["items"]
                    if it and isinstance(it, list) and isinstance(it[0], dict):
                        if any(k in it[0] for k in ("date","day","ts")): series = it
                        else: findings = it
                elif isinstance(res, list):
                    it = res
                    if it and isinstance(it[0], dict):
                        if any(k in it[0] for k in ("date","day","ts")): series = it
                        else: findings = it

                df_series = None
                if series:
                    df_series = pd.DataFrame(series).copy()
                    if "date" not in df_series.columns:
                        if "day" in df_series.columns: df_series["date"] = df_series["day"]
                        elif "ts" in df_series.columns: df_series["date"] = pd.to_datetime(df_series["ts"]).dt.date.astype(str)
                    if "cost_usd" not in df_series.columns:
                        for alt in ("cost","amount","total_cost_usd","value","impact_usd"):
                            if alt in df_series.columns:
                                df_series["cost_usd"] = pd.to_numeric(df_series[alt], errors="coerce")
                                break
                    df_series = df_series.dropna(subset=["date","cost_usd"]).sort_values("date")
                    x = df_series["cost_usd"].astype(float); mu = x.mean(); sd = max(x.std(ddof=0),1e-9)
                    df_series["z"] = (x - mu)/sd
                    df_series["explain"] = df_series.apply(lambda r: f"Cost {r.cost_usd:.2f} USD is {r.z:+.1f}σ from mean ({mu:.1f}).", axis=1)

                df_findings = None
                if findings:
                    df_findings = pd.DataFrame(findings).copy()
                    rename = {"explanation":"explain","service_name":"service","region_name":"region","severity_score":"severity","impact_usd":"cost_usd"}
                    for k,v in rename.items():
                        if k in df_findings.columns and v not in df_findings.columns: df_findings[v] = df_findings[k]

                st.session_state.anom_df = df_series
                st.session_state.anom_findings = df_findings
            except Exception as e:
                st.error(f"Could not parse anomalies response: {e}")

    df_series = st.session_state.get("anom_df")
    df_findings = st.session_state.get("anom_findings")
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        if isinstance(df_series, pd.DataFrame) and not df_series.empty:
            st.line_chart(df_series.set_index("date")["cost_usd"])
        else:
            today = datetime.utcnow().date()
            rows = [{"date": (today - timedelta(days=60-i)).isoformat(), "cost_usd": 250 + random.uniform(-25,25)} for i in range(60)]
            st.line_chart(pd.DataFrame(rows).set_index("date")["cost_usd"])
    with c2:
        if isinstance(df_series, pd.DataFrame) and not df_series.empty:
            st.dataframe(df_series[["date","cost_usd","z","explain"]], use_container_width=True, height=320)
        elif isinstance(df_findings, pd.DataFrame) and not df_findings.empty:
            cols = [c for c in ["service","region","cost_usd","severity","explain"] if c in df_findings.columns]
            st.dataframe(df_findings[cols], use_container_width=True, height=320)
        else:
            st.caption("Click “Fetch anomalies” to load backend data.")

# ========= Forecast =========
with tab_fore:
    st.subheader("EOM / EOQ Forecast")
    today = datetime.utcnow().date()
    rows = [{"date": (today - timedelta(days=60-i)).isoformat(), "cost_usd": 250 + random.uniform(-25,25)} for i in range(60)]
    df_cost = pd.DataFrame(rows)
    x = df_cost["cost_usd"].mean()
    eom = (datetime(today.year, today.month, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
    eoqm = ((today.month - 1)//3 + 1)*3
    eoq = (datetime(today.year, eoqm, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
    st.metric("Daily Average (USD)", round(float(x),2))
    st.metric(f"EOM ({eom.isoformat()})", round(float(df_cost["cost_usd"].sum() + x*max(0,(eom-today).days)),2))
    st.metric(f"EOQ ({eoq.isoformat()})", round(float(df_cost["cost_usd"].sum() + x*max(0,(eoq-today).days)),2))
    st.area_chart(df_cost.set_index("date")["cost_usd"])

# ========= Costs =========
with tab_cost:
    st.subheader("Historical Cost")
    today = datetime.utcnow().date()
    rows = [{"date": (today - timedelta(days=90-i)).isoformat(), "cost_usd": 250 + random.uniform(-25,25) + (150 if random.random()<0.05 else 0)} for i in range(90)]
    df_cost = pd.DataFrame(rows)
    st.bar_chart(df_cost.set_index("date")["cost_usd"])
    with st.expander("Show raw data"):
        st.dataframe(df_cost, use_container_width=True, height=300)
    st.download_button("Download costs (CSV)", df_cost.to_csv(index=False), "cost_history.csv")

st.divider()

# ========= Logos =========
ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
GCP_LOGO_PATH = os.path.join(ASSETS_DIR, "gcp.png")
AZURE_LOGO_PATH = os.path.join(ASSETS_DIR, "azure.png")
LOGO_BOX = 160  # px

def load_logo_square(path: str, box: int) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    img.thumbnail((box, box), Image.LANCZOS)
    canvas = Image.new("RGBA", (box, box), (0, 0, 0, 0))
    x = (box - img.width) // 2
    y = (box - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas

gcp_logo  = load_logo_square(GCP_LOGO_PATH, LOGO_BOX)
az_logo   = load_logo_square(AZURE_LOGO_PATH, LOGO_BOX)

col_gcp, col_az = st.columns(2)
with col_gcp:
    st.image(gcp_logo, use_container_width=False)
    st.markdown("**Coming soon**")
    st.caption("Recommendations, anomalies, and forecasting for GKE, Cloud Storage, and BigQuery.")
with col_az:
    st.image(az_logo, use_container_width=False)
    st.markdown("**Coming soon**")
    st.caption("AKS, Blob Storage, and Azure SQL optimization are planned for upcoming releases.")

st.caption("© Agentic AI Hackathon • Bedrock Agents + Lambda + Athena")
