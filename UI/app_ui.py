# app.py
# -------------------------------------------------
# Agentic FinOps UI — Bedrock Agent + Lambda + Athena (resilient)
# -------------------------------------------------

import os
import uuid
import json
import time
import random
import re
from datetime import datetime, timedelta
from typing import Any, List, Tuple

import pandas as pd
import streamlit as st

# ========= App Config =========
st.set_page_config(page_title="Agentic FinOps", page_icon="💸", layout="wide")

# ========= Hidden Config / Constants =========
REGION = os.getenv("AWS_REGION", "us-east-1")

# Bedrock Agent (fixed per your env)
BEDROCK_AGENT_ID = "IO47D3HMWR"
BEDROCK_ALIAS_ID  = "OLYTGC8D37"

# Lambda ARNs (internal only; not shown in UI)
ARN_ANOMALY_PROXY = "arn:aws:lambda:us-east-1:784161806232:function:anomaly-http-proxy"
ARN_FINOPS_PROXY  = "arn:aws:lambda:us-east-1:784161806232:function:finops-agent-proxy"

# Athena settings (override via env if needed)
ATHENA_DB         = os.getenv("ATHENA_DB", "cost_agent_v2")
ATHENA_TABLE_RECS = os.getenv("ATHENA_RECS_TABLE", "recommendations_v2")
ATHENA_WORKGROUP  = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT_S3  = os.getenv("ATHENA_OUTPUT_S3", "s3://athena-query-results-agentic-ai/athena/")  # set "" to use WG default

# ========= Session =========
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat" not in st.session_state:
    st.session_state.chat = []  # [{role, text, ts}]
if "reco_df" not in st.session_state:
    st.session_state.reco_df = None
if "reco_last_run" not in st.session_state:
    st.session_state.reco_last_run = None
if "reco_render_seq" not in st.session_state:
    st.session_state.reco_render_seq = 0
if "anom_df" not in st.session_state:
    st.session_state.anom_df = None
if "anom_findings" not in st.session_state:
    st.session_state.anom_findings = None

# ========= Utilities =========
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(s).lower()).strip("-")

# ========= AWS Helpers =========
def call_bedrock_agent(prompt: str) -> str:
    """Invoke Bedrock Agent with streaming; sanitize output."""
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
        # scrub any redactions/guardrail placeholders
        text = text.replace("<REDACTED>", " ").strip()
        text = re.sub(r"\[?REDACTED\]?", " ", text)
        text = re.sub(r"<[^>]*REDACTED[^>]*>", " ", text)
        return re.sub(r"\s{2,}", " ", text).strip()
    except Exception as e:
        return f"[Agent call error] {e}"

def lambda_invoke(function_arn: str, payload: dict, invocation_type: str = "RequestResponse"):
    """Invoke Lambda and return parsed JSON (handles proxy/non-proxy formats)."""
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

def s3_read_json_uri(s3_uri: str) -> dict:
    """Read JSON from s3://bucket/key."""
    assert s3_uri.startswith("s3://"), f"Bad S3 URI: {s3_uri}"
    rest = s3_uri[5:]
    bucket, _, key = rest.partition("/")
    import boto3
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(obj["Body"].read())

def athena_query(sql: str, quiet: bool = False) -> Tuple[pd.DataFrame, bool, str]:
    """
    Run Athena query and return (DataFrame, ok:bool, reason:str).
    If quiet=True, no UI errors are shown (used during polling/first load).
    """
    import boto3
    ath = boto3.client("athena", region_name=REGION)

    start_kwargs = dict(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
    )
    if ATHENA_OUTPUT_S3:
        start_kwargs["ResultConfiguration"] = {"OutputLocation": ATHENA_OUTPUT_S3}

    try:
        qid = ath.start_query_execution(**start_kwargs)["QueryExecutionId"]
    except Exception as e:
        if not quiet:
            st.error(f"Athena start failed: {e}")
        return pd.DataFrame(), False, str(e)

    # Poll
    state = "QUEUED"
    reason = ""
    while True:
        time.sleep(0.4)
        meta = ath.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = meta["Status"]["State"]
        reason = meta["Status"].get("StateChangeReason", "")
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break

    if state != "SUCCEEDED":
        if not quiet:
            st.error(f"Athena query {state}: {reason or '(no reason)'}")
        return pd.DataFrame(), False, reason or state

    # Fetch (with pagination)
    res = ath.get_query_results(QueryExecutionId=qid)
    cols = [c["Name"] for c in res["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows = []

    def _consume(page, skip_header: bool):
        rs = page["ResultSet"]["Rows"]
        start = 1 if skip_header else 0
        for r in rs[start:]:
            vals = [f.get("VarCharValue") for f in r["Data"]]
            rows.append(vals)

    _consume(res, skip_header=True)
    token = res.get("NextToken")
    while token:
        res = ath.get_query_results(QueryExecutionId=qid, NextToken=token)
        _consume(res, skip_header=False)
        token = res.get("NextToken")

    return pd.DataFrame(rows, columns=cols), True, ""

# ========= Local utilities for charts =========
@st.cache_data(show_spinner=False)
def placeholder_cost_series(days=90, seed=21):
    random.seed(seed)
    today = datetime.utcnow().date()
    base = 250.0
    rows = []
    for i in range(days):
        d = today - timedelta(days=days - 1 - i)
        noise = random.uniform(-25, 25)
        spike = random.uniform(60, 150) if random.random() < 0.07 else 0
        rows.append({"date": d.isoformat(), "cost_usd": round(base + noise + spike, 2)})
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False)
def quick_forecast(df_cost: pd.DataFrame):
    df = df_cost.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    daily_avg = df["cost_usd"].mean()
    today = datetime.utcnow().date()
    # EOM
    eom = (datetime(today.year, today.month, 28) + timedelta(days=4)).date().replace(day=1) - timedelta(days=1)
    rem_eom = max(0, (eom - today).days)
    eom_proj = df["cost_usd"].sum() + daily_avg * rem_eom
    # EOQ
    q = (today.month - 1) // 3 + 1
    q_end_month = q * 3
    eoq = datetime(today.year, q_end_month, 28) + timedelta(days=4)
    eoq = eoq.date().replace(day=1) - timedelta(days=1)
    rem_eoq = max(0, (eoq - today).days)
    eoq_proj = df["cost_usd"].sum() + daily_avg * rem_eoq
    return {
        "daily_avg": round(float(daily_avg), 2),
        "eom_date": eom.isoformat(),
        "eom_projection_usd": round(float(eom_proj), 2),
        "eoq_date": eoq.isoformat(),
        "eoq_projection_usd": round(float(eoq_proj), 2),
    }

# ========= Header =========
left, right = st.columns([0.7, 0.3])
with left:
    st.title("Agentic FinOps Assistant")
    st.caption("Powered by AWS Bedrock Agents and Lambda backends.")
with right:
    st.metric("Session", st.session_state.session_id[:8])
    st.metric("Region", REGION)

# ========= Tabs =========
tab_chat, tab_reco, tab_anom, tab_fore, tab_cost = st.tabs(
    ["💬 Chat", "🛠️ Recommendations", "⚠️ Anomalies", "📈 Forecast", "💵 Costs"]
)

# =============================
# CHAT (ChatGPT-style ordering + clear + no redactions)
# =============================
with tab_chat:
    header_cols = st.columns([0.8, 0.2])
    with header_cols[0]:
        st.subheader("Chat with Agent")
    with header_cols[1]:
        if st.button("🧹 Clear chat", help="Clear the conversation history"):
            st.session_state.chat = []
            st.rerun()

    # Render history (oldest -> newest)
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            st.markdown(m["text"])

    user_msg = st.chat_input("Ask about cross-cloud pricing, anomalies, forecasting, rightsizing…")
    if user_msg:
        st.session_state.chat.append({"role": "user", "text": user_msg, "ts": time.time()})
        with st.chat_message("assistant"):
            placeholder = st.empty()
            buf = ""
            reply = call_bedrock_agent(user_msg)
            for ch in reply:
                buf += ch
                time.sleep(0.003)
                placeholder.markdown(buf)
        st.session_state.chat.append({"role": "assistant", "text": reply, "ts": time.time()})
        st.rerun()

# =============================
# RECOMMENDATIONS (S3 -> immediate, then auto-enrich from Athena)
# =============================

TABLE_FQN = f"{ATHENA_DB}.{ATHENA_TABLE_RECS}"

def _pretty_category_from_path(path: List[str]) -> str:
    s = "_".join(path).lower()
    if ("ec2" in s and ("right" in s or "size" in s)) or "rightsizing" in s:
        return "EC2 Right-size"
    if "s3" in s or "tier" in s:
        return "S3 Storage Optimization"
    if "snapshot" in s and "ebs" in s:
        return "Snapshot Hygiene"
    if "ebs" in s:
        return "EBS Optimization"
    if "lambda" in s:
        return "Lambda Optimization"
    if "cloudfront" in s:
        return "CloudFront Optimization"
    if "rds" in s:
        return "RDS Optimization"
    return "Other"

def _collect_tables(obj: Any, path: List[str], out_frames: List[pd.DataFrame]):
    """Recursively collect list-of-dict tables from any depth in the JSON."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _collect_tables(v, path + [k], out_frames)
    elif isinstance(obj, list) and obj and isinstance(obj[0], dict):
        df = pd.DataFrame(obj)
        if "category" not in df.columns:
            df["category"] = _pretty_category_from_path(path)
        out_frames.append(df)

def normalize_summary_to_df(summary_obj: dict) -> pd.DataFrame:
    if not isinstance(summary_obj, dict):
        return pd.DataFrame()

    frames: List[pd.DataFrame] = []
    # preferred: 'preview'
    prev = summary_obj.get("preview")
    if isinstance(prev, list) and prev and isinstance(prev[0], dict):
        dfp = pd.DataFrame(prev)
        if "category" not in dfp.columns:
            dfp["category"] = dfp.get("type", "Recommendations")
        frames.append(dfp)

    _collect_tables(summary_obj, [], frames)

    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Light tidy-ups
    rename = {
        "subtype": "proposal",
        "est_monthly_saving_usd": "est_saving_usd",
        "assumption": "assumption/hint",
        "action_sql_hint": "action_hint",
    }
    for k, v in rename.items():
        if k in df.columns and v not in df.columns:
            df[v] = df[k]
    return df

def compute_totals(df: pd.DataFrame) -> float:
    for col in ("est_monthly_saving_usd", "est_saving_usd", "estimated_saving_usd", "saving_usd"):
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()
    return 0.0

def get_latest_run_id_quiet() -> str | None:
    df, ok, _ = athena_query(f"SELECT MAX(run_id) AS run_id FROM {TABLE_FQN}", quiet=True)
    return (None if (not ok or df.empty) else df.iloc[0]["run_id"])

def load_recs_for_run_quiet(run_id: str) -> pd.DataFrame:
    sql = f"""
        SELECT *
        FROM {TABLE_FQN}
        WHERE run_id = '{run_id}'
        ORDER BY category, COALESCE(est_monthly_saving_usd, 0) DESC
    """
    df, ok, _ = athena_query(sql, quiet=True)
    return df if ok else pd.DataFrame()

def render_reco_tables(df_all: pd.DataFrame, pass_id: str = "p0"):
    # Topline
    total_save = compute_totals(df_all)
    st.metric("Estimated Total Monthly Saving (USD)", f"{total_save:,.2f}")
    if st.session_state.get("reco_last_run"):
        st.caption(f"Run ID: {st.session_state['reco_last_run']}")

    cats = sorted(df_all["category"].dropna().unique()) if "category" in df_all else ["All"]
    # Two columns per row
    for i in range(0, len(cats), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(cats):
                continue
            cat = cats[i + j]
            with cols[j]:
                st.markdown(f"**{cat}**")
                sdf = df_all[df_all["category"] == cat] if "category" in df_all else df_all.copy()

                # Pretty headers
                rename = {
                    "subtype": "proposal",
                    "est_monthly_saving_usd": "est_saving_usd",
                    "assumption": "assumption/hint",
                    "action_sql_hint": "action_hint",
                }
                for k, v in rename.items():
                    if k in sdf.columns and v not in sdf.columns:
                        sdf[v] = sdf[k]

                preferred = [
                    # EC2
                    "resource_id","current_instance","proposed_instance","region",
                    # S3
                    "bucket","prefix","current_class","proposed_class",
                    # EBS / Snapshots
                    "snapshot_id","volume_id","age_days",
                    # Lambda
                    "function_name","current_memory_mb","proposed_memory_mb","invocations",
                    # CloudFront
                    "distribution_id","current_tier","proposed_tier",
                    # RDS
                    "db_instance_id","current_class","proposed_class_rds",
                    # Common
                    "proposal","est_saving_usd","assumption/hint","action_hint",
                ]
                show_cols = [c for c in preferred if c in sdf.columns]
                if not show_cols:
                    cols_all = list(sdf.columns)
                    for front in ["proposal", "est_saving_usd"]:
                        if front in cols_all:
                            cols_all.insert(0, cols_all.pop(cols_all.index(front)))
                    show_cols = cols_all

                st.dataframe(
                    sdf[show_cols],
                    use_container_width=True,
                    height=320,
                    key=f"df-{_slug(cat)}-{pass_id}",
                )
                st.download_button(
                    f"Download {str(cat)} (CSV)",
                    sdf[show_cols].to_csv(index=False),
                    f"{_slug(cat)}.csv",
                    key=f"dl-{_slug(cat)}-{pass_id}",
                )

with tab_reco:
    st.subheader("Heuristics Recommendations")

    left_btns, right_btns = st.columns([0.4, 0.25])
    with left_btns:
        run_clicked = st.button("▶️ Run recommendations")
    with right_btns:
        refresh_clicked = st.button("🔄 Refresh (Athena)")

    # Placeholder we can update twice: first S3, then Athena-enriched
    render_slot = st.empty()

    if run_clicked:
        # 1) Kick proxy and render S3 immediately
        with st.spinner("Starting recommendations…"):
            res = lambda_invoke(
                ARN_FINOPS_PROXY,
                {"action": "recommendations", "use_llm": True},
                "RequestResponse",
            )

        started_run = res.get("run_id") if isinstance(res, dict) else None
        s3_uri     = res.get("summary_s3_uri") if isinstance(res, dict) else None

        if s3_uri:
            try:
                sj = s3_read_json_uri(s3_uri)
                df_flat = normalize_summary_to_df(sj.get("summary", sj))
                if not df_flat.empty:
                    # Show immediately from S3
                    st.session_state.reco_df = df_flat
                    st.session_state.reco_last_run = started_run or "(from S3)"
                    st.session_state.reco_render_seq += 1
                    with render_slot.container():
                        render_reco_tables(df_flat, pass_id=str(st.session_state.reco_render_seq))
                    st.success("Initial recommendations loaded from S3.")

                    # 2) Try a short, quiet enrichment from Athena (brings in EBS/Lambda/etc.)
                    with st.spinner("Enhancing with Athena results…"):
                        deadline = time.time() + 70  # up to ~70s
                        merged = df_flat.copy()
                        while time.time() < deadline:
                            latest = get_latest_run_id_quiet()
                            if latest and (not started_run or latest >= started_run):
                                df_ath = load_recs_for_run_quiet(latest)
                                if not df_ath.empty:
                                    st.session_state.reco_last_run = latest
                                    combo = pd.concat([merged, df_ath], ignore_index=True)
                                    dedup_key = combo.astype(str).agg("|".join, axis=1)
                                    combo = combo.loc[~dedup_key.duplicated()].reset_index(drop=True)
                                    merged = combo
                                    st.session_state.reco_df = merged
                                    st.session_state.reco_render_seq += 1
                                    with render_slot.container():
                                        render_reco_tables(merged, pass_id=str(st.session_state.reco_render_seq))
                                    st.toast("Athena categories added.", icon="✅")
                                    break
                            time.sleep(2.0)
                else:
                    st.warning("summary.json did not contain recognizable rows.")
            except Exception as e:
                st.error(f"Failed to read summary from S3: {e}")
        else:
            st.warning("Proxy did not return summary_s3_uri; use Refresh (Athena) to load the latest run.")

    if refresh_clicked:
        with st.spinner("Fetching latest run from Athena…"):
            latest = get_latest_run_id_quiet()
            if latest:
                df_all = load_recs_for_run_quiet(latest)
                if not df_all.empty:
                    st.session_state.reco_last_run = latest
                    st.session_state.reco_df = df_all
                    st.session_state.reco_render_seq += 1
                    with render_slot.container():
                        render_reco_tables(df_all, pass_id=str(st.session_state.reco_render_seq))
                else:
                    st.warning("Athena returned no rows for the latest run.")
            else:
                st.warning("Athena is not ready or database/table not found.")

    # First load or normal re-render
    if st.session_state.get("reco_df") is not None:
        st.session_state.reco_render_seq += 1
        with render_slot.container():
            render_reco_tables(st.session_state.reco_df, pass_id=str(st.session_state.reco_render_seq))
    else:
        with render_slot.container():
            st.info("Click **Run recommendations** to show results instantly (from S3). We’ll add more categories automatically once Athena completes. Or click **Refresh (Athena)** to pull the latest run directly.")

# =============================
# ANOMALIES (robust parsing; chart updates)
# =============================
with tab_anom:
    st.subheader("Cost Anomalies")

    if st.button("🔎 Fetch anomalies"):
        with st.spinner("Fetching anomalies from backend…"):
            payload = {"action": "anomalies"}
            res = lambda_invoke(ARN_ANOMALY_PROXY, payload, "RequestResponse")

            try:
                # Normalize into either a timeseries or findings table
                series = None
                findings = None

                if isinstance(res, dict) and "series" in res:
                    series = res["series"]
                elif isinstance(res, dict) and "items" in res:
                    items = res["items"]
                    if items and isinstance(items, list) and isinstance(items[0], dict):
                        if any(k in items[0] for k in ("date", "day", "ts")):
                            series = items
                        else:
                            findings = items
                elif isinstance(res, list):
                    items = res
                    if items and isinstance(items[0], dict):
                        if any(k in items[0] for k in ("date", "day", "ts")):
                            series = items
                        else:
                            findings = items

                df_series = None
                if series:
                    df_series = pd.DataFrame(series).copy()
                    if "date" not in df_series.columns:
                        if "day" in df_series: df_series["date"] = df_series["day"]
                        elif "ts" in df_series: df_series["date"] = pd.to_datetime(df_series["ts"]).dt.date.astype(str)
                    if "cost_usd" not in df_series.columns:
                        for alt in ("cost", "amount", "total_cost_usd", "value", "impact_usd"):
                            if alt in df_series:
                                df_series["cost_usd"] = pd.to_numeric(df_series[alt], errors="coerce")
                                break
                    df_series = df_series.dropna(subset=["date", "cost_usd"]).sort_values("date")
                    # z-scores for table
                    x = df_series["cost_usd"].astype(float)
                    mu, sd = x.mean(), max(x.std(ddof=0), 1e-9)
                    df_series["z"] = (x - mu) / sd
                    df_series["explain"] = df_series.apply(
                        lambda r: f"Cost {r.cost_usd:.2f} USD is {r.z:+.1f}σ from mean ({mu:.1f}).", axis=1
                    )

                df_findings = None
                if findings:
                    df_findings = pd.DataFrame(findings).copy()
                    rename = {
                        "explanation": "explain",
                        "service_name": "service",
                        "region_name": "region",
                        "severity_score": "severity",
                        "impact_usd": "cost_usd",
                    }
                    for k, v in rename.items():
                        if k in df_findings.columns and v not in df_findings.columns:
                            df_findings[v] = df_findings[k]

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
            df_cost = placeholder_cost_series(days=60, seed=9)
            st.line_chart(df_cost.set_index("date")["cost_usd"])
    with c2:
        if isinstance(df_series, pd.DataFrame) and not df_series.empty:
            st.dataframe(df_series[["date", "cost_usd", "z", "explain"]],
                         use_container_width=True, height=320)
        elif isinstance(df_findings, pd.DataFrame) and not df_findings.empty:
            cols = [c for c in ["service","region","cost_usd","severity","explain"] if c in df_findings.columns]
            st.dataframe(df_findings[cols], use_container_width=True, height=320)
        else:
            st.caption("Click “Fetch anomalies” to load backend data.")
            st.dataframe(pd.DataFrame(columns=["date","cost_usd","z","explain"]),
                         use_container_width=True, height=320)

# =============================
# FORECAST (visuals)
# =============================
with tab_fore:
    st.subheader("EOM / EOQ Forecast")
    df_cost = placeholder_cost_series(days=60, seed=12)
    fc = quick_forecast(df_cost)
    l, r = st.columns([0.5, 0.5])
    with l:
        st.metric("Daily Average (USD)", fc["daily_avg"])
        st.metric(f"EOM ({fc['eom_date']})", fc["eom_projection_usd"])
    with r:
        st.metric(f"EOQ ({fc['eoq_date']})", fc["eoq_projection_usd"])
    st.area_chart(df_cost.set_index("date")["cost_usd"])

# =============================
# COSTS (visuals)
# =============================
with tab_cost:
    st.subheader("Historical Cost")
    df_cost = placeholder_cost_series(days=90, seed=21)
    st.bar_chart(df_cost.set_index("date")["cost_usd"])
    with st.expander("Show raw data"):
        st.dataframe(df_cost, use_container_width=True, height=300)
    st.download_button("Download costs (CSV)", df_cost.to_csv(index=False), "cost_history.csv")

st.divider()

# =============================
# Roadmap placeholder (inline SVG logos with exact same size)
# =============================
gcp_col, azure_col = st.columns(2)

def svg_logo(html_svg: str):
    # Wrap to center and keep consistent size
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;">
            {html_svg}
        </div>
        """,
        unsafe_allow_html=True,
    )

GCP_SVG_128 = """
<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <rect width="128" height="128" rx="28" fill="#0b1220" />
  <g transform="translate(16,24)">
    <path d="M40 16c-9.94 0-18 8.06-18 18h8c0-5.52 4.48-10 10-10 5.52 0 10 4.48 10 10h8c0-9.94-8.06-18-18-18z" fill="#ea4335"/>
    <path d="M22 34a18 18 0 003.3 10.26L18 52.56A26 26 0 0114 34h8z" fill="#fbbc04"/>
    <path d="M45.7 58H28a26 26 0 01-10-5.44l7.3-8.3A18 18 0 0045.7 58z" fill="#34a853"/>
    <path d="M68 34a26 26 0 01-3.65 13.3L56.7 41.7A18 18 0 0060 34h8z" fill="#4285f4"/>
    <circle cx="40" cy="34" r="9" fill="#0b1220"/>
  </g>
</svg>
"""

AZURE_SVG_128 = """
<svg width="128" height="128" viewBox="0 0 128 128" xmlns="http://www.w3.org/2000/svg">
  <rect width="128" height="128" rx="28" fill="#0b1220" />
  <path d="M24 96 L64 24 L104 96 Z" fill="#0078d4"/>
  <path d="M44 96 L74 40 L84 60 L66 96 Z" fill="#36a3ff"/>
</svg>
"""

with gcp_col:
    svg_logo(GCP_SVG_128)
    st.markdown("**Coming soon**")
    st.caption("Recommendations, anomalies, and forecasting for GKE, Cloud Storage, and BigQuery.")

with azure_col:
    svg_logo(AZURE_SVG_128)
    st.markdown("**Coming soon**")
    st.caption("AKS, Blob Storage, and Azure SQL optimization are planned for upcoming releases.")

st.caption("© Agentic AI Hackathon • Bedrock Agents + Lambda + Athena")
