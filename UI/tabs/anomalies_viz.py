# tabs/anomalies_viz.py
from __future__ import annotations

from datetime import datetime, timedelta
import pandas as pd
import streamlit as st
import altair as alt

from functions.set_config import set_config
from functions.lambda_invoke import lambda_invoke
from functions.athena_strict import (
    athena_query_strict,
    ts_expr_for_cur,
    AthenaMeta,
)

# ---- Load config ----
(
    REGION,
    BEDROCK_AGENT_ID,
    BEDROCK_ALIAS_ID,
    ARN_FINOPS_PROXY,
    ARN_ANOMALY_PROXY,
    ATHENA_DB,
    ATHENA_TABLE_RECS,
    ATHENA_WORKGROUP,
    ATHENA_OUTPUT_S3,
    TABLE_FQN,
) = set_config()

ATHENA_TABLE_RAW = "raw"               # synthetic_cur.raw
ANOMALY_TABLE    = "cost_anomalies"    # synthetic_cur.cost_anomalies


def _sql_raw_daily(start_dt: datetime, end_dt: datetime) -> str:
    ts = ts_expr_for_cur("line_item_usage_start_date")
    return f"""
    WITH base AS (
      SELECT
        {ts}                                         AS ts,
        line_item_product_code                       AS service,
        CAST(line_item_unblended_cost AS DOUBLE)     AS cost
      FROM {ATHENA_DB}.{ATHENA_TABLE_RAW}
    )
    SELECT
      date_trunc('day', ts) AS usage_date,
      service,
      SUM(cost) AS cost
    FROM base
    WHERE ts IS NOT NULL
      AND ts >= TIMESTAMP '{start_dt.strftime("%Y-%m-%d")} 00:00:00'
      AND ts <  TIMESTAMP '{end_dt.strftime("%Y-%m-%d")} 00:00:00'
    GROUP BY 1,2
    ORDER BY 1,2
    """


def _sql_anomalies(start_dt: datetime, end_dt: datetime) -> str:
    return f"""
    SELECT
      CAST(usage_date AS DATE)       AS usage_date,
      service,
      region,
      CAST(cost AS DOUBLE)           AS cost,
      CAST(severity AS DOUBLE)       AS severity
    FROM {ATHENA_DB}.{ANOMALY_TABLE}
    WHERE usage_date >= DATE '{start_dt.strftime("%Y-%m-%d")}'
      AND usage_date <  DATE '{end_dt.strftime("%Y-%m-%d")}'
    ORDER BY 1,2
    """


def _service_chart(df_raw: pd.DataFrame, df_anom: pd.DataFrame, service: str) -> alt.Chart:
    dfr = df_raw[df_raw["service"] == service].copy()
    if dfr.empty:
        return alt.Chart(pd.DataFrame({"usage_date": [], "cost": []})).mark_line()

    dfr = dfr.sort_values("usage_date")
    dfr["cost_smooth"] = dfr["cost"].rolling(3, center=True, min_periods=1).mean()

    dfa = df_anom[df_anom["service"] == service].copy()
    anom_days = set(dfa["usage_date"].dt.date.tolist())
    non_anom = dfr[~dfr["usage_date"].dt.date.isin(anom_days)]
    baseline = (non_anom["cost"].median() if not non_anom.empty else dfr["cost"].median()) or 0.0

    line = alt.Chart(dfr).mark_line(stroke="#0b1a3a", strokeWidth=2).encode(
        x=alt.X("usage_date:T", axis=alt.Axis(title="")),
        y=alt.Y("cost_smooth:Q", axis=alt.Axis(title="")),
    )
    rule = alt.Chart(pd.DataFrame({"y": [baseline]})).mark_rule(color="#c62828", strokeWidth=2).encode(y="y:Q")
    pts = alt.Chart(dfa).mark_point(color="#e53935", size=60).encode(
        x="usage_date:T",
        y="cost:Q",
        tooltip=[
            "usage_date:T",
            "service:N",
            "region:N",
            alt.Tooltip("cost:Q", format="$.2f"),
            alt.Tooltip("severity:Q", format=".1f"),
        ],
    )
    labels = alt.Chart(dfa).mark_text(align="left", dx=6, dy=-6, color="#e53935", fontSize=11).encode(
        x="usage_date:T", y="cost:Q", text=alt.Text("severity:Q", format=".1f")
    )
    return (rule + line + pts + labels).properties(title=f"{service} — Cost vs Baseline", height=180) \
             .configure_axis(grid=True, gridOpacity=0.15)


def anomalies_viz(st, tab_anom):
    with tab_anom:
        st.subheader("Cost Anomalies (Strict Athena)")

        # Controls
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            days = st.number_input("Window (days)", min_value=7, max_value=120, value=30, step=1)
        with c2:
            fetch = st.button("🔎 Fetch anomalies", use_container_width=True)

        # 1) Optional: call Lambda to (re)compute anomalies
        if fetch:
            with st.spinner("Invoking anomaly Lambda…"):
                try:
                    lambda_invoke(ARN_ANOMALY_PROXY, {"action": "anomalies", "days": int(days)})
                    st.success("Anomaly Lambda finished.")
                except Exception as e:
                    st.error(f"Lambda invoke failed: {e}")

        # 2) Query Athena (strict helper shows real reasons when things break)
        end_dt = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=int(days))

        try:
            with st.spinner("Loading daily raw cost…"):
                df_raw, meta_raw = athena_query_strict(
                    sql=_sql_raw_daily(start_dt, end_dt + timedelta(days=1)),
                    database=ATHENA_DB,
                    workgroup=ATHENA_WORKGROUP,
                    output_s3=ATHENA_OUTPUT_S3,
                    region=REGION,
                )
        except Exception as e:
            st.error(str(e))
            st.stop()

        try:
            with st.spinner("Loading anomalies…"):
                df_anom, meta_anom = athena_query_strict(
                    sql=_sql_anomalies(start_dt, end_dt + timedelta(days=1)),
                    database=ATHENA_DB,
                    workgroup=ATHENA_WORKGROUP,
                    output_s3=ATHENA_OUTPUT_S3,
                    region=REGION,
                )
        except Exception as e:
            st.error(str(e))
            # Still render raw chart(s) without anomalies
            df_anom = pd.DataFrame(columns=["usage_date", "service", "region", "cost", "severity"])

        # Cast types
        if not df_raw.empty:
            df_raw["usage_date"] = pd.to_datetime(df_raw["usage_date"])
            df_raw["cost"] = pd.to_numeric(df_raw["cost"], errors="coerce").fillna(0.0)
        if not df_anom.empty:
            df_anom["usage_date"] = pd.to_datetime(df_anom["usage_date"])
            df_anom["cost"] = pd.to_numeric(df_anom["cost"], errors="coerce").fillna(0.0)
            df_anom["severity"] = pd.to_numeric(df_anom["severity"], errors="coerce").fillna(0.0)

        # Service selector
        if df_raw.empty:
            st.info("No raw cost rows for the selected window.")
            return

        services = sorted(df_raw["service"].unique().tolist())
        selected = st.multiselect("Services", services, default=services[: min(5, len(services))])

        # Charts
        if selected:
            charts = [_service_chart(df_raw, df_anom, svc) for svc in selected]
            st.altair_chart(alt.vconcat(*charts).resolve_scale(y="independent"), use_container_width=True)

        # Debug / preview
        with st.expander("Debug & previews"):
            colA, colB = st.columns(2)
            with colA:
                st.caption("Raw preview")
                st.dataframe(df_raw.head(200), use_container_width=True, height=280)
            with colB:
                st.caption("Anomalies preview")
                st.dataframe(df_anom.head(200), use_container_width=True, height=280)
