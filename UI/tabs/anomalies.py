
import pandas as pd
from datetime import datetime, timedelta
import random
from functions.lambda_invoke import lambda_invoke
from functions.set_config import set_config

REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

def anomalies(st,tab_anom):
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