
import pandas as pd
from datetime import datetime, timedelta
from functions.slug import _slug
import random
from functions.athena_helpers import *
from functions.lambda_invoke import lambda_invoke
from functions.set_config import set_config
import time

FORECAST_PROXY_ARN, REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

def recommendations(st,tab_reco):
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
            st.clear()
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
            status_slot.info(f"original run is id - `{original}`")

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
                    print("df recommendations")
                    print(df)
                    if not df.empty:
                        render_by_category(df, new_id)
                    else:
                        status_slot.warning(f"Backend invoked, but Athena returned no rows for run_id `{new_id}` yet. Try Refresh again in a moment.")
                else:
                    status_slot.error("Could not determine a run_id from Athena after invoking backend.")
        else:
            # Initial/normal load: just show the latest available run
            rid = get_latest_run_id_cached()
            if rid is "":
                status_slot.error("Could not determine a run_id from Athena.")
            else:
                df = fetch_recs_by_run(rid)
                if df.empty:
                    status_slot.error(f"No recommendations found for run_id `{rid}`.")
                else:
                    render_by_category(df, rid)