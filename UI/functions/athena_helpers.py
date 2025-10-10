import streamlit as st
import pandas as pd
from typing import Optional
from functions.athena_query import athena_query
from functions.set_config import set_config
from pandas.compat.numpy.function import validate_argmax

FORECAST_PROXY_ARN, REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()


@st.cache_data(show_spinner=False, ttl=30)
def get_latest_run_id_cached() -> Optional[str]:
    status_slot = st.empty()
    # CAST avoids type surprises; alias name is exactly 'run_id'
    sql = f"SELECT CAST(MAX(run_id) AS VARCHAR) AS run_id FROM {TABLE_FQN}"
    print(sql)
    df, ok, err = athena_query(sql, quiet=True)
    print(df)
    # if df.empty:
    #     return None
    #val = df.iloc[0].get("run_id")
    val = df.iloc[0, 0]
    print(validate_argmax)
    print("run_id : " +val)
    print("df : " +df)
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return status_slot.info("run_id not found - code - athena_helpers")
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
      action_sql_hint,
      rline_item_resource_id
    FROM {TABLE_FQN}
    WHERE run_id = '{rid}'
    ORDER BY category
    """
    print(sql)
    df, ok, err = athena_query(sql, quiet=True)
    if not ok:
        st.warning(f"Athena error: {err}")
        return pd.DataFrame()
    if not df.empty:
        # make printable and fill empties for display
        for c in df.columns:
            df[c] = df[c].astype(str).replace({"None": "", "null": ""})
    return df
