# app.py
# -------------------------------------------------
# Agentic FinOps UI — Bedrock Agent + Lambda + Athena (direct-from-Athena)
# -------------------------------------------------

from datetime import datetime, timedelta
from typing import Any, List, Tuple, Optional

from numpy.ma import anomalies

from pandas.compat.numpy.function import validate_argmax
import streamlit as st

from functions.set_config import set_config
from tabs.header import header

from tabs.logos import logos

REGION, BEDROCK_AGENT_ID, BEDROCK_ALIAS_ID, ARN_FINOPS_PROXY, ARN_ANOMALY_PROXY, ATHENA_DB, ATHENA_TABLE_RECS, ATHENA_WORKGROUP, ATHENA_OUTPUT_S3, TABLE_FQN= set_config()

st.set_page_config(page_title="Agentic FinOps", page_icon="💸", layout="wide")

header(st)
logos(st)

