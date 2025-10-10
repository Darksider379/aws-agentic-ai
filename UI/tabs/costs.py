# tabs/costs.py
import os
import math
import pandas as pd
import streamlit as st
import altair as alt

# Use your existing Athena helper; we'll normalize its return.
try:
    from functions.athena_query import athena_query  # returns DataFrame or (DataFrame, ...)
except Exception:
    athena_query = None

# ---------- Config / ENV ----------
AWS_REGION        = os.getenv("AWS_REGION", "us-east-1")
ATHENA_DB         = os.getenv("ATHENA_DB", "synthetic_cur")
ATHENA_TABLE      = os.getenv("ATHENA_TABLE", "raw")  # CUR raw table
ATHENA_WORKGROUP  = os.getenv("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT     = os.getenv("ATHENA_OUTPUT", "s3://athena-query-results-agentic-ai/athena-output/")
DEFAULT_DAYS      = int(os.getenv("HISTORY_DAYS", "120"))
COST_MODE         = os.getenv("COST_MODE", "unblended").lower()

# Optional explicit overrides if you already know your schema
CUR_DATE_COL_ENV  = os.getenv("CUR_DATE_COL")   # e.g., line_item_usage_start_date
CUR_COST_COL_ENV  = os.getenv("CUR_COST_COL")   # e.g., line_item_unblended_cost

# Optional date format override for date_parse (Presto/Trino patterns)
# Examples: '%d/%m/%y %H:%i'  or  '%d/%m/%y'
CUR_DATE_FMT_ENV  = os.getenv("CUR_DATE_FMT")   # if set, we try date_parse({col}, CUR_DATE_FMT)

# Optional CSV fallback (local path; S3 not supported here)
HIST_COST_CSV     = os.getenv("HIST_COST_CSV", "")

# Candidate columns to probe in Glue information_schema
CANDIDATE_DATE_COLS = [
    "line_item_usage_start_date",
    "usage_start_date",
    "usage_start_time",
    "bill_billing_period_start_date",
]

# Candidate cost columns grouped by "mode" — add custom fields here if needed
CANDIDATE_COSTS = {
    "unblended": [
        "line_item_unblended_cost",
        "unblended_cost",
        "net_unblended_cost",
        "total_cost_usd",
        "total_cost",
        "amount",
        "cost",
    ],
    "blended": [
        "line_item_blended_cost",
        "blended_cost",
        "total_cost_usd",
        "total_cost",
        "amount",
        "cost",
    ],
    "amortized": [
        "amortized_cost",
        "reservation_amortized_upfront_fee_for_billing_period",
        "line_item_unblended_cost",
        "total_cost_usd",
        "total_cost",
        "amount",
        "cost",
    ],
    "net_unblended": [
        "net_unblended_cost",
        "line_item_unblended_cost",
        "total_cost_usd",
        "total_cost",
        "amount",
        "cost",
    ],
}


# ---------- Helpers ----------
def _run_sql(sql: str) -> pd.DataFrame:
    """Run SQL through your athena_query wrapper, normalizing return shape."""
    if athena_query is None:
        raise RuntimeError("functions.athena_query.athena_query not found")
    out = athena_query(sql)
    if isinstance(out, tuple):  # tolerate (df, qid, …)
        out = out[0]
    return out if isinstance(out, pd.DataFrame) else pd.DataFrame()


def _present_columns() -> list[str]:
    """Return list of lowercase columns present in the target table."""
    cols_sql = f"""
    SELECT LOWER(column_name) AS name
    FROM information_schema.columns
    WHERE LOWER(table_schema) = LOWER('{ATHENA_DB}')
      AND LOWER(table_name)   = LOWER('{ATHENA_TABLE}')
    ORDER BY 1
    """
    try:
        df = _run_sql(cols_sql)
        return list(df["name"].astype(str).str.lower()) if not df.empty else []
    except Exception:
        return []


def _mode_key() -> str:
    if COST_MODE == "amortized":
        return "amortized"
    if COST_MODE in ("net", "net_unblended"):
        return "net_unblended"
    if COST_MODE == "blended":
        return "blended"
    return "unblended"


def _pick_date_col(present: list[str]) -> str:
    if CUR_DATE_COL_ENV and CUR_DATE_COL_ENV.lower() in present:
        return CUR_DATE_COL_ENV
    for c in CANDIDATE_DATE_COLS:
        if c.lower() in present:
            return c
    # Final fallback — adjust if your CUR differs
    return "line_item_usage_start_date"


def _pick_cost_col(present: list[str]) -> str:
    if CUR_COST_COL_ENV and CUR_COST_COL_ENV.lower() in present:
        return CUR_COST_COL_ENV
    # Try mode-specific list first
    for c in CANDIDATE_COSTS.get(_mode_key(), []):
        if c.lower() in present:
            return c
    # Then any known candidate
    for c in sum(CANDIDATE_COSTS.values(), []):
        if c.lower() in present:
            return c
    # Last resort — keeps SQL valid even if it fails (better error msg)
    return "line_item_unblended_cost"


def _date_cast_expr(date_col: str) -> str:
    """
    Build a robust Presto/Trino expression to cast CUR date strings to date,
    handling DD/MM/YY and other common patterns.
    Returns an expression that yields a DATE.
    """
    parts = []
    parts.append(f"try_cast({date_col} AS timestamp)")
    parts.append(f"try(date_parse({date_col}, '%d/%m/%y %H:%i'))")
    parts.append(f"try(date_parse({date_col}, '%d/%m/%Y %H:%i:%s'))")
    parts.append(f"try(date_parse({date_col}, '%d/%m/%Y %H:%i'))")
    parts.append(f"try(date_parse({date_col}, '%Y-%m-%d %H:%i:%s'))")
    parts.append(f"try(date_parse({date_col}, '%d/%m/%y'))")
    parts.append(f"try(date_parse({date_col}, '%d/%m/%Y'))")
    parts.append(f"try(date_parse({date_col}, '%Y-%m-%d'))")

    if CUR_DATE_FMT_ENV:
        parts.insert(1, f"try(date_parse({date_col}, '{CUR_DATE_FMT_ENV}'))")

    coalesce_ts = "coalesce(" + ", ".join(parts) + ")"
    return f"CAST(date_trunc('day', {coalesce_ts}) AS date)"


def _query_daily(date_col: str, cost_col: str, days: int | None) -> pd.DataFrame:
    """
    If `days` is provided -> filter last N days.
    If `days` is None    -> no date filter; return full series available (may be large).
    """
    date_expr = _date_cast_expr(date_col)
    date_filter = (
        f"WHERE {date_expr} >= date_add('day', -{int(days) + 2}, current_date)"
        if days is not None else ""
    )
    sql = f"""
    WITH base AS (
      SELECT
        {date_expr} AS dt,
        try_cast({cost_col} AS double) AS cost
      FROM {ATHENA_DB}.{ATHENA_TABLE}
      {date_filter}
    )
    SELECT dt AS date, SUM(COALESCE(cost,0)) AS cost_usd
    FROM base
    WHERE dt IS NOT NULL
    GROUP BY 1
    ORDER BY 1
    """
    return _run_sql(sql)


def _fetch_daily_cost(n_days: int,
                      chosen_date_col: str | None = None,
                      chosen_cost_col: str | None = None) -> tuple[pd.DataFrame, str, str, str]:
    """
    Return (DataFrame, picked_date_col, picked_cost_col, fetch_mode)
    fetch_mode: 'window' (requested N days) or 'fallback_all' (we pulled all, then clipped).
    """
    present = _present_columns()
    date_col = chosen_date_col or _pick_date_col(present)
    cost_col = chosen_cost_col or _pick_cost_col(present)

    # 1) Try last N days
    df = _query_daily(date_col, cost_col, n_days)
    mode_used = "window"

    # 2) If empty, try without time filter (pull what's available, then clip)
    if df is None or df.empty:
        df = _query_daily(date_col, cost_col, None)
        mode_used = "fallback_all"

    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "cost_usd"]), date_col, cost_col, mode_used

    df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
    df = df.dropna(subset=["cost_usd"]).sort_values("date")
    df = df.tail(int(n_days)).reset_index(drop=True)
    return df, date_col, cost_col, mode_used


def _detect_spikes(df: pd.DataFrame, z_thresh: float = 2.0, roll: int = 7) -> pd.DataFrame:
    """Flag spikes using rolling z-score."""
    if df.empty:
        return df.assign(is_spike=False)
    t = df.copy()
    t["date_dt"] = pd.to_datetime(t["date"])
    t = t.sort_values("date_dt")
    t["roll_mean"] = t["cost_usd"].rolling(roll, min_periods=max(2, roll // 2)).mean()
    t["roll_std"]  = t["cost_usd"].rolling(roll, min_periods=max(2, roll // 2)).std(ddof=0)
    t["z"] = (t["cost_usd"] - t["roll_mean"]) / t["roll_std"]
    t["is_spike"] = (t["z"] > z_thresh).fillna(False)
    return t


# ---------- Tab renderer (Forecast-style template) ----------
def cost(st, tab_cost):
    with tab_cost:
        st.subheader("Historical Cost")

        # Controls like Forecast: number input + big stretch button
        c1, c2 = st.columns([1, 3])
        with c1:
            days = st.number_input("Horizon (days)", min_value=14, max_value=365,
                                   value=DEFAULT_DAYS, step=1)
            fetch = st.button("📊 Fetch cost history", width="stretch")
        with c2:
            pass  # mirror Forecast cleanup (no caption)

        # Advanced controls + CSV fallback
        with st.expander("Advanced (diagnostics & overrides)", expanded=False):
            present = _present_columns()
            st.caption("Detected columns (from Glue/`information_schema.columns`):")
            st.code(", ".join(present) or "(none)")
            default_date = _pick_date_col(present)
            default_cost = _pick_cost_col(present)
            date_override = st.selectbox("Date column", options=present or [default_date],
                                         index=(present.index(default_date) if default_date in present else 0))
            cost_override = st.selectbox("Cost column", options=present or [default_cost],
                                         index=(present.index(default_cost) if default_cost in present else 0))
            csv_file = st.file_uploader("Optional CSV fallback (columns: date,cost_usd or similar)", type=["csv"])

            st.caption("Tip: if your dates are day-first (e.g., 21/03/25), set env "
                       "`CUR_DATE_FMT='%d/%m/%y %H:%i'` or `%d/%m/%y` to improve parsing.")

        if not fetch:
            st.info("Set a horizon and click **Fetch cost history**.")
            return

        # CSV fallback, if provided
        if csv_file is not None:
            try:
                df = pd.read_csv(csv_file)
                cols = {c.lower(): c for c in df.columns}
                date_col = cols.get("date") or cols.get("day") or list(df.columns)[0]
                value_col = cols.get("cost_usd") or cols.get("amount") or list(df.columns)[-1]
                df = df.rename(columns={date_col: "date", value_col: "cost_usd"})
                df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
                df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
                df = df.dropna(subset=["cost_usd"]).sort_values("date").tail(int(days)).reset_index(drop=True)
            except Exception as e:
                st.error(f"Failed to parse uploaded CSV: {e}")
                df = pd.DataFrame()
            picked_date, picked_cost, mode_used = "(csv)", "(csv)", "csv"
        else:
            # Try Athena using selected overrides
            try:
                df, picked_date, picked_cost, mode_used = _fetch_daily_cost(
                    int(days),
                    chosen_date_col=date_override,
                    chosen_cost_col=cost_override
                )
            except Exception as e:
                st.error(f"Failed to fetch daily cost from Athena: {e}")
                df = pd.DataFrame()
                picked_date, picked_cost, mode_used = date_override, cost_override, "error"

        # Diagnostics line (always visible)
        st.caption(f"Using date column: **{picked_date}**, cost column: **{picked_cost}**  —  fetch mode: **{mode_used}**")

        if df.empty:
            st.info("No cost data available.")
            return

        # Spike detection (7-day rolling, z>2)
        dfz = _detect_spikes(df, z_thresh=2.0, roll=7)

        # Metrics — $ and commas, same styling as Forecast
        daily_avg   = float(dfz["cost_usd"].mean())
        spike_count = int(dfz["is_spike"].sum())
        max_row     = dfz.loc[dfz["cost_usd"].idxmax()]
        max_label   = f"{pd.to_datetime(max_row['date']).date().isoformat()}  —  ${max_row['cost_usd']:,.2f}"

        m1, m2, m3 = st.columns(3)
        m1.metric("Avg Daily Cost", f"${daily_avg:,.2f}")
 #       m2.metric("Spike Count", f"{spike_count}")
        m3.metric("Max Day", max_label)

        # ------------------- Chart (robust ticks + domain) -------------------
        chart_df = dfz[["date", "cost_usd", "is_spike"]].copy()
        chart_df["date"] = pd.to_datetime(chart_df["date"])

        n = len(chart_df)
        if n == 0:
            st.info("No points to plot.")
            return

        # Build explicit tick values so Altair doesn't drop the axis on odd windows (e.g., 45 days)
        start = chart_df["date"].min()
        end   = chart_df["date"].max()

        # ~8–12 ticks depending on window size
        step_days = max(1, math.floor(n / 10))
        tick_values = pd.date_range(start=start, end=end, freq=f"{step_days}D")

        # Friendly tooltip/axis format
        date_fmt = "%b %d"  # e.g., "Sep 11"

        x_axis = alt.X(
            "date:T",
            title=None,
            axis=alt.Axis(
                values=tick_values.to_pydatetime().tolist(),
                format=date_fmt,
                labelAngle=-45,
                labelOverlap=True,
                ticks=True
            ),
            # fix the continuous domain so the axis always renders
            scale=alt.Scale(domain=[start, end], clamp=True),
        )

        # Bar size adapts to horizon; keeps bars readable at ~30–60 days
        bar_size = 12 if n <= 60 else 6

        bars = alt.Chart(chart_df).mark_bar(size=bar_size, clip=True).encode(
            x=x_axis,
            y=alt.Y("cost_usd:Q", title=None),
            color=alt.condition(
                alt.datum.is_spike,
                alt.value("#ff4d4f"),   # spikes
                alt.value("#82c6ff"),   # normal
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format=date_fmt),
                alt.Tooltip("cost_usd:Q", title="Cost (USD)", format=",.2f"),
                alt.Tooltip("is_spike:N", title="Spike?"),
            ],
        )

        avg_rule = alt.Chart(pd.DataFrame({"y": [daily_avg]})).mark_rule(
            strokeDash=[4, 4], color="#aaaaaa"
        ).encode(y="y:Q")

        st.altair_chart((bars + avg_rule).properties(height=420), use_container_width=True)