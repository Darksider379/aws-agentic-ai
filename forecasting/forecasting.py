# forecasting.py
# -------------------------------------------------
# Forecast daily AWS cost from CUR (Athena) and publish results to S3 + Athena table
# Realism knobs:
# - COST_MODE=unblended (recommended for billed reality)
# - EXCLUDE_FEE=true (default) -> drop one-time "Fee" items
# - EXCLUDE_NON_USAGE_PRODUCTS=true (optional) -> drop Support/Marketplace/Tax
# - CLIP_OUTLIERS=true (optional), OUTLIER_Q=0.98, OUTLIER_MULT=1.20
# Prophet tuned to be steadier (additive mode + lower changepoint prior)

# 0) Load env BEFORE imports that may read os.environ
import os, sys
from pathlib import Path

def load_env_file(path: str = "config.ini"):
    """
    Load .ini/.env, but DO NOT override Lambda env (env set in the Lambda config wins).
    """
    in_lambda = bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    if os.environ.get("CONFIG_INI_DISABLE") in ("1", "true", "yes"):
        print("[env] CONFIG_INI_DISABLE=1 -> skipping config.ini load")
        return
    try:
        from dotenv import load_dotenv
        if Path(path).exists():
            load_dotenv(dotenv_path=path, override=not in_lambda)
            print(f"[env] loaded {path} via python-dotenv (override={not in_lambda})")
            return
    except Exception as e:
        print(f"[env] dotenv not used ({e}); falling back")
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[env] loaded {path} via fallback parser (setdefault)")
    else:
        print(f"[env] {path} not found; relying on process env")

env_path = "config.ini"
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    if i + 1 < len(sys.argv):
        env_path = sys.argv[i + 1]
load_env_file(env_path)

# 1) Now safe to import
import io, re, json, secrets
from datetime import datetime, timezone
import pandas as pd
import boto3

# Prophet optional; auto-fallback if unavailable
try:
    from prophet import Prophet
    HAVE_PROPHET = True
except Exception:
    HAVE_PROPHET = False

# Reuse your helper (must read env lazily inside the function)
from functions.run_athena import run_athena

# ---------- Env ----------
AWS_REGION            = os.environ.get("AWS_REGION", "us-east-1")
ATHENA_DB             = os.environ["ATHENA_DB"]
ATHENA_TABLE          = os.environ["ATHENA_TABLE"]
ATHENA_WORKGROUP      = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT         = os.environ["ATHENA_OUTPUT"]  # s3://...
RESULTS_BUCKET        = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX        = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")

# Choose which destination table you are using (raw CSV is simplest/robust)
ATHENA_FORECAST_TABLE = os.environ.get("ATHENA_FORECAST_TABLE", "forecast_daily_csv_raw")
FORECAST_TABLE_PREFIX = os.environ.get("FORECAST_TABLE_PREFIX", "forecast_table")

# Cost realism & filters
COST_MODE = os.environ.get("COST_MODE", "public_plus_effective").lower()  # recommended: "unblended"
EXCLUDE_FEE = os.environ.get("EXCLUDE_FEE", "true").lower() in ("1","true","yes")
EXCLUDE_NON_USAGE_PRODUCTS = os.environ.get("EXCLUDE_NON_USAGE_PRODUCTS", "false").lower() in ("1","true","yes")

# Optional timestamp hints for weird CURs
TS_COLUMN = os.environ.get("TS_COLUMN")   # e.g., line_item_usage_start_date
TS_FORMAT = os.environ.get("TS_FORMAT")   # e.g., "%d/%m/%y %H:%i"

# Outlier clamp (recommended for realism)
CLIP_OUTLIERS = os.environ.get("CLIP_OUTLIERS", "false").lower() in ("1","true","yes")
OUTLIER_Q = float(os.environ.get("OUTLIER_Q", "0.98"))
OUTLIER_MULT = float(os.environ.get("OUTLIER_MULT", "1.20"))

s3 = boto3.client("s3", region_name=AWS_REGION)

# ---------- Utilities ----------
def _new_run_id(context=None) -> str:
    """Per-invocation unique run_id (safe for Lambda warm starts)."""
    now = datetime.now(timezone.utc)
    ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond:06d}Z"
    suffix = (getattr(context, "aws_request_id", "") or secrets.token_hex(3))[:12]
    return f"{ts}-{suffix}"

def _s3_key_join(*parts: str) -> str:
    return "/".join(p.strip("/") for p in parts if p and p != "")

def _write_json_to_s3(obj: dict, bucket: str, key: str):
    body = json.dumps(obj, default=str).encode("utf-8")
    s3.put_object(Bucket=bucket, Key=key, Body=body,
                  ContentType="application/json",
                  CacheControl="no-store, no-cache, must-revalidate")
    return f"s3://{bucket}/{key}"

def _write_csv_to_s3(df: pd.DataFrame, bucket: str, key: str):
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"),
                  ContentType="text/csv",
                  CacheControl="no-store, no-cache, must-revalidate")
    return f"s3://{bucket}/{key}"

def _extract_horizon_days_from_event(event, default_days=90) -> int:
    try_keys = [("horizon_days", int), ("horizon", int)]
    if isinstance(event, dict):
        for k, caster in try_keys:
            if k in event:
                try: return caster(event[k])
                except: pass
        args = event.get("arguments") or event.get("parameters") or {}
        if isinstance(args, dict):
            for k, caster in try_keys:
                if k in args:
                    try: return caster(args[k])
                    except: pass
        text = event.get("inputText") or event.get("prompt") or event.get("query") or ""
        if isinstance(text, str) and text:
            m = re.search(r"\b(30|60|90)\s*(?:day|days|d)?\b", text, flags=re.IGNORECASE)
            if m:
                return int(m.group(1))
    return int(os.environ.get("FORECAST_DEFAULT_DAYS", default_days))

# ---------- Schema sniffers & SQL builders ----------
def _list_columns(table: str) -> set[str]:
    cols: set[str] = set()
    df = run_athena(f"DESCRIBE {table}")
    if not df.empty:
        if len(df.columns) >= 2:
            for i in range(len(df)):
                name = str(df.iloc[i, 0]).strip().lower()
                if not name or name.startswith("#"):
                    continue
                cols.add(name)
        else:
            for i in range(len(df)):
                cell = str(df.iloc[i, 0]) if df.iloc[i, 0] is not None else ""
                cell = cell.strip()
                if not cell or cell.startswith("#"):
                    continue
                name = re.split(r"\s{2,}|\t", cell)[0].strip().lower()
                if name:
                    cols.add(name)
    if not cols:
        df2 = run_athena(f"SELECT * FROM {table} LIMIT 1")
        for c in df2.columns:
            if c and isinstance(c, str):
                cols.add(c.strip().lower())
    return cols

def _num(name: str) -> str:
    return f"COALESCE(TRY_CAST({name} AS DOUBLE), 0.0)"

def _build_cost_expr(cols: set[str]) -> str:
    if COST_MODE == "blended" and "line_item_blended_cost" in cols:
        return _num("line_item_blended_cost")
    if COST_MODE == "unblended" and "line_item_unblended_cost" in cols:
        return _num("line_item_unblended_cost")
    parts = []
    if "pricing_public_on_demand_cost" in cols: parts.append(_num("pricing_public_on_demand_cost"))
    if "reservation_effective_cost" in cols:   parts.append(_num("reservation_effective_cost"))
    if "savings_plan_effective_cost" in cols:  parts.append(_num("savings_plan_effective_cost"))
    if parts: return " + ".join(parts)
    if "line_item_unblended_cost" in cols: return _num("line_item_unblended_cost")
    if "line_item_blended_cost" in cols:   return _num("line_item_blended_cost")
    raise RuntimeError("No suitable cost columns found.")

def _build_where_clause(cols: set[str]) -> str:
    clauses = []
    if "line_item_line_item_type" in cols:
        if EXCLUDE_FEE:
            clauses.append("line_item_line_item_type IN ('Usage','DiscountedUsage','SavingsPlanCoveredUsage')")
        else:
            clauses.append("line_item_line_item_type IN ('Usage','DiscountedUsage','SavingsPlanCoveredUsage','Fee')")
    elif "line_item_usage_amount" in cols:
        clauses.append("COALESCE(TRY_CAST(line_item_usage_amount AS DOUBLE), 0.0) >= 0")
    # Optional: drop non-usage products that inflate bills
    if EXCLUDE_NON_USAGE_PRODUCTS and "line_item_product_code" in cols:
        clauses.append(
            "COALESCE(line_item_product_code,'') NOT IN ("
            "'AwsSupport','AWSSupportBusiness','AWSSupportEnterprise','AWSSupportDeveloper',"
            "'AWSMarketplace','Tax','AWSTax'"
            ")"
        )
    return ("WHERE " + " AND ".join(clauses)) if clauses else ""

# ---------- Timestamp parsers (Athena-side) ----------
def _ddmmyy_pivot_expr(ts_col: str, with_time: bool) -> str:
    if with_time:
        pat = "'^(\\d{1,2})/(\\d{1,2})/(\\d{2})\\s+(\\d{1,2}):(\\d{2})$'"
        y = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 3) AS integer)"
        m = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 2) AS integer)"
        d = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 1) AS integer)"
        H = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 4) AS integer)"
        I = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 5) AS integer)"
        yyyy = f"(CASE WHEN {y} <= 69 THEN 2000 + {y} ELSE 1900 + {y} END)"
        s = f"format('%04d-%02d-%02d %02d:%02d:%02d', {yyyy}, {m}, {d}, {H}, {I}, 0)"
        return f"IF(regexp_like({ts_col}, {pat}), try(date_parse({s}, '%Y-%m-%d %H:%i:%s')), NULL)"
    else:
        pat = "'^(\\d{1,2})/(\\d{1,2})/(\\d{2})$'"
        y = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 3) AS integer)"
        m = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 2) AS integer)"
        d = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 1) AS integer)"
        yyyy = f"(CASE WHEN {y} <= 69 THEN 2000 + {y} ELSE 1900 + {y} END)"
        s = f"format('%04d-%02d-%02d %02d:%02d:%02d', {yyyy}, {m}, {d}, 0, 0, 0)"
        return f"IF(regexp_like({ts_col}, {pat}), try(date_parse({s}, '%Y-%m-%d %H:%i:%s')), NULL)"

def _yymmdd_hyphen_pivot_expr(ts_col: str) -> str:
    pat = "'^(\\d{2})-(\\d{1,2})-(\\d{1,2})(?:\\s+(\\d{1,2}):(\\d{2})(?::(\\d{2}))?)?$'"
    y = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 1) AS integer)"
    m = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 2) AS integer)"
    d = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 3) AS integer)"
    H = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 4) AS integer), 0)"
    I = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 5) AS integer), 0)"
    S = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 6) AS integer), 0)"
    yyyy = f"(CASE WHEN {y} <= 69 THEN 2000 + {y} ELSE 1900 + {y} END)"
    s = f"format('%04d-%02d-%02d %02d:%02d:%02d', {yyyy}, {m}, {d}, {H}, {I}, {S})"
    return f"IF(regexp_like({ts_col}, {pat}), try(date_parse({s}, '%Y-%m-%d %H:%i:%s')), NULL)"

def _yymmdd_slash_pivot_expr(ts_col: str) -> str:
    pat = "'^(\\d{2})/(\\d{1,2})/(\\d{1,2})(?:\\s+(\\d{1,2}):(\\d{2})(?::(\\d{2}))?)?$'"
    y = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 1) AS integer)"
    m = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 2) AS integer)"
    d = f"TRY_CAST(regexp_extract({ts_col}, {pat}, 3) AS integer)"
    H = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 4) AS integer), 0)"
    I = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 5) AS integer), 0)"
    S = f"COALESCE(TRY_CAST(regexp_extract({ts_col}, {pat}, 6) AS integer), 0)"
    yyyy = f"(CASE WHEN {y} <= 69 THEN 2000 + {y} ELSE 1900 + {y} END)"
    s = f"format('%04d-%02d-%02d %02d:%02d:%02d', {yyyy}, {m}, {d}, {H}, {I}, {S})"
    return f"IF(regexp_like({ts_col}, {pat}), try(date_parse({s}, '%Y-%m-%d %H:%i:%s')), NULL)"

def _timestamp_expr_for_col(ts_col: str) -> str:
    tries = [
        _ddmmyy_pivot_expr(ts_col, with_time=True),
        _yymmdd_hyphen_pivot_expr(ts_col),
        _yymmdd_slash_pivot_expr(ts_col),
        _ddmmyy_pivot_expr(ts_col, with_time=False),
    ]
    if TS_FORMAT:
        fmt = TS_FORMAT.replace("'", "''")
        tries.append(f"try(date_parse({ts_col}, '{fmt}'))")
    tries.append(f"try(from_iso8601_timestamp({ts_col}))")
    for p in ["%Y-%m-%d %H:%i:%s","%Y-%m-%d %H:%i",
              "%d/%m/%Y %H:%i:%s","%d/%m/%Y %H:%i",
              "%m/%d/%Y %H:%i:%s","%m/%d/%Y %H:%i",
              "%Y-%m-%d","%d/%m/%Y","%m/%d/%Y"]:
        tries.append(f"try(date_parse({ts_col}, '{p}'))")
    tries.append(f"CAST({ts_col} AS timestamp)")
    return "COALESCE(" + ", ".join(tries) + ")"

def _best_timestamp_expr(cols: set[str]) -> str:
    candidates = []
    if TS_COLUMN:
        candidates.append(TS_COLUMN.strip().lower())
    for c in ["line_item_usage_start_date", "line_item_usage_end_date", "usage_start_time", "usage_start_date"]:
        if c in cols and c not in candidates: candidates.append(c)
    if not candidates:
        raise RuntimeError("No usable timestamp column found.")
    return "COALESCE(" + ", ".join(_timestamp_expr_for_col(c) for c in candidates) + ")"

def _build_daily_sql(table: str, cols: set[str]) -> str:
    cost = _build_cost_expr(cols)
    where = _build_where_clause(cols)
    ts_expr = _best_timestamp_expr(cols)
    return f"""
WITH base AS (
  SELECT date_trunc('day', {ts_expr}) AS raw_ts, {cost} AS cost_piece
  FROM {table}
  {where}
),
norm AS (
  SELECT CAST(
           date_trunc('day',
             CASE
               WHEN raw_ts IS NULL THEN NULL
               WHEN year(raw_ts) < 100 THEN
                 date_parse(format('%04d-%02d-%02d %02d:%02d:%02d',
                                   CASE WHEN year(raw_ts) <= 69 THEN 2000 + year(raw_ts) ELSE 1900 + year(raw_ts) END,
                                   month(raw_ts), day_of_month(raw_ts), 0, 0, 0),
                            '%Y-%m-%d %H:%i:%s')
               ELSE raw_ts
             END
           ) AS DATE
         ) AS day,
         cost_piece
  FROM base
),
daily AS (
  SELECT day, SUM(cost_piece) AS cost_usd
  FROM norm
  WHERE day IS NOT NULL
  GROUP BY 1
),
bounds AS ( SELECT max(day) AS max_day FROM daily )
SELECT d.day, CAST(d.cost_usd AS DOUBLE) AS cost_usd
FROM daily d
CROSS JOIN bounds b
WHERE b.max_day IS NOT NULL
  AND d.day >= date_add('day', -400, b.max_day)
ORDER BY d.day;
"""

# ---------- Forecast logic ----------
def _forecast(df_daily: pd.DataFrame, horizon_days: int) -> pd.DataFrame:
    df = df_daily.rename(columns={"day":"ds","cost_usd":"y"}).copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce").dt.tz_localize(None)
    df["y"] = pd.to_numeric(df["y"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["ds"]).sort_values("ds")
    if len(df) < 7:
        raise ValueError("Not enough history (>= 7 daily points required).")

    if HAVE_PROPHET and len(df) >= 30:
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="additive",        # steadier than multiplicative
            changepoint_prior_scale=0.05        # smoother trend
        )
        m.fit(df)
        future = m.make_future_dataframe(periods=int(horizon_days), freq="D", include_history=False)
        fcst = m.predict(future)[["ds","yhat","yhat_lower","yhat_upper"]]
    else:
        avg7 = df.tail(7)["y"].mean()
        future = pd.date_range(df["ds"].max() + pd.Timedelta(days=1), periods=int(horizon_days), freq="D")
        fcst = pd.DataFrame({
            "ds": future,
            "yhat": avg7,
            "yhat_lower": avg7 * 0.8,
            "yhat_upper": avg7 * 1.2
        })
    fcst["ds"] = pd.to_datetime(fcst["ds"]).dt.date
    fcst[["yhat","yhat_lower","yhat_upper"]] = fcst[["yhat","yhat_lower","yhat_upper"]].clip(lower=0)
    return fcst

# ---------- Athena forecast table mgmt (RAW CSV table) ----------
def _build_forecast_table_if_needed():
    base_loc = f"s3://{RESULTS_BUCKET}/{_s3_key_join(RESULTS_PREFIX, FORECAST_TABLE_PREFIX)}/"
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_FORECAST_TABLE} (
      ds string,
      yhat string,
      yhat_lower string,
      yhat_upper string
    )
    PARTITIONED BY (run_id string)
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"', 'escapeChar' = '\\\\')
    LOCATION '{base_loc}'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """
    run_athena(ddl)

def _add_partition_for_run(run_id: str):
    part_loc = f"s3://{RESULTS_BUCKET}/{_s3_key_join(RESULTS_PREFIX, FORECAST_TABLE_PREFIX, f'run_id={run_id}')}/"
    alter = f"ALTER TABLE {ATHENA_FORECAST_TABLE} ADD IF NOT EXISTS PARTITION (run_id='{run_id}') LOCATION '{part_loc}';"
    run_athena(alter)

# ---------- Data normalization ----------
def _normalize_daily_df(daily: pd.DataFrame) -> pd.DataFrame:
    try:
        print("[debug] raw day sample:", list(daily["day"].head(5)))
    except Exception:
        pass
    dta = pd.to_datetime(daily["day"], format="%Y-%m-%d", errors="coerce")
    if dta.notna().sum() >= 5:
        dt = dta
    else:
        dt = pd.to_datetime(daily["day"], errors="coerce")
    mask = (dt.dt.year >= 1970) & (dt.dt.year <= 2100)
    daily = daily.copy()
    daily["day"] = dt.where(mask).dt.date
    daily = daily.dropna(subset=["day"])
    return daily

# ---------- Main entry ----------
def run_forecast(event=None, context=None):
    run_id = os.environ.get("RUN_ID_OVERRIDE") or _new_run_id(context)
    horizon = _extract_horizon_days_from_event(event or {}, default_days=90)
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"[cfg] REGION={AWS_REGION} DB={ATHENA_DB} TABLE={ATHENA_TABLE} WG={ATHENA_WORKGROUP}")
    print(f"[cfg] RESULTS_BUCKET={RESULTS_BUCKET} RESULTS_PREFIX={RESULTS_PREFIX} COST_MODE={COST_MODE} EXCLUDE_FEE={EXCLUDE_FEE} EXCLUDE_NON_USAGE_PRODUCTS={EXCLUDE_NON_USAGE_PRODUCTS}")
    print(f"[cfg] RUN_ID={run_id} horizon={horizon}")
    print("[athena] fetching table schema…")

    cols = _list_columns(ATHENA_TABLE)
    if not cols:
        raise RuntimeError(f"DESCRIBE returned no columns for {ATHENA_TABLE}.")
    print(f"[athena] columns detected (sample): {sorted(list(cols))[:12]} ...")

    sql = _build_daily_sql(ATHENA_TABLE, cols)
    print("[athena] fetching daily history…")
    daily = run_athena(sql)
    if daily.empty:
        raise RuntimeError("Athena returned 0 rows for daily cost.")

    daily.columns = [c.strip().lower() for c in daily.columns]
    daily = _normalize_daily_df(daily)
    daily["cost_usd"] = pd.to_numeric(daily["cost_usd"], errors="coerce").fillna(0.0)

    # ---- realism: optional outlier clipping ----
    clip_info = None
    if CLIP_OUTLIERS and not daily.empty:
        qv = daily["cost_usd"].quantile(OUTLIER_Q)
        if pd.notna(qv) and qv > 0:
            cap = qv * OUTLIER_MULT
            before_max = float(daily["cost_usd"].max())
            daily["cost_usd"] = daily["cost_usd"].clip(upper=cap)
            after_max = float(daily["cost_usd"].max())
            clip_info = {"q": OUTLIER_Q, "mult": OUTLIER_MULT, "cap": float(cap),
                         "before_max": before_max, "after_max": after_max}
            print(f"[denoise] clipped outliers above ${cap:,.2f} (prev max ${before_max:,.2f}, new max ${after_max:,.2f})")

    if len(daily) < 7:
        raise ValueError(f"Not enough history after normalization (got {len(daily)} rows).")

    # Diagnostics: recent averages
    hist_avg_30 = float(daily.tail(30)["cost_usd"].mean()) if len(daily) >= 30 else float(daily["cost_usd"].mean())
    hist_avg_7  = float(daily.tail(7)["cost_usd"].mean()) if len(daily) >= 7 else hist_avg_30
    print(f"[diag] history rows={len(daily)} range={daily['day'].min()}..{daily['day'].max()} "
          f"avg7=${hist_avg_7:,.2f} avg30=${hist_avg_30:,.2f}")

    fcst = _forecast(daily, horizon_days=horizon)

    base_ui_prefix = _s3_key_join(RESULTS_PREFIX, "runs", run_id, "forecast")
    ui_hist_key = _s3_key_join(base_ui_prefix, "history_daily.csv")
    ui_fcst_key = _s3_key_join(base_ui_prefix, "forecast_daily.csv")
    ui_summary_key = _s3_key_join(base_ui_prefix, "summary.json")

    hist_uri = _write_csv_to_s3(daily, RESULTS_BUCKET, ui_hist_key)
    fcst_uri = _write_csv_to_s3(fcst,  RESULTS_BUCKET, ui_fcst_key)

    _build_forecast_table_if_needed()
    table_part_prefix = _s3_key_join(RESULTS_PREFIX, FORECAST_TABLE_PREFIX, f"run_id={run_id}")
    tbl_key = _s3_key_join(table_part_prefix, "forecast_daily.csv")
    _write_csv_to_s3(fcst[["ds","yhat","yhat_lower","yhat_upper"]], RESULTS_BUCKET, tbl_key)
    _add_partition_for_run(run_id)

    fc = fcst.copy()
    fc["ds"] = pd.to_datetime(fc["ds"])
    this_month_total = float(fc.loc[fc["ds"].dt.to_period("M")==pd.Timestamp.utcnow().to_period("M"), "yhat"].sum())

    summary = {
        "run_id": run_id,
        "created_at": now_iso,
        "horizon_days": int(horizon),
        "history_days": int(len(daily)),
        "diagnostics": {
            "cost_mode": COST_MODE,
            "exclude_fee": EXCLUDE_FEE,
            "exclude_non_usage_products": EXCLUDE_NON_USAGE_PRODUCTS,
            "clip_outliers": CLIP_OUTLIERS,
            "clip_info": clip_info,
            "history_avg_7d": round(hist_avg_7, 2),
            "history_avg_30d": round(hist_avg_30, 2)
        },
        "totals": {
            "forecast_sum_next_30d_usd": float(fc.head(30)["yhat"].sum()),
            "forecast_sum_next_60d_usd": float(fc.head(60)["yhat"].sum()),
            "forecast_sum_next_90d_usd": float(fc.head(90)["yhat"].sum()),
            "this_month_forecast_usd": this_month_total
        },
        "artifacts": {
            "history_csv": hist_uri,
            "forecast_csv": fcst_uri,
            "athena_forecast_table": ATHENA_FORECAST_TABLE,
            "athena_partition_run_id": run_id,
            "athena_partition_s3": f"s3://{RESULTS_BUCKET}/{table_part_prefix}/"
        }
    }
    summary_uri = _write_json_to_s3(summary, RESULTS_BUCKET, ui_summary_key)

    print(f"[done] history={hist_uri}")
    print(f"[done] forecast={fcst_uri}")
    print(f"[done] forecast_table={ATHENA_FORECAST_TABLE} partition=run_id={run_id} -> s3://{RESULTS_BUCKET}/{table_part_prefix}/")
    print(f"[done] summary={summary_uri}")

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json", "Cache-Control": "no-store, no-cache, must-revalidate"},
        "body": json.dumps({"run_id": run_id, "summary_s3_uri": summary_uri, "summary": summary})
    }

# -------------- Lambda convenience handler --------------
def lambda_handler(event, context):
    return run_forecast(event, context)

# -------------- Local run --------------
if __name__ == "__main__":
    prompt = None
    if "--prompt" in sys.argv:
        j = sys.argv.index("--prompt")
        if j + 1 < len(sys.argv):
            prompt = sys.argv[j + 1]
    evt = {}
    if prompt:
        evt["inputText"] = prompt
    print(f"[cfg] REGION={AWS_REGION} DB={ATHENA_DB} TABLE={ATHENA_TABLE} WG={ATHENA_WORKGROUP}")
    print(f"[cfg] RESULTS_BUCKET={RESULTS_BUCKET} RESULTS_PREFIX={RESULTS_PREFIX} COST_MODE={COST_MODE}")
    out = run_forecast(evt, None)
    print(json.dumps(json.loads(out["body"]), indent=2))
