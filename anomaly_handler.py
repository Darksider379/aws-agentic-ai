# lambda/anomaly_handler.py
# Anomaly job: CUR -> detect -> write CSV -> ensure Athena table (non-partitioned)
# Local run: python anomaly_handler.py --env config.ini

import os, sys, json, io
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import boto3

# ------------------------------------------------------------------------------
# 0) Load env (supports: --env config.ini)
# ------------------------------------------------------------------------------
def load_env_file(path: str = "config.ini"):
    """Load KEY=VALUE pairs into os.environ (dotenv optional, then fallback)."""
    try:
        from dotenv import load_dotenv
        if Path(path).exists():
            load_dotenv(dotenv_path=path, override=True)
            print(f"[env] loaded {path} via python-dotenv")
            return
    except Exception as e:
        print(f"[env] dotenv not used ({e}); falling back")
    if Path(path).exists():
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
        print(f"[env] loaded {path} via fallback parser")
    else:
        print(f"[env] {path} not found; relying on process env")

env_path = "config.ini"
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    if i + 1 < len(sys.argv):
        env_path = sys.argv[i + 1]
load_env_file(env_path)

# ------------------------------------------------------------------------------
# 1) Env + clients
# ------------------------------------------------------------------------------
ATHENA_DB         = os.environ["ATHENA_DB"]
ATHENA_TABLE      = os.environ["ATHENA_TABLE"]
ATHENA_WORKGROUP  = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT     = os.environ["ATHENA_OUTPUT"]

RESULTS_BUCKET    = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX    = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")
ANOM_TABLE        = os.environ.get("ANOMALY_TABLE", "cost_anomalies")

USE_LLM          = os.environ.get("USE_LLM", "false").lower() == "true"
BEDROCK_REGION   = os.environ.get("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

ANOMALY_MIN_HISTORY = int(os.environ.get("ANOMALY_MIN_HISTORY", "14"))
ANOMALY_MIN_VOTES   = int(os.environ.get("ANOMALY_MIN_VOTES", "2"))
ANOMALY_SPAN_EWMA   = int(os.environ.get("ANOMALY_SPAN_EWMA", "14"))
ANOMALY_Z_RTHRESH   = float(os.environ.get("ANOMALY_Z_RTHRESH", "3.5"))
ANOMALY_IQR_WIN     = int(os.environ.get("ANOMALY_IQR_WIN", "21"))
ANOMALY_NOTIFY_TOPN = int(os.environ.get("ANOMALY_NOTIFY_TOPN", "10"))

from functions.run_athena import run_athena  # your existing helper

# Optional Bedrock LLM
bedrock = None
if USE_LLM:
    bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

# ------------------------------------------------------------------------------
# 2) CUR schema discovery + adaptive SQL
# ------------------------------------------------------------------------------
def _cur_columns(schema: str, table: str) -> set[str]:
    sql = f"""
      SELECT lower(column_name) AS c
      FROM information_schema.columns
      WHERE lower(table_schema) = lower('{schema}')
        AND lower(table_name)  = lower('{table}')
    """
    df = run_athena(sql)
    return set(df["c"].str.lower()) if not df.empty else set()

def _pick(colset: set[str], *candidates: str, required: bool = True, default: str | None = None) -> str | None:
    for c in candidates:
        if c and c.lower() in colset:
            return c
    if required and default is None:
        raise RuntimeError(f"None of the expected columns present: {candidates}")
    return default

def query_daily_costs_90d() -> pd.DataFrame:
    cols = _cur_columns(ATHENA_DB, ATHENA_TABLE)

    # usage start (required)
    col_usage_start = _pick(
        cols,
        "line_item_usage_start_date",
        "line_item_usage_start_time",
        "line_item_usage_start_datetime",
        "usage_start_date",
        "usage_start_time",
        "start_time",
        "usagestartdate",
        "usagestarttime",
    )

    # account id (optional)
    col_acct = _pick(
        cols,
        "line_item_usage_account_id",
        "line_item_usageaccountid",
        "aws_account_id",
        "bill_payer_account_id",
        "bill_payer_accountid",
        "payer_account_id",
        "linked_account_id",
        "account_id",
        "accountid",
        required=False,
        default=None,
    )
    acct_expr = f"COALESCE(CAST({col_acct} AS VARCHAR), 'UNKNOWN')" if col_acct else "'UNKNOWN'"

    # service (required-ish)
    col_service = _pick(
        cols,
        "product_product_name",
        "product_servicename",
        "product_servicecode",
        "line_item_product_code",
        "product_productfamily",
        "service_name",
        "productname",
        "service",
        "productcode",
    )

    # region (optional; derive from AZ if missing)
    if "product_region" in cols:
        region_expr = "product_region"
    elif "region" in cols:
        region_expr = "region"
    else:
        az_col = _pick(cols, "line_item_availability_zone", "availability_zone", required=False, default=None)
        region_expr = (
            f"IF({az_col} IS NULL OR {az_col} = '', 'GLOBAL', regexp_replace({az_col}, '[a-z]$', ''))"
            if az_col else "'GLOBAL'"
        )

    # cost (required)
    cost_expr = None
    for cand in ("pricing_public_on_demand_cost", "line_item_unblended_cost", "line_item_blended_cost", "cost", "amount"):
        if cand in cols:
            cost_expr = cand
            break
    if not cost_expr:
        raise RuntimeError("No cost column found (pricing_public_on_demand_cost / line_item_unblended_cost / line_item_blended_cost / cost / amount).")

    # tolerant TIMESTAMP -> DATE parsing
    usage_ts_norm = (
        f"COALESCE("
        f"  TRY(from_iso8601_timestamp({col_usage_start})),"
        f"  TRY(date_parse({col_usage_start}, '%Y-%m-%d %H:%i:%s')),"
        f"  TRY(date_parse({col_usage_start}, '%Y-%m-%d %H:%i')),"
        f"  TRY(date_parse({col_usage_start}, '%Y-%m-%d')),"
        f"  TRY(date_parse({col_usage_start}, '%d/%m/%y %H:%i')),"   # e.g., 20/09/24 00:00
        f"  TRY(date_parse({col_usage_start}, '%d/%m/%Y %H:%i')),"
        f"  TRY(date_parse({col_usage_start}, '%d/%m/%y')),"
        f"  TRY(date_parse({col_usage_start}, '%d/%m/%Y'))"
        f")"
    )
    usage_date_norm = f"CAST({usage_ts_norm} AS DATE)"

    sql = f"""
    WITH base AS (
      SELECT
        {usage_date_norm}                                   AS usage_date,
        {acct_expr}                                         AS account_id,
        COALESCE({col_service}, 'UNKNOWN')                  AS service,
        COALESCE({region_expr}, 'GLOBAL')                   AS region,
        TRY_CAST(COALESCE({cost_expr}, 0) AS DOUBLE)        AS raw_cost
      FROM {ATHENA_DB}.{ATHENA_TABLE}
      WHERE {usage_date_norm} IS NOT NULL
        AND {usage_date_norm} >= date_add('day', -90, current_date)
    )
    SELECT
      usage_date, account_id, service, region,
      ROUND(SUM(raw_cost), 6) AS amortized_cost
    FROM base
    GROUP BY 1,2,3,4
    ORDER BY 2,3,4,1
    """
    df = run_athena(sql)
    if df.empty:
        return df
    df["usage_date"] = pd.to_datetime(df["usage_date"]).dt.date
    df["amortized_cost"] = pd.to_numeric(df["amortized_cost"], errors="coerce").fillna(0.0)
    return df

# ------------------------------------------------------------------------------
# 3) Detectors
# ------------------------------------------------------------------------------
def _rolling_mad(x: pd.Series) -> float:
    arr = np.asarray(x, dtype=float)
    med = np.median(arr)
    return float(np.median(np.abs(arr - med)))

def _robust_z(series: pd.Series, window: int, z_thresh: float) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    roll = s.rolling(window, min_periods=max(5, window//2))
    median = roll.median()
    mad = roll.apply(_rolling_mad, raw=False).replace(0, np.nan)
    rz = 0.6745 * (s - median) / mad
    return pd.DataFrame({"robust_z": rz, "robust_z_flag": rz.abs() >= z_thresh})

def _ewma_drift(series: pd.Series, span: int, k: float) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    mu = s.ewm(span=span, adjust=False).mean()
    sigma = s.ewm(span=span, adjust=False).std(bias=False).replace(0, np.nan)
    z = (s - mu) / sigma
    return pd.DataFrame({"ewma_z": z, "ewma_flag": z.abs() >= k})

def _iqr_spike(series: pd.Series, window: int, whisker: float = 1.5) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    roll = s.rolling(window, min_periods=max(5, window//2))
    q1 = roll.quantile(0.25)
    q3 = roll.quantile(0.75)
    iqr = (q3 - q1)
    lower = q1 - whisker * iqr
    upper = q3 + whisker * iqr
    flag = (s < lower) | (s > upper)
    return pd.DataFrame({"iqr_lower": lower, "iqr_upper": upper, "iqr_flag": flag})

def _ensemble_flags(df_flags: pd.DataFrame, min_votes: int) -> tuple[pd.Series, pd.Series]:
    vote_cols = [c for c in df_flags.columns if c.endswith("_flag")]
    votes = df_flags[vote_cols].sum(axis=1)
    return (votes >= min_votes), votes

def _score_severity(row: pd.Series) -> float:
    zsum = abs(row.get("robust_z", 0)) + abs(row.get("ewma_z", 0))
    return float(row.get("ensemble_votes", 0) * 2 + zsum)

def _explain(row: pd.Series, dims: dict) -> str:
    return (
        f"On {row['usage_date']}, cost={row['value']:.2f} for "
        f"account={dims['account_id']}, service={dims['service']}, region={dims['region']} "
        f"flagged (votes={int(row['ensemble_votes'])}); "
        f"robust_z={row.get('robust_z', float('nan')):.2f}, "
        f"ewma_z={row.get('ewma_z', float('nan')):.2f}, "
        f"outside_IQR={'yes' if row.get('iqr_flag', False) else 'no'}."
    )

# ------------------------------------------------------------------------------
# 4) Pipeline
# ------------------------------------------------------------------------------
def detect_anomalies_grouped(df_daily: pd.DataFrame) -> List[Dict[str, Any]]:
    if df_daily.empty:
        return []

    keys = ["account_id", "service", "region"]
    results: List[Dict[str, Any]] = []

    for dims, g in df_daily.groupby(keys, dropna=False):
        sub = g[["usage_date", "amortized_cost"]].sort_values("usage_date")
        if len(sub) < ANOMALY_MIN_HISTORY:
            continue

        s = sub.set_index("usage_date")["amortized_cost"]
        rz = _robust_z(s, window=max(ANOMALY_IQR_WIN, 21), z_thresh=ANOMALY_Z_RTHRESH)
        ew = _ewma_drift(s, span=ANOMALY_SPAN_EWMA, k=3.0)
        iq = _iqr_spike(s, window=ANOMALY_IQR_WIN, whisker=1.5)
        det = pd.concat([rz, ew, iq], axis=1)

        ens_flag, votes = _ensemble_flags(det, ANOMALY_MIN_VOTES)
        det["ensemble_votes"] = votes
        det["anomaly"] = ens_flag
        det["value"] = s
        det = det.reset_index().rename(columns={"index": "usage_date"})

        latest_day = det["usage_date"].max()
        today = det.loc[det["usage_date"] == latest_day].iloc[-1]
        if bool(today["anomaly"]):
            dim_map = dict(zip(keys, dims))
            item = {
                "usage_date": str(today["usage_date"]),
                **dim_map,
                "cost": float(today["value"]),
                "robust_z": float(today.get("robust_z") or 0.0),
                "ewma_z": float(today.get("ewma_z") or 0.0),
                "iqr_flag": bool(today.get("iqr_flag", False)),
                "ensemble_votes": int(today["ensemble_votes"]),
                "severity": _score_severity(today),
                "explanation_seed": _explain(today, dim_map),
            }
            results.append(item)

    results.sort(key=lambda x: x["severity"], reverse=True)
    return results

def _llm_polish(items: List[Dict[str, Any]], topn: int) -> List[Dict[str, Any]]:
    if not USE_LLM or not items:
        return items
    polished = []
    for it in items[:topn]:
        prompt = (
            f"{it.get('explanation_seed','')}\n"
            "Add 2–3 probable drivers and 2 checks (Athena/Tags). Keep under 80 words."
        )
        try:
            resp = bedrock.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[{"role":"user","content":[{"text":prompt}]}],
                inferenceConfig={"maxTokens": 180, "temperature": 0.1, "topP": 0.9},
            )
            txt = "".join(
                block.get("text", "")
                for block in resp.get("output", {}).get("message", {}).get("content", [])
            ).strip()
            it2 = dict(it)
            it2["explanation"] = txt or it.get("explanation_seed", "")
            polished.append(it2)
        except Exception as e:
            print(f"[llm][polish] {e}")
            polished.append(it)
    return polished + items[len(polished):]

# ------------------------------------------------------------------------------
# 5) Write anomalies CSV to S3 + ensure Athena table (non-partitioned)
# ------------------------------------------------------------------------------
def write_anomalies_csv_to_s3(items: list[dict], run_id: str,
                              bucket: str, prefix: str,
                              key_prefix: str = "anomalies",
                              filename: str = "anomalies.csv") -> str:
    """
    Writes items to s3://bucket/prefix/{key_prefix}/{run_id}/{filename}
    Returns folder URI: s3://.../{key_prefix}/{run_id}/
    """
    cols = [
        "usage_date","account_id","service","region","cost",
        "robust_z","ewma_z","iqr_flag","ensemble_votes","severity",
        "explanation_seed","explanation","run_id","created_at"
    ]
    df = pd.DataFrame(items)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]

    # normalize boolean to lowercase strings to be extra-safe with CSV -> Athena
    df["iqr_flag"] = df["iqr_flag"].map(lambda x: str(bool(x)).lower() if pd.notna(x) else "")

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    body = csv_buf.getvalue().encode("utf-8")

    s3 = boto3.client("s3")
    key = f"{prefix.rstrip('/')}/{key_prefix}/{run_id}/{filename}"
    s3.put_object(Bucket=bucket, Key=key, Body=body)
    return f"s3://{bucket}/{prefix.rstrip('/')}/{key_prefix}/{run_id}/"

def ensure_anomaly_table():
    """Create the non-partitioned cost_anomalies table if it doesn't exist."""
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DB}.{ANOM_TABLE} (
      usage_date DATE,
      account_id STRING,
      service STRING,
      region STRING,
      cost DOUBLE,
      robust_z DOUBLE,
      ewma_z DOUBLE,
      iqr_flag BOOLEAN,
      ensemble_votes INT,
      severity DOUBLE,
      explanation_seed STRING,
      explanation STRING,
      run_id STRING,
      created_at STRING
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe'
    WITH SERDEPROPERTIES ('serialization.format' = ',', 'field.delim' = ',')
    LOCATION 's3://{RESULTS_BUCKET}/{RESULTS_PREFIX}/anomalies/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """
    run_athena(ddl)

# ------------------------------------------------------------------------------
# 6) Lambda entry
# ------------------------------------------------------------------------------
def lambda_handler(event, context):
    run_id = os.environ.get("RUN_ID") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"[cfg] DB={ATHENA_DB} TABLE={ATHENA_TABLE} OUT={ATHENA_OUTPUT} RUN_ID={run_id}")

    # Ensure the table exists (idempotent)
    ensure_anomaly_table()

    # Fetch + detect
    df = query_daily_costs_90d()
    print(f"[anomaly] daily rows: {len(df)}")

    anomalies = detect_anomalies_grouped(df)
    if USE_LLM:
        anomalies = _llm_polish(anomalies, ANOMALY_NOTIFY_TOPN)

    # Stamp metadata
    now_iso = datetime.now(timezone.utc).isoformat()
    for a in anomalies:
        a["run_id"] = run_id
        a["created_at"] = now_iso

    # Write CSV
    s3_uri = write_anomalies_csv_to_s3(
        anomalies,
        run_id=run_id,
        bucket=RESULTS_BUCKET,
        prefix=RESULTS_PREFIX,
        key_prefix="anomalies",
        filename="anomalies.csv",
    )
    print(f"[anomaly] wrote {len(anomalies)} rows -> {s3_uri}")

    out = {
        "run_id": run_id,
        "table": f"{ATHENA_DB}.{ANOM_TABLE}",
        "s3_prefix": s3_uri,
        "count": len(anomalies),
        "top": anomalies[:ANOMALY_NOTIFY_TOPN],
    }
    return {"statusCode": 200, "body": json.dumps(out, default=str)}

# Local test
if __name__ == "__main__":
    print(json.dumps(lambda_handler({}, None), indent=2))
