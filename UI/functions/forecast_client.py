# functions/forecast_client.py
import os
import json
from typing import Dict, Any, List, Tuple
from io import BytesIO

import boto3
import pandas as pd

REGION = os.getenv("AWS_REGION", "us-east-1")
FORECAST_LAMBDA = (
    os.getenv("FORECAST_PROXY_ARN")                      # preferred
    or os.getenv("FORECAST_LAMBDA_ARN")
    or os.getenv("FORECAST_PROXY_FUNCTION")
    or os.getenv("FORECAST_FUNCTION_NAME")
    or "forecasting-proxy"
)
DEFAULT_DAYS = int(os.getenv("FORECAST_DEFAULT_DAYS", "90"))


def _robust_json_loads(maybe_json):
    if isinstance(maybe_json, (dict, list)):
        return maybe_json
    if isinstance(maybe_json, (bytes, bytearray)):
        try:
            return json.loads(maybe_json.decode("utf-8"))
        except Exception:
            return {}
    if isinstance(maybe_json, str):
        try:
            return json.loads(maybe_json)
        except Exception:
            return {}
    return {}


def call_lambda(days: int | None = None) -> Dict[str, Any]:
    """Invoke forecasting-proxy and return the normalized dict."""
    days = int(days or DEFAULT_DAYS)
    payload = {"action": "forecast", "days": days}

    lam = boto3.client("lambda", region_name=REGION)
    raw = lam.invoke(
        FunctionName=FORECAST_LAMBDA,
        InvocationType="RequestResponse",
        Payload=json.dumps(payload).encode("utf-8"),
    )["Payload"].read()

    body = _robust_json_loads(raw)
    # Also handle API Gateway style {statusCode, body}
    if isinstance(body, dict) and body.get("statusCode") and "body" in body:
        body = _robust_json_loads(body["body"])
    return body


def _read_s3_csv(s3_uri: str) -> pd.DataFrame:
    """Load CSV from s3://... into a DataFrame."""
    if not s3_uri or not s3_uri.startswith("s3://"):
        return pd.DataFrame()
    _, rest = s3_uri.split("s3://", 1)
    bucket, key = rest.split("/", 1)
    s3 = boto3.client("s3", region_name=REGION)
    obj = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(BytesIO(obj["Body"].read()))


def _normalize_forecast_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make sure df has columns: date, cost_usd."""
    if df.empty:
        return df
    cols = {c.lower(): c for c in df.columns}
    if "date" in cols and "cost_usd" in cols:
        df = df.rename(columns={cols["date"]: "date", cols["cost_usd"]: "cost_usd"})
    elif "ds" in cols and "yhat" in cols:                # Prophet style
        df = df.rename(columns={cols["ds"]: "date", cols["yhat"]: "cost_usd"})
    elif "day" in cols and "spend_usd" in cols:
        df = df.rename(columns={cols["day"]: "date", cols["spend_usd"]: "cost_usd"})
    else:
        # best effort: first col = date, last col = value
        if len(df.columns) >= 2:
            df = df.rename(columns={df.columns[0]: "date", df.columns[-1]: "cost_usd"})

    if "date" in df.columns and "cost_usd" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
        df = df.dropna(subset=["cost_usd"]).sort_values("date")
        return df[["date", "cost_usd"]]
    return pd.DataFrame(columns=["date", "cost_usd"])


def extract_summary_and_series(raw: Dict[str, Any]) -> Tuple[dict, pd.DataFrame]:
    """
    Supports two shapes:
      1) {'summary': {...}, 'series': [...]}
      2) {'totals': {...}, 'artifacts': {'forecast_csv': 's3://...'}}
    Returns: (summary_dict, df_series)
    """
    # Shape 1
    if "summary" in raw or "series" in raw:
        summary = raw.get("summary", raw)
        daily = summary.get("daily_avg_usd") or summary.get("daily_average_usd") or summary.get("daily_avg")
        eom = summary.get("eom_usd") or summary.get("month_total_usd") or summary.get("eom")
        eoq = summary.get("eoq_usd") or summary.get("quarter_total_usd") or summary.get("eoq")

        series = raw.get("series") or raw.get("forecast") or summary.get("series") or []
        recs: List[dict] = []
        for it in series:
            d = it.get("date") or it.get("ds") or it.get("day") or it.get("ts")
            v = it.get("cost_usd") or it.get("spend_usd") or it.get("yhat") or it.get("value")
            if d is not None and v is not None:
                recs.append({"date": str(d)[:10], "cost_usd": float(v)})
        df = pd.DataFrame.from_records(recs).sort_values("date")
        return (
            {
                "daily_avg_usd": float(daily) if daily is not None else None,
                "eom_usd": float(eom) if eom is not None else None,
                "eoq_usd": float(eoq) if eoq is not None else None,
            },
            df,
        )

    # Shape 2 (your sample)
    totals = raw.get("totals", {}) or {}
    artifacts = raw.get("artifacts", {}) or {}
    daily_avg = (float(totals.get("d30_usd")) / 30.0) if totals.get("d30_usd") is not None else None
    eom_usd = float(totals.get("this_month_usd")) if totals.get("this_month_usd") is not None else None
    eoq_usd = float(totals.get("d90_usd")) if totals.get("d90_usd") is not None else None

    df = pd.DataFrame()
    if artifacts.get("forecast_csv"):
        try:
            df = _normalize_forecast_df(_read_s3_csv(artifacts["forecast_csv"]))
        except Exception:
            df = pd.DataFrame()

    return {"daily_avg_usd": daily_avg, "eom_usd": eom_usd, "eoq_usd": eoq_usd}, df
