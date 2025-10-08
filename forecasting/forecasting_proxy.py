# lambda_function.py
# Forecasting proxy for Bedrock Agents (OpenAPI action group) and API Gateway.
#
# - If invoked by API Gateway: returns {statusCode, body} (proxy integration).
# - If invoked by Bedrock (OpenAPI action group): returns the REQUIRED Bedrock envelope.
#
# The Lambda calls your forecasting backend either via:
#   INVOKE_MODE=api     -> POST to FORECAST_API_URL (API Gateway/LB/etc.)
#   INVOKE_MODE=lambda  -> Invoke FORECAST_LAMBDA_ARN
#
# ===== Env vars to set on this Lambda =====
#   INVOKE_MODE=api | lambda
#   FORECAST_API_URL=https://<api-id>.execute-api.us-east-1.amazonaws.com/default/forecasting
#   FORECAST_LAMBDA_ARN=arn:aws:lambda:us-east-1:<acct>:function:forecasting
#
# NOTE: Ensure region spelling is correct ("us-east-1" with dash).

import json, os, math, urllib.request, boto3
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# -------- configuration --------
INVOKE_MODE = os.environ.get("INVOKE_MODE", "api").lower()  # "api" or "lambda"
FORECAST_API_URL = os.environ.get("FORECAST_API_URL")
FORECAST_LAMBDA_ARN = os.environ.get("FORECAST_LAMBDA_ARN")
_ALLOWED = {30, 60, 90}

_lambda = boto3.client("lambda")

# ---------- sanitizers ----------
def _to_float_or_none(x) -> Optional[float]:
    try:
        f = float(x)
        return f if math.isfinite(f) else None
    except Exception:
        return None

def _round2(x) -> Optional[float]:
    f = _to_float_or_none(x)
    return round(f, 2) if f is not None else None

def _sanitize(obj: Any) -> Any:
    """Coerce to vanilla JSON: no Decimal/NaN/datetime/bytes."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import decimal
        if isinstance(obj, decimal.Decimal):
            return _to_float_or_none(obj)
    except Exception:
        pass
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    return str(obj)

# ---------- helpers ----------
def _coerce_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _safe_json_loads(txt: str) -> Any:
    try:
        return json.loads(txt)
    except Exception:
        return {"error": "invalid_json", "raw_snippet": (txt or "")[:200]}

# ---------- backend callers ----------
def _call_backend_api(horizon_days: int) -> Dict[str, Any]:
    if not FORECAST_API_URL:
        raise RuntimeError("FORECAST_API_URL not set")
    req = urllib.request.Request(
        FORECAST_API_URL,
        method="POST",
        data=json.dumps({"horizon_days": horizon_days}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        text = resp.read().decode("utf-8")
        j = _safe_json_loads(text)
        # If API GW returns envelope for some reason, unwrap
        if isinstance(j, dict) and "body" in j:
            bd = j.get("body")
            return _safe_json_loads(bd) if isinstance(bd, str) else bd
        return j

def _call_backend_lambda(horizon_days: int) -> Dict[str, Any]:
    if not FORECAST_LAMBDA_ARN:
        raise RuntimeError("FORECAST_LAMBDA_ARN not set")
    inv = _lambda.invoke(
        FunctionName=FORECAST_LAMBDA_ARN,
        InvocationType="RequestResponse",
        Payload=json.dumps({"horizon_days": horizon_days}).encode("utf-8"),
    )
    raw = inv["Payload"].read().decode("utf-8").strip()
    j = _safe_json_loads(raw)
    if isinstance(j, dict) and "body" in j:
        bd = j.get("body")
        return _safe_json_loads(bd) if isinstance(bd, str) else bd
    return j

def _call_backend(horizon_days: int) -> Dict[str, Any]:
    return _call_backend_lambda(horizon_days) if INVOKE_MODE == "lambda" else _call_backend_api(horizon_days)

# ---------- budget verdict ----------
def _budget_verdict(forecast_total_usd: Optional[float], budget_usd: Optional[float]) -> Optional[Dict[str, Any]]:
    if forecast_total_usd is None or budget_usd is None:
        return None
    delta = forecast_total_usd - budget_usd
    return {
        "budget_usd": _round2(budget_usd),
        "status": "over" if delta > 0 else "under",
        "delta_usd": _round2(abs(delta)),
    }

# ---------- core proxy ----------
def _run_proxy(event: Dict[str, Any]) -> Dict[str, Any]:
    horizons: List[int] = []
    budget_usd: Optional[float] = None
    include_values = False

    # parse inputs from dict (works for API GW body, Bedrock requestBody, or direct invoke)
    if isinstance(event, dict):
        rb = event.get("requestBody", {}).get("content", {}).get("application/json", {}).get("properties", {})
        src = rb if isinstance(rb, dict) else event

        if isinstance(src.get("horizons"), list):
            for h in src["horizons"]:
                hi = _coerce_int(h)
                if hi in _ALLOWED:
                    horizons.append(hi)

        if not horizons:
            hi = _coerce_int(src.get("horizon_days"))
            if hi in _ALLOWED:
                horizons = [hi]

        if not horizons:
            horizons = [90]  # default

        b = src.get("budget_usd")
        if isinstance(b, (int, float, str)):
            try:
                budget_usd = float(b)
            except Exception:
                budget_usd = None

        include_values = bool(src.get("include_values", False))

    backend_horizon = 90 if (len(horizons) > 1 or 90 in horizons) else horizons[0]

    backend = _call_backend(backend_horizon)
    summary = backend.get("summary", {}) if isinstance(backend, dict) else {}
    run_id = backend.get("run_id") or summary.get("run_id")
    totals = summary.get("totals", {}) or {}
    artifacts = summary.get("artifacts", {}) or {}

    out: Dict[str, Any] = {
        "run_id": run_id,
        "horizons_requested": horizons,
        "backend_horizon_used_days": backend_horizon,
        "totals": {
            "d30_usd": _round2(totals.get("forecast_sum_next_30d_usd")),
            "d60_usd": _round2(totals.get("forecast_sum_next_60d_usd")),
            "d90_usd": _round2(totals.get("forecast_sum_next_90d_usd")),
            "this_month_usd": _round2(totals.get("this_month_forecast_usd")),
        },
        "artifacts": {
            "history_csv": artifacts.get("history_csv"),
            "forecast_csv": artifacts.get("forecast_csv"),
            "athena_forecast_table": artifacts.get("athena_forecast_table"),
            "athena_partition_run_id": artifacts.get("athena_partition_run_id"),
            "athena_partition_s3": artifacts.get("athena_partition_s3"),
        },
        "note": (
            "Computed from a single 90-day backend run"
            if backend_horizon == 90 else f"Computed from a {backend_horizon}-day backend run"
        ),
    }

    # ---- budget comparison fix ----
    if backend_horizon == 30:
        horizon_forecast = out["totals"]["d30_usd"]
    elif backend_horizon == 60:
        horizon_forecast = out["totals"]["d60_usd"]
    else:
        horizon_forecast = out["totals"]["d90_usd"]

    bv = _budget_verdict(horizon_forecast, budget_usd)
    if bv:
        out["budget"] = bv
        out["note_budget"] = f"Compared forecasted {backend_horizon}-day total vs budget_usd"

    if include_values:
        out["values_source"] = "Download per-day values from artifacts.history_csv and artifacts.forecast_csv"

    return _sanitize(out)

# ---------- Bedrock OpenAPI response wrapper ----------
def _ok_openapi(event: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    # Bedrock requires this exact envelope for OpenAPI action groups.
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event.get("actionGroup") or "forecasting",
            "apiPath": event.get("apiPath") or "/forecasting-proxy",
            "httpMethod": event.get("httpMethod") or "POST",
            "responseBody": {
                "application/json": {
                    "body": _sanitize(payload)
                }
            }
        }
    }

# ---------- entrypoint ----------
def lambda_handler(event, context):
    try:
        if isinstance(event, dict):
            print("DEBUG keys:", list(event.keys())[:12])
    except Exception:
        pass

    try:
        # 1) API Gateway (proxy)
        if isinstance(event, dict) and ("httpMethod" in event and "requestContext" in event):
            body = event.get("body")
            data = json.loads(body) if isinstance(body, str) and body else (body or {})
            result = _run_proxy(data)
            return {
                "statusCode": 200,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps(result)
            }

        # 2) Bedrock Agent (OpenAPI action group)
        if isinstance(event, dict) and ("agent" in event or "apiPath" in event or "actionGroup" in event):
            result = _run_proxy(event)
            return _ok_openapi(event, result)

        # 3) Fallback (CLI/test)
        result = _run_proxy(event if isinstance(event, dict) else {})
        return result

    except Exception as e:
        err = {"error": str(e), "type": type(e).__name__}
        if isinstance(event, dict) and ("agent" in event or "apiPath" in event or "actionGroup" in event):
            return _ok_openapi(event, err)
        return err
