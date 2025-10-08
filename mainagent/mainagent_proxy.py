# lambda_function.py
# Proxy: returns latest/requested summary.json and (optionally) triggers finops-agent with strong checks.
###Rename this file to lambda_function.py
import os
import json
import uuid
import boto3
from datetime import datetime, timezone
from botocore.exceptions import ClientError, BotoCoreError

S3_BUCKET       = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX  = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")
AGENT_FN        = os.environ["AGENT_FUNCTION_NAME"]
AWS_REGION      = os.environ["AWS_REGION"]  # provided by Lambda
TRIGGER_DEFAULT = os.environ.get("TRIGGER_BY_DEFAULT", "true").lower() in ("1", "true", "yes")
WRITE_TRIGGER_MARKER = os.environ.get("WRITE_TRIGGER_MARKER", "false").lower() in ("1","true","yes")

s3  = boto3.client("s3", region_name=AWS_REGION)
lam = boto3.client("lambda", region_name=AWS_REGION)

# cache across warm invocations
_AGENT_EXISTS = None

def _resp(status: int, body_obj, cors=True):
    headers = {"Content-Type":"application/json","Cache-Control":"no-store, no-cache, must-revalidate"}
    if cors:
        headers.update({
            "Access-Control-Allow-Origin":"*",
            "Access-Control-Allow-Headers":"Content-Type,Authorization",
            "Access-Control-Allow-Methods":"GET,OPTIONS",
        })
    return {"statusCode":status,"headers":headers,
            "body":json.dumps(body_obj, separators=(",",":"), ensure_ascii=False)}

def _list_latest_run_id() -> str | None:
    prefix = f"{RESULTS_PREFIX.strip('/')}/runs/"
    paginator = s3.get_paginator("list_objects_v2")
    latest = None
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            p = cp.get("Prefix")  # e.g. cost-agent-v2/runs/20251007T123652Z/
            run_id = p.split("/")[-2] if p.endswith("/") else p.split("/")[-1]
            if not latest or run_id > latest:
                latest = run_id
    return latest

def _get_summary_json(run_id: str):
    key = f"{RESULTS_PREFIX.strip('/')}/runs/{run_id}/summary.json"
    try:
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        raw = obj["Body"].read()
        return json.loads(raw), f"s3://{S3_BUCKET}/{key}"
    except ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey","404"):
            return None, None
        raise

def _ensure_agent_exists() -> bool:
    global _AGENT_EXISTS
    if _AGENT_EXISTS is not None:
        return _AGENT_EXISTS
    try:
        lam.get_function(FunctionName=AGENT_FN)
        _AGENT_EXISTS = True
    except ClientError as e:
        print(f"[proxy] get_function failed: {e}")
        _AGENT_EXISTS = False
    return _AGENT_EXISTS

def _dry_run_permission_check() -> tuple[bool, int | None, str | None]:
    """Use InvocationType='DryRun' to validate permission; expect 204."""
    try:
        resp = lam.invoke(FunctionName=AGENT_FN, InvocationType="DryRun")
        return (resp.get("StatusCode") == 204, resp.get("StatusCode"), None)
    except ClientError as e:
        return (False, None, f"ClientError:{e.response.get('Error',{}).get('Code')}")
    except BotoCoreError as e:
        return (False, None, f"BotoCoreError:{e}")

def _write_trigger_marker(correlation_id: str, payload: dict):
    """Optional breadcrumb to S3 for audit/traceability."""
    if not WRITE_TRIGGER_MARKER:
        return
    key = f"{RESULTS_PREFIX.strip('/')}/triggers/{correlation_id}.json"
    doc = {
        "correlation_id": correlation_id,
        "agent_function": AGENT_FN,
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "payload": payload
    }
    try:
        s3.put_object(Bucket=S3_BUCKET, Key=key,
                      Body=json.dumps(doc, separators=(",",":")).encode("utf-8"),
                      ContentType="application/json",
                      CacheControl="no-store, no-cache, must-revalidate")
        print(f"[proxy] wrote trigger marker s3://{S3_BUCKET}/{key}")
    except Exception as e:
        print(f"[warn] trigger marker write failed: {e}")

def _trigger_agent_async(payload: dict | None = None) -> dict:
    """
    Strongly-checked async trigger:
      1) ensure function exists
      2) permission dry-run (expect 204)
      3) async invoke (expect 202)
      4) return diagnostic struct incl. correlation_id
    """
    diag = {
        "exists": False,
        "dry_run_ok": False,
        "dry_run_status": None,
        "dry_run_error": None,
        "invoked": False,
        "invoke_status": None,
        "correlation_id": None,
    }

    if not _ensure_agent_exists():
        return diag  # exists False

    diag["exists"] = True

    ok, st, err = _dry_run_permission_check()
    diag["dry_run_ok"] = ok
    diag["dry_run_status"] = st
    diag["dry_run_error"] = err
    if not ok:
        return diag  # stop here; no permission

    cid = str(uuid.uuid4())
    body = {"source":"proxy","reason":"refresh","correlation_id":cid}
    if payload: body.update(payload)

    try:
        resp = lam.invoke(FunctionName=AGENT_FN,
                          InvocationType="Event",
                          Payload=json.dumps(body).encode("utf-8"))
        diag["invoke_status"] = resp.get("StatusCode")
        diag["invoked"] = (resp.get("StatusCode") == 202)
        diag["correlation_id"] = cid
        print(f"[proxy] invoked {AGENT_FN} async status={diag['invoke_status']} cid={cid}")
        _write_trigger_marker(cid, body)
    except Exception as e:
        print(f"[warn] agent async invoke failed: {e} cid={cid}")
        diag["invoke_status"] = None
        diag["invoked"] = False
        diag["correlation_id"] = cid

    return diag

def lambda_handler(event, context):
    # CORS preflight
    if isinstance(event, dict) and event.get("httpMethod") == "OPTIONS":
        return _resp(200, {"ok": True})

    qs = (event.get("queryStringParameters") or {}) if isinstance(event, dict) else {}
    requested_run = (qs.get("run_id") or "latest").strip().lower()
    trigger_qs = qs.get("trigger")
    should_trigger = TRIGGER_DEFAULT if trigger_qs is None else (trigger_qs.lower() in ("1","true","yes"))

    # Resolve run id
    if requested_run in ("", "latest"):
        run_id = _list_latest_run_id()
        if not run_id:
            # bootstrap: optionally trigger agent to create first run
            diag = _trigger_agent_async({"reason":"bootstrap-no-runs"}) if should_trigger else None
            return _resp(404, {"error":"no_runs_found","message":"No summary runs found yet.","trigger_diagnostics":diag})
    else:
        run_id = requested_run

    # Fetch summary
    summary, s3_uri = _get_summary_json(run_id)
    if not summary:
        diag = _trigger_agent_async({"reason":"missing-run","run_id":run_id}) if should_trigger else None
        return _resp(404, {"error":"not_found","message":f"summary.json not found for run_id={run_id}","trigger_diagnostics":diag})

    # Trigger refresh AFTER serving current result (snappy UX)
    diag = _trigger_agent_async({"prev_run_id":run_id}) if should_trigger else None

    return _resp(200, {
        "run_id": run_id,
        "summary_s3_uri": s3_uri,
        "summary": summary,
        "triggered": bool(diag and diag.get("invoked")),
        "trigger_diagnostics": diag  # includes exists, dry_run_ok(204), invoke_status(202), correlation_id
    })
