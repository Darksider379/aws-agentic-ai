# lambda_function.py
# Bedrock Agent (OpenAPI tool) + API Gateway compatible Lambda
# - If invoked by a Bedrock Agent (OpenAPI API-style), returns the API-style schema
# - If invoked by a Bedrock Agent (function-style), returns the function-style schema
# - If invoked via API Gateway / direct invoke, returns statusCode/body

import os
import json
import time
import re
from typing import Optional, List, Dict, Tuple, Any

import boto3
from botocore.exceptions import ClientError

# ========= Config (Environment) =========
ATHENA_DB        = os.environ.get("ATHENA_DB", "cost_comparison")
ATHENA_TABLE     = os.environ.get("ATHENA_TABLE", "cross_cloud_offerings")
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT    = os.environ["ATHENA_OUTPUT"]  # required (s3://bucket/prefix)

athena = boto3.client("athena")

# ========= Heuristics for free-text inference =========
EQUIV = {
    "kubernetes": [r"kubernetes", r"\beks\b", r"\baks\b", r"\bgke\b"],
    "object_storage": [r"\bs3\b", r"blob storage", r"cloud storage", r"object storage"],
    "functions": [r"\blambda\b", r"azure functions", r"cloud functions"],
    "managed_sql": [r"rds", r"sql database", r"cloud sql"],
    "compute": [r"\bec2\b", r"virtual machine", r"compute engine", r"\bvm\b"],
}

def infer_group_from_text(q: Optional[str]) -> Optional[str]:
    ql = (q or "").lower()
    for group, pats in EQUIV.items():
        if any(re.search(p, ql) for p in pats):
            return group
    return None

# ========= Query building =========
def build_where_clause(payload: Dict[str, Any]) -> str:
    where = ["price_usd IS NOT NULL"]

    group = (payload.get("service_group") or "").lower()
    if not group and payload.get("query"):
        group = infer_group_from_text(payload["query"]) or ""

    if group == "kubernetes":
        where.append("("
                     "LOWER(service_name) LIKE '%kubernetes%' OR "
                     "LOWER(service_name) LIKE '%eks%' OR "
                     "LOWER(service_name) LIKE '%aks%' OR "
                     "LOWER(service_name) LIKE '%gke%'"
                     ")")
        where.append("LOWER(price_period) IN ('per hour','hour')")

    elif group == "object_storage":
        where.append("("
                     "LOWER(service_name) LIKE '%s3%' OR "
                     "LOWER(service_name) LIKE '%blob%' OR "
                     "LOWER(service_name) LIKE '%cloud storage%' OR "
                     "LOWER(service_name) LIKE '%object%'"
                     ")")
        where.append("(LOWER(price_period) LIKE '%month%')")
        where.append("(LOWER(unit_standardized) LIKE '%gb%')")

    elif group == "functions":
        where.append("("
                     "LOWER(service_name) LIKE '%lambda%' OR "
                     "LOWER(service_name) LIKE '%azure functions%' OR "
                     "LOWER(service_name) LIKE '%cloud functions%'"
                     ")")
        where.append("("
                     "LOWER(price_period) LIKE '%request%' OR "
                     "LOWER(unit_standardized) LIKE '%ms%' OR "
                     "LOWER(unit_standardized) LIKE '%gb-s%'"
                     ")")

    elif group == "managed_sql":
        where.append("("
                     "LOWER(service_name) LIKE '%rds%' OR "
                     "LOWER(service_name) LIKE '%sql database%' OR "
                     "LOWER(service_name) LIKE '%cloud sql%'"
                     ")")

    elif group == "compute":
        where.append("("
                     "LOWER(service_name) LIKE '%ec2%' OR "
                     "LOWER(service_name) LIKE '%virtual machine%' OR "
                     "LOWER(service_name) LIKE '%compute engine%' OR "
                     "LOWER(service_name) LIKE '%vm%'"
                     ")")
        where.append("LOWER(price_period) IN ('per hour','hour')")

    # If no recognized group but a name is supplied, fuzzy match service_name
    if not group and payload.get("service_name"):
        where.append(f"LOWER(service_name) LIKE '%{payload['service_name'].lower()}%'")

    # Provider filter
    if payload.get("providers"):
        provs = [f"'{str(p).strip()}'" for p in payload["providers"] if str(p).strip()]
        if provs:
            where.append(f"provider IN ({', '.join(provs)})")

    # Region filter
    if payload.get("regions"):
        regs = [f"'{str(r).strip()}'" for r in payload["regions"] if str(r).strip()]
        if regs:
            where.append(f"region IN ({', '.join(regs)})")

    return " AND ".join(where)

def build_query(payload: Dict[str, Any], top_k: int = 1) -> str:
    where = build_where_clause(payload)
    return f"""
    WITH ranked AS (
      SELECT provider, service_category, service_name, sku_hint, region,
             price_usd, unit_standardized, price_period, pricing_link,
             ROW_NUMBER() OVER (PARTITION BY provider ORDER BY price_usd ASC) AS rnk
      FROM {ATHENA_DB}.{ATHENA_TABLE}
      WHERE {where}
    )
    SELECT provider, service_name, region, price_usd, unit_standardized,
           price_period, pricing_link
    FROM ranked
    WHERE rnk <= {int(top_k)}
    ORDER BY price_usd ASC, provider ASC
    """

# ========= Athena execution =========
def run_athena(sql: str, timeout_s: int = 45) -> List[Dict[str, Any]]:
    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": ATHENA_DB},
        WorkGroup=ATHENA_WORKGROUP,
        ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
    )
    qid = resp["QueryExecutionId"]

    t0 = time.time()
    while True:
        qe = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]
        state = qe["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        if time.time() - t0 > timeout_s:
            raise RuntimeError(f"Athena query TIMEOUT (QueryExecutionId={qid})")
        time.sleep(0.5)

    if state != "SUCCEEDED":
        reason = qe["Status"].get("StateChangeReason", "Unknown")
        raise RuntimeError(f"Athena query {state}. Reason: {reason} (QueryExecutionId={qid})")

    res = athena.get_query_results(QueryExecutionId=qid)
    cols = [c["Label"] for c in res["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows: List[Dict[str, Any]] = []
    for row in res["ResultSet"]["Rows"][1:]:
        data = [c.get("VarCharValue") for c in row.get("Data", [])]
        rows.append({k: v for k, v in zip(cols, data)})

    for r in rows:
        if "price_usd" in r and r["price_usd"] is not None:
            try:
                r["price_usd"] = float(r["price_usd"])
            except Exception:
                pass
        if (r.get("price_period") or "").lower() == "hour":
            r["price_period"] = "per hour"
        if (r.get("price_period") or "").lower() == "month":
            r["price_period"] = "per month"

    return rows

# ========= Bedrock helpers =========
def _is_openapi_api_event(evt: dict) -> bool:
    return isinstance(evt, dict) and "messageVersion" in evt and "apiPath" in evt and "httpMethod" in evt

def _bedrock_api_ok_json(event: dict, obj: dict) -> dict:
    # Echo back EXACT apiPath + httpMethod from request
    s = json.dumps(obj)
    print(f"[dbg] respond API ok len={len(s)}")
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "apiPath": event["apiPath"],
            "httpMethod": event["httpMethod"],
            "httpStatusCode": 200,
            "responseBody": {
                "application/json": {
                    "body": s  # must be STRING
                }
            }
        }
    }

def _bedrock_api_err_text(event: dict, msg: str, code: int = 400) -> dict:
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "apiPath": event["apiPath"],
            "httpMethod": event["httpMethod"],
            "httpStatusCode": int(code),
            "responseBody": {
                "text/plain": { "body": msg }
            }
        }
    }

def _is_bedrock_function_event(evt: dict) -> bool:
    # Legacy/alternate function-style events (no apiPath/httpMethod)
    return isinstance(evt, dict) and "messageVersion" in evt and ("function" in evt or "operation" in evt or "operationId" in evt)

def _bedrock_fn_ok_json(event: dict, obj: dict) -> dict:
    s = json.dumps(obj)
    fn = event.get("function") or event.get("operation") or event.get("operationId") or "getCrossCloudPricing"
    ag = event.get("actionGroup", "openapi")
    print(f"[dbg] respond FN ok ag={ag} fn={fn} len={len(s)}")
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": ag,
            "function": fn,
            "functionResponse": {
                "responseBody": {
                    "JSON": { "body": s }  # must be STRING
                }
            }
        }
    }

def _bedrock_fn_err_text(event: dict, msg: str) -> dict:
    fn = event.get("function") or event.get("operation") or event.get("operationId") or "unknown"
    ag = event.get("actionGroup", "openapi")
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": ag,
            "function": fn,
            "functionResponse": {
                "responseBody": { "TEXT": { "body": msg } }
            }
        }
    }

def _extract_openapi_call(evt: dict) -> Tuple[str, Dict[str, Any]]:
    """
    Merge parameters (list or dict) + requestBody/requestBodyJson.
    Return (operationId, params)
    """
    op = evt.get("function") or evt.get("operationId") or evt.get("operation") or "getCrossCloudPricing"

    def _val(v):
        if not isinstance(v, dict): return v
        if "stringValue" in v:  return v["stringValue"]
        if "listValue"   in v:  return [_val(x) for x in v["listValue"]]
        if "boolValue"   in v:  return bool(v["boolValue"])
        if "numberValue" in v:  return v["numberValue"]
        return v

    params: Dict[str, Any] = {}

    # parameters as list
    if isinstance(evt.get("parameters"), list):
        for item in evt["parameters"]:
            name = item.get("name")
            if name:
                params[name] = _val(item.get("value", {}))

    # parameters as dict
    if isinstance(evt.get("parameters"), dict):
        for k, v in evt["parameters"].items():
            params[k] = _val(v)

    # requestBody / requestBodyJson
    rb = evt.get("requestBody") or evt.get("requestBodyJson")
    if isinstance(rb, str):
        try:
            rb = json.loads(rb)
        except Exception:
            rb = None
    if isinstance(rb, dict):
        for k, v in rb.items():
            params.setdefault(k, v)

    print("[dbg] extracted op:", op, "params keys:", list(params.keys()))
    return op, params

def _params_to_payload_for_pricing(params: Dict[str, Any], input_text: Optional[str]) -> Dict[str, Any]:
    prov = params.get("providers")
    regs = params.get("regions")
    if isinstance(prov, str): prov = [prov]
    if isinstance(regs, str): regs = [regs]

    payload: Dict[str, Any] = {
        "service_group": params.get("service_group"),
        "service_name": params.get("service_name"),
        "providers": prov,
        "regions": regs,
        "top_k": params.get("top_k", 1),
        "query": input_text or params.get("query"),
    }
    # drop null/empty
    return {k: v for k, v in payload.items() if v not in (None, "", [], {})}

# ========= Main handler =========
def lambda_handler(event, context):
    try:
        # --- Debug who called us ---
        try:
            print("[invoker] keys:", list(event.keys()) if isinstance(event, dict) else type(event))
        except Exception:
            pass

        # --- 1) Bedrock OpenAPI tool (API-style) path ---
        if _is_openapi_api_event(event):
            print("[dbg] API event:", event.get("apiPath"), event.get("httpMethod"))
            op, params = _extract_openapi_call(event)
            input_text = event.get("inputText") or event.get("input")

            payload = _params_to_payload_for_pricing(params, input_text)
            # accept "1" or 1
            try:
                top_k = int(str(payload.get("top_k", 1)))
            except Exception:
                top_k = 1

            if not any(payload.get(k) for k in ("service_group", "service_name", "query")):
                return _bedrock_api_err_text(
                    event,
                    "Provide service_group (kubernetes/object_storage/functions/managed_sql/compute) "
                    "or service_name; optional providers/regions/top_k."
                )

            sql = build_query(payload, top_k=top_k)
            print("[ATHENA] SQL (first 300):", sql[:300])
            rows = run_athena(sql)
            cheapest = min(rows, key=lambda r: r.get("price_usd", 1e99)) if rows else None

            assumptions: List[str] = []
            g = (payload.get("service_group") or "").lower()
            if g == "kubernetes":
                assumptions.append("Compared control-plane hourly fees only; node/compute costs not included.")
            elif g == "object_storage":
                assumptions.append("Compared per-GB per-month storage prices.")
            elif g == "functions":
                assumptions.append("Compared per-request or per-GB-second prices.")
            elif g == "compute":
                assumptions.append("Compared VM/instance hourly rates; size normalization not applied.")

            resp = {"query": payload, "assumptions": assumptions, "results": rows, "cheapest": cheapest}
            return _bedrock_api_ok_json(event, resp)

        # --- 2) Bedrock function-style path (no apiPath/httpMethod) ---
        if _is_bedrock_function_event(event):
            op, params = _extract_openapi_call(event)
            input_text = event.get("inputText") or event.get("input")
            payload = _params_to_payload_for_pricing(params, input_text)
            try:
                top_k = int(str(payload.get("top_k", 1)))
            except Exception:
                top_k = 1

            if not any(payload.get(k) for k in ("service_group", "service_name", "query")):
                return _bedrock_fn_err_text(
                    event,
                    "Please provide a service_group (kubernetes, object_storage, functions, managed_sql, compute) "
                    "or a service_name (e.g., EKS, S3). You may also add regions/providers/top_k."
                )

            sql = build_query(payload, top_k=top_k)
            print("[ATHENA] SQL (first 300):", sql[:300])
            rows = run_athena(sql)
            cheapest = min(rows, key=lambda r: r.get("price_usd", 1e99)) if rows else None

            assumptions: List[str] = []
            g = (payload.get("service_group") or "").lower()
            if g == "kubernetes":
                assumptions.append("Compared control-plane hourly fees only; node/compute costs not included.")
            elif g == "object_storage":
                assumptions.append("Compared per-GB per-month storage prices.")
            elif g == "functions":
                assumptions.append("Compared per-request or per-GB-second prices.")
            elif g == "compute":
                assumptions.append("Compared VM/instance hourly rates; size normalization not applied.")

            resp = {"query": payload, "assumptions": assumptions, "results": rows, "cheapest": cheapest}
            return _bedrock_fn_ok_json(event, resp)

        # --- 3) API Gateway / direct invoke path ---
        body = event.get("body") if isinstance(event, dict) else None
        payload = json.loads(body) if isinstance(body, str) else (body or event or {})
        try:
            top_k = int(str((payload or {}).get("top_k", 1)))
        except Exception:
            top_k = 1

        if not any((payload or {}).get(k) for k in ("service_group", "service_name", "query")):
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "Provide service_group, service_name, or query"})
            }

        sql = build_query(payload, top_k=top_k)
        print("[ATHENA] SQL (first 300):", sql[:300])
        rows = run_athena(sql)
        cheapest = min(rows, key=lambda r: r.get("price_usd", 1e99)) if rows else None

        assumptions: List[str] = []
        g = (payload.get("service_group") or "").lower()
        if g == "kubernetes":
            assumptions.append("Compared control-plane hourly fees only; node/compute costs not included.")
        elif g == "object_storage":
            assumptions.append("Compared per-GB per-month storage prices.")
        elif g == "functions":
            assumptions.append("Compared per-request or per-GB-second prices.")
        elif g == "compute":
            assumptions.append("Compared VM/instance hourly rates; size normalization not applied.")

        resp = {"query": payload, "assumptions": assumptions, "results": rows, "cheapest": cheapest}
        return {"statusCode": 200, "headers": {"Content-Type": "application/json"},
                "body": json.dumps(resp)}

    except Exception as e:
        # Return Agent-shaped error if the caller was Bedrock; else API-GW 500
        try:
            if _is_openapi_api_event(event):
                return _bedrock_api_err_text(event, f"Error: {str(e)}", code=500)
            if _is_bedrock_function_event(event):
                return _bedrock_fn_err_text(event, f"Error: {str(e)}")
        except Exception:
            pass
        return {"statusCode": 500, "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": str(e)})}
