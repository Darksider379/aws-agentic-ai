# file: lambda_function.py
import os
import json
import time
import re
import boto3
from botocore.exceptions import ClientError

ATHENA_DB        = os.environ.get("ATHENA_DB", "cost_comparison")
ATHENA_TABLE     = os.environ.get("ATHENA_TABLE", "cross_cloud_offerings")
ATHENA_WORKGROUP = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT    = os.environ["ATHENA_OUTPUT"]  # required, e.g. s3://your-bucket/athena/out/

athena = boto3.client("athena")

# Simple equivalence map to normalize fuzzy asks to a "group"
EQUIV = {
    "kubernetes": [r"kubernetes", r"\beks\b", r"\baks\b", r"\bgke\b"],
    "object_storage": [r"\bs3\b", r"blob storage", r"cloud storage", r"object storage"],
    "functions": [r"\blambda\b", r"azure functions", r"cloud functions"],
    "managed_sql": [r"rds", r"sql database", r"cloud sql"],
}

def infer_group_from_text(q: str) -> str | None:
    ql = (q or "").lower()
    for group, pats in EQUIV.items():
        if any(re.search(p, ql) for p in pats):
            return group
    return None

def build_where_clause(payload: dict) -> str:
    """Returns SQL WHERE clause string."""
    where = ["price_usd IS NOT NULL"]

    # service_group: either explicit or inferred from free text
    service_group = payload.get("service_group")
    if not service_group and "query" in payload:
        service_group = infer_group_from_text(payload["query"])

    if service_group:
        group = service_group.lower()
        if group == "kubernetes":
            where.append("(LOWER(service_name) LIKE '%kubernetes%' OR LOWER(service_name) LIKE '%eks%' OR LOWER(service_name) LIKE '%aks%' OR LOWER(service_name) LIKE '%gke%')")
        elif group == "object_storage":
            where.append("(LOWER(service_name) LIKE '%s3%' OR LOWER(service_name) LIKE '%blob%' OR LOWER(service_name) LIKE '%cloud storage%' OR LOWER(service_name) LIKE '%object%')")
        elif group == "functions":
            where.append("(LOWER(service_name) LIKE '%lambda%' OR LOWER(service_name) LIKE '%azure functions%' OR LOWER(service_name) LIKE '%cloud functions%')")
        elif group == "managed_sql":
            where.append("(LOWER(service_name) LIKE '%rds%' OR LOWER(service_name) LIKE '%sql database%' OR LOWER(service_name) LIKE '%cloud sql%')")
    elif "service_name" in payload:
        where.append(f"LOWER(service_name) LIKE '%{payload['service_name'].lower()}%'")

    # provider filter (optional)
    providers = payload.get("providers")
    if providers:
        prov_list = [f"'{x.strip()}'" for x in providers if x and x.strip()]
        if prov_list:
            where.append(f"provider IN ({', '.join(prov_list)})")

    # region filter (optional)
    regions = payload.get("regions")
    if regions:
        reg_list = [f"'{x.strip()}'" for x in regions if x and x.strip()]
        if reg_list:
            where.append(f"region IN ({', '.join(reg_list)})")

    # price_period filter (optional)
    price_period = payload.get("price_period")
    if price_period:
        where.append(f"LOWER(price_period) = '{price_period.lower()}'")

    # unit filter (optional)
    unit = payload.get("unit_standardized")
    if unit:
        where.append(f"LOWER(unit_standardized) = '{unit.lower()}'")

    return " AND ".join(where)

def build_query(payload: dict, top_k: int = 1) -> str:
    """
    Build a query that gets the cheapest row per provider (AWS/Azure/GCP).
    """
    where = build_where_clause(payload)
    sql = f"""
    WITH ranked AS (
      SELECT
        provider,
        service_category,
        service_name,
        sku_hint,
        region,
        price_usd,
        unit_standardized,
        price_period,
        pricing_link,
        ROW_NUMBER() OVER (PARTITION BY provider ORDER BY price_usd ASC) AS rnk
      FROM {ATHENA_DB}.{ATHENA_TABLE}
      WHERE {where}
    )
    SELECT provider, service_name, region, price_usd, unit_standardized, price_period, pricing_link
    FROM ranked
    WHERE rnk <= {int(top_k)}
    ORDER BY price_usd ASC, provider ASC
    """
    return sql

def run_athena(sql: str, timeout_s: int = 30) -> list[dict]:
    """
    Executes Athena SQL and returns rows as list of dicts.
    """
    try:
        resp = athena.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": ATHENA_DB},
            WorkGroup=ATHENA_WORKGROUP,
            ResultConfiguration={"OutputLocation": ATHENA_OUTPUT},
        )
    except ClientError as e:
        raise RuntimeError(f"Athena start failed: {e}")

    qid = resp["QueryExecutionId"]

    # Poll until completion
    t0 = time.time()
    state = "RUNNING"
    while time.time() - t0 < timeout_s:
        st = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]["State"]
        if st in ("SUCCEEDED", "FAILED", "CANCELLED"):
            state = st
            break
        time.sleep(0.5)

    if state != "SUCCEEDED":
        raise RuntimeError(f"Athena query {state}. QueryExecutionId={qid}")

    # Fetch results
    res = athena.get_query_results(QueryExecutionId=qid)
    cols = [c["Label"] for c in res["ResultSet"]["ResultSetMetadata"]["ColumnInfo"]]
    rows = []
    for row in res["ResultSet"]["Rows"][1:]:
        data = [c.get("VarCharValue") for c in row.get("Data", [])]
        rows.append({k: v for k, v in zip(cols, data)})

    # paginate if needed
    next_token = res.get("NextToken")
    while next_token:
        res = athena.get_query_results(QueryExecutionId=qid, NextToken=next_token)
        for row in res["ResultSet"]["Rows"]:
            data = [c.get("VarCharValue") for c in row.get("Data", [])]
            if len(data) != len(cols):
                continue
            rows.append({k: v for k, v in zip(cols, data)})
        next_token = res.get("NextToken")

    # cast price to float for convenience
    for r in rows:
        if "price_usd" in r and r["price_usd"] is not None:
            try:
                r["price_usd"] = float(r["price_usd"])
            except Exception:
                pass
    return rows

def lambda_handler(event, context):
    try:
        body = event.get("body")
        payload = json.loads(body) if isinstance(body, str) else (body or event)
        top_k = int(payload.get("top_k", 1))

        sql = build_query(payload, top_k=top_k)
        rows = run_athena(sql)

        cheapest = None
        if rows:
            cheapest = min(rows, key=lambda r: (float(r["price_usd"]) if r.get("price_usd") else 1e99))

        resp = {
            "query": payload,
            "results": rows,
            "cheapest": cheapest
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(resp)
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)})
        }
