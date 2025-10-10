# mainagent.py
# -------------------------------------------------
# 0) Load env before imports that read os.environ
import os, sys
from pathlib import Path
from datetime import datetime, timezone

def load_env_file(path: str = "config.ini"):
    """Load KEY=VALUE .ini/.env into os.environ (dotenv if available, else fallback)."""
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

# support: python mainagent.py --env config.ini
env_path = "config.ini"
if "--env" in sys.argv:
    i = sys.argv.index("--env")
    if i + 1 < len(sys.argv):
        env_path = sys.argv[i + 1]
load_env_file(env_path)

# -------------------------------------------------
# 1) Imports that depend on env
import json
import re
from typing import List, Dict, Optional, Tuple
import pandas as pd
import boto3

# Required env
ATHENA_DB         = os.environ["ATHENA_DB"]
ATHENA_TABLE      = os.environ["ATHENA_TABLE"]
ATHENA_WORKGROUP  = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT     = os.environ["ATHENA_OUTPUT"]
RESULTS_BUCKET    = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX    = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")
ATHENA_RECS_TABLE = os.environ.get("ATHENA_RECS_TABLE", "recommendations_v2")

# Optional Bedrock/LLM
USE_LLM          = os.environ.get("USE_LLM", "false").lower() == "true"
BEDROCK_REGION   = os.environ.get("BEDROCK_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# Optional LLM caps from env/config.ini
LLM_ABS_CAP      = float(os.environ.get("LLM_ABS_CAP", "100000"))    # $/month
LLM_REL_CAP_PCT  = float(os.environ.get("LLM_REL_CAP_PCT", "30"))    # percent

# Enrichment look-back days (dataset-aware; see _cur_latest_date_str)
LOOKBACK_DAYS    = int(os.environ.get("ENRICH_LOOKBACK_DAYS", "365"))

# AWS clients (region auto from env in Lambda)
athena  = boto3.client("athena", region_name=os.environ.get("AWS_REGION", None))
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)
s3      = boto3.client("s3",     region_name=os.environ.get("AWS_REGION", None))

# -------------------------------------------------
# 2) Local modules
from functions.hourly_usage_breakdown import hourly_usage_breakdown
from recommendations.recommend_ec2_rightsize import recommend_ec2_rightsize
from recommendations.recommend_s3_tiering import recommend_s3_tiering
from recommendations.recommend_snapshot_hygiene import recommend_snapshot_hygiene
from recommendations.write_recommendations_csv_to_s3 import write_recommendations_csv_to_s3
from recommendations.build_recommendations_table_if_needed import build_recommendations_table_if_needed
from functions.run_athena import run_athena

# -------------------------------------------------
# 3) Query cost summary for LLM (top N contributors)
def query_cost_summary_topn(limit: int = 20) -> pd.DataFrame:
    sql = f"""
    SELECT line_item_product_code AS service,
           product_region         AS region,
           SUM(line_item_blended_cost) AS cost_usd
    FROM {ATHENA_TABLE}
    GROUP BY 1,2
    ORDER BY cost_usd DESC
    LIMIT {int(limit)}
    """
    df = run_athena(sql)
    if df is None or df.empty:
        return pd.DataFrame(columns=["service","region","cost_usd"])
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    return df

# -------------------------------------------------
# 4) Category-to-service mapping for LLM relative caps
CATEGORY_TO_SERVICE = {
    "EC2 Right-size": ["Amazon Elastic Compute Cloud", "AmazonEC2", "EC2"],
    "S3 Storage Optimization": ["Amazon Simple Storage Service", "AmazonS3", "Amazon S3"],
    "CloudFront Optimization": ["Amazon CloudFront", "AmazonCloudFront", "CloudFront"],
    "EBS Optimization": ["Amazon Elastic Block Store", "AmazonEBS", "EBS"],
    "Lambda Optimization": ["AWS Lambda", "AWSLambda", "Lambda"],
}

# -------------------------------------------------
# 5) LLM helpers via Bedrock Converse (Nova Pro)
def _extract_json_array(text: str) -> str:
    """Extract a JSON array from arbitrary text."""
    if not text:
        raise ValueError("Empty model output")
    fence = re.search(r"```json\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    start = text.find("[")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1].strip()
    lines = [ln.strip().rstrip(",") for ln in text.splitlines() if ln.strip()]
    objs = []
    for ln in lines:
        try:
            obj = json.loads(ln)
            if isinstance(obj, dict):
                objs.append(obj)
        except Exception:
            pass
    if objs:
        return json.dumps(objs)
    raise ValueError(f"Could not find JSON array in model output. First 300 chars:\n{text[:300]}")

def _converse_json_array(instruction: str, system_text: str | None = None,
                         max_tokens: int = 900, temperature: float = 0.1, top_p: float = 0.9) -> list:
    """Ask Nova Pro for a STRICT JSON array; try JSON mode, then strict retry with few-shot."""
    def _read_text_from_converse(resp: dict) -> str:
        out_msg = resp.get("output", {}).get("message", {})
        txt = ""
        for block in out_msg.get("content", []):
            if "text" in block:
                txt += block["text"]
        return txt.strip()

    messages = [{"role": "user", "content": [{"text": instruction}]}]
    kwargs = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature, "topP": top_p},
        "responseFormat": {"type": "JSON"},
    }
    if system_text:
        kwargs["system"] = [{"text": system_text}]
    try:
        resp = bedrock.converse(**kwargs)
        txt = _read_text_from_converse(resp)
        try:
            val = json.loads(txt)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                for _, v in val.items():
                    if isinstance(v, list):
                        return v
        except Exception:
            pass
        arr = _extract_json_array(txt)
        return json.loads(arr)
    except Exception:
        pass

    # Strict retry
    strict_system = ((system_text + " ") if system_text else "") + \
        "Return ONLY a JSON array. Do not add any prose, backticks, or explanations."
    few_shot = (
        "You must respond with a JSON array only.\n"
        '[{"category":"EC2 Right-size","subtype":"m5.large→m7g.large","recommendation":"...","estimated_saving_usd":100.0,"region":"us-east-1"}]'
    )
    messages2 = [
        {"role": "user", "content": [{"text": few_shot}]},
        {"role": "assistant", "content": [{"text":
            '[{"category":"EC2 Right-size","subtype":"m5.large→m7g.large","recommendation":"...","estimated_saving_usd":100.0,"region":"us-east-1"}]'}]},
        {"role": "user", "content": [{"text": instruction}]},
    ]
    kwargs2 = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages2,
        "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature, "topP": top_p,
                            "stopSequences": ["\n\nExplanation:", "```"]},
        "system": [{"text": strict_system}],
    }
    resp2 = bedrock.converse(**kwargs2)
    txt2 = _read_text_from_converse(resp2)
    try:
        val2 = json.loads(txt2)
        if isinstance(val2, list):
            return val2
        if isinstance(val2, dict):
            for v in val2.values():
                if isinstance(v, list):
                    return v
            return [val2]
    except Exception:
        pass
    arr2 = _extract_json_array(txt2)
    return json.loads(arr2)

def ask_llm_for_recommendations_as_list(df_summary: pd.DataFrame,
                                        allowed_pairs: List[Tuple[str, str]]) -> List[Dict]:
    """Call Nova Pro via Converse and return normalized list of rec dicts."""
    if df_summary is None or df_summary.empty:
        return []

    allow_lines = "\n".join(f"- {svc} @ {reg}" for svc, reg in allowed_pairs)
    summary_csv = df_summary.to_csv(index=False)
    system_text = (
        "You are a precise AWS FinOps assistant. "
        "Return ONLY a JSON array with no prose or code fences."
    )
    prompt_text = (
        "Given the following AWS cost breakdown (CSV), propose monthly savings opportunities.\n"
        "IMPORTANT:\n"
        f"Recommend ONLY for these service+region pairs:\n{allow_lines}\n"
        "Output MUST be a JSON array only with keys: "
        "category, subtype, recommendation, estimated_saving_usd, region.\n"
        f"\nCSV:\n{summary_csv}\n"
        "Return ONLY the JSON array."
    )

    try:
        items = _converse_json_array(prompt_text, system_text, max_tokens=900, temperature=0.1, top_p=0.9)
        out: List[Dict] = []
        for it in items:
            out.append({
                "category": it.get("category", "LLM Suggestion"),
                "subtype": it.get("subtype", ""),
                "region": it.get("region", ""),
                "assumption": it.get("assumption", "LLM-derived heuristic"),
                "metric": it.get("metric", ""),
                "est_monthly_saving_usd": float(it.get("estimated_saving_usd", 0.0)),
                "action_sql_hint": it.get("recommendation", ""),
                "source_note": "llm",
            })
        return out
    except Exception as e:
        print(f"[bedrock] converse failed or parse error: {e}")
        return []

# -------------------------------------------------
# 5.1) Column discovery helpers
from functools import lru_cache

_INSTANCE_PAIR_RE = re.compile(r'([a-z0-9.]+)\s*[\u2192>\-]+\s*([a-z0-9.]+)', re.IGNORECASE)

def _infer_source_instance_type(subtype: str) -> Optional[str]:
    if not subtype:
        return None
    m = _INSTANCE_PAIR_RE.search(subtype)
    if m:
        return m.group(1).strip().lower()
    tokens = re.findall(r'[a-z0-9]+\.[a-z0-9]+', subtype, flags=re.IGNORECASE)
    return tokens[0].lower() if tokens else None

@lru_cache(maxsize=1)
def _cur_columns_lower() -> set[str]:
    df = run_athena(f"SHOW COLUMNS IN {ATHENA_TABLE}")
    if df is None or df.empty:
        return set()
    colname = "column" if "column" in df.columns else df.columns[0]
    return set(str(x).strip().lower() for x in df[colname].tolist())

def _first_existing_col(*candidates: str) -> Optional[str]:
    cols = _cur_columns_lower()
    for c in candidates:
        if c.lower() in cols:
            return c
    return None

def _resource_id_col() -> Optional[str]:
    return _first_existing_col("rline_item_resource_id", "line_item_resource_id", "resource_id")

def _region_col() -> Optional[str]:
    return _first_existing_col("product_region", "line_item_availability_zone", "availability_zone", "region")

def _cost_col() -> Optional[str]:
    return _first_existing_col("line_item_blended_cost", "line_item_unblended_cost", "line_item_net_unblended_cost")

def _product_code_col() -> Optional[str]:
    return _first_existing_col("line_item_product_code", "product_code", "product_product_name")

def _instance_type_col() -> Optional[str]:
    return _first_existing_col("product_instance_type", "instance_type")

def _usage_type_col() -> Optional[str]:
    return _first_existing_col("line_item_usage_type", "usage_type")

def _item_desc_col() -> Optional[str]:
    return _first_existing_col("line_item_line_item_description", "line_item_description", "item_description", "usage_description")

def _safe_regex_literal(s: str) -> str:
    return s.replace(".", "\\.")

# -------------------------------------------------
# 5.2) Predicates (service / region / instance / date / regex)
def _build_region_pred(region: str) -> str:
    c = _region_col()
    if not region or not c:
        return ""
    if c.lower().endswith("availability_zone"):
        return f" AND LOWER({c}) LIKE LOWER('{region}') || '%'"
    return f" AND LOWER({c}) = LOWER('{region}')"

def _build_service_pred_any(service_aliases: List[str]) -> str:
    aliases = [a.lower() for a in service_aliases if a]
    if not aliases:
        return ""

    cols = []
    c_code = _product_code_col()
    if c_code:
        cols.append(c_code)
    for cand in ("product_product_name", "product_service", "product_servicecode", "product_product_family"):
        c = _first_existing_col(cand)
        if c and c not in cols:
            cols.append(c)
    if not cols:
        return ""

    col_predicates = []
    for c in cols:
        likes = [f"LOWER({c}) LIKE '%{a.replace('%','%%')}%'" for a in aliases]
        col_predicates.append("(" + " OR ".join(likes) + ")")

    return " AND (" + " OR ".join(col_predicates) + ")"

def _build_ec2_inst_pred(inst: str) -> str:
    parts = []
    c1 = _instance_type_col()
    c2 = _usage_type_col()
    c3 = _item_desc_col()
    if not any([c1, c2, c3]):
        return ""
    lit = _safe_regex_literal(inst)
    if c1:
        parts.append(f"LOWER({c1}) = '{inst.lower()}'")
    if c2:
        parts.append(f"REGEXP_LIKE(LOWER({c2}), '.*:{lit}($|[:/])')")
    if c3:
        parts.append(f"REGEXP_LIKE(LOWER({c3}), '(^|[^a-z0-9]){lit}([^a-z0-9]|$)')")
    return " AND (" + " OR ".join(parts) + ")"

def _build_regex_any_pred(regexes: List[str]) -> str:
    if not regexes:
        return ""
    cols = []
    c_ut = _usage_type_col()
    c_desc = _item_desc_col()
    if c_ut:
        cols.append(("LOWER(" + c_ut + ")", True))
    if c_desc:
        cols.append(("LOWER(" + c_desc + ")", True))
    if not cols:
        return ""
    all_ors = []
    for expr, _ in cols:
        or_one_col = " OR ".join([f"REGEXP_LIKE({expr}, '{rx}')" for rx in regexes])
        all_ors.append("(" + or_one_col + ")")
    return " AND (" + " OR ".join(all_ors) + ")"

# ---------- CUR latest-date anchor ----------
@lru_cache(maxsize=1)
def _cur_latest_date_str() -> Optional[str]:
    col = _first_existing_col(
        "line_item_usage_start_date",
        "usage_start_date",
        "bill_billing_period_start_date",
        "usage_start_time",
        "line_item_usage_start_time"
    )
    if not col:
        return None

    parsed_date = f"""
    COALESCE(
      TRY(CAST({col} AS DATE)),
      TRY(CAST(FROM_ISO8601_TIMESTAMP({col}) AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%Y-%m-%d %H:%i:%s') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%Y-%m-%d %H:%i') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%Y %H:%i:%s') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%Y %H:%i') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%Y %H:%i:%s') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%Y %H:%i') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%y %H:%i:%s') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%y %H:%i') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%y %H:%i:%s') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%y %H:%i') AS DATE))
    )
    """.strip()

    sql = f"SELECT DATE_FORMAT(MAX({parsed_date}), '%Y-%m-%d') AS d FROM {ATHENA_TABLE}"
    df = run_athena(sql)
    if df is None or df.empty or "d" not in df.columns:
        return None
    v = str(df["d"].iloc[0]) if df["d"].iloc[0] is not None else None
    v = v.strip() if v else None
    return v or None

# Robust, exception-safe date predicate using CUR anchor
def _date_pred(days: int) -> str:
    col = _first_existing_col(
        "line_item_usage_start_date",
        "usage_start_date",
        "bill_billing_period_start_date",
        "usage_start_time",
        "line_item_usage_start_time"
    )
    if not col:
        return ""
    parsed_ts = f"""
    COALESCE(
      TRY(CAST({col} AS TIMESTAMP)),
      TRY(FROM_ISO8601_TIMESTAMP({col})),
      TRY(DATE_PARSE({col}, '%Y-%m-%d %H:%i:%s')),
      TRY(DATE_PARSE({col}, '%Y-%m-%d %H:%i')),
      TRY(DATE_PARSE({col}, '%m/%d/%Y %H:%i:%s')),
      TRY(DATE_PARSE({col}, '%m/%d/%Y %H:%i')),
      TRY(DATE_PARSE({col}, '%d/%m/%Y %H:%i:%s')),
      TRY(DATE_PARSE({col}, '%d/%m/%Y %H:%i')),
      TRY(DATE_PARSE({col}, '%d/%m/%y %H:%i:%s')),
      TRY(DATE_PARSE({col}, '%d/%m/%y %H:%i')),
      TRY(DATE_PARSE({col}, '%m/%d/%y %H:%i:%s')),
      TRY(DATE_PARSE({col}, '%m/%d/%y %H:%i'))
    )
    """.strip()
    parsed_date_only = f"""
    COALESCE(
      TRY(CAST({col} AS DATE)),
      TRY(CAST(FROM_ISO8601_TIMESTAMP({col}) AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%Y-%m-%d') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%Y') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%Y') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%m/%d/%y') AS DATE)),
      TRY(CAST(DATE_PARSE({col}, '%d/%m/%y') AS DATE))
    )
    """.strip()
    parsed_date = f"COALESCE(CAST({parsed_ts} AS DATE), {parsed_date_only})"
    anchor = _cur_latest_date_str()
    anchor_expr = f"DATE '{anchor}'" if anchor else "CURRENT_DATE"
    return f" AND ({parsed_date}) >= DATE_ADD('day', -{int(days)}, {anchor_expr})"

# -------------------------------------------------
# 5.3) Generic resource-id lookups + service-specific wrappers
def _lookup_resource_ids_generic(
    service_aliases: List[str],
    region: Optional[str],
    days: int = 30,
    limit: int = 5,
    extra_regex_any: Optional[List[str]] = None,
    region_optional: bool = False,
) -> List[str]:
    rid = _resource_id_col()
    cost = _cost_col()
    if not rid or not cost:
        return []

    service_pred = _build_service_pred_any(service_aliases)
    date_pred    = _date_pred(days)
    regex_pred   = _build_regex_any_pred(extra_regex_any or [])
    notnull_pred = f" AND {rid} IS NOT NULL AND {rid} <> ''"

    preds_A = [service_pred, date_pred, regex_pred, notnull_pred]
    preds_A_with_region = None
    if region and not region_optional:
        preds_A.append(_build_region_pred(region))
    elif region and region_optional:
        preds_A_with_region = preds_A + [_build_region_pred(region)]

    preds_B = [service_pred, date_pred, regex_pred, notnull_pred]  # no region

    def _run(preds: List[str], tag: str) -> List[str]:
        sql = f"""
        SELECT {rid} AS r_id, SUM(COALESCE({cost}, 0)) AS cost_usd
        FROM {ATHENA_TABLE}
        WHERE 1=1
          {''.join(p for p in preds if p)}
        GROUP BY 1
        ORDER BY cost_usd DESC
        LIMIT {limit}
        """
        if os.getenv("DEBUG_ENRICH", "").lower() in ("1", "true", "yes"):
            print(f"[enrich][GENERIC:{tag}] SQL:\n{sql}")
        df = run_athena(sql)
        if df is None or df.empty or "r_id" not in df.columns:
            return []
        return [str(x) for x in df["r_id"].dropna().tolist() if str(x).strip()]

    # Normal attempts
    if preds_A_with_region is not None:
        ids = _run(preds_A_with_region, "A_with_region")
        if ids:
            return ids
        ids = _run(preds_B, "B_no_region")
        if ids:
            return ids
    else:
        ids = _run(preds_A, "A")
        if ids:
            return ids
        ids = _run(preds_B, "B_no_region")
        if ids:
            return ids

    # FINAL FALLBACK: ignore service/regex, keep date (and region if provided)
    fb_preds = [date_pred, notnull_pred]
    if region:
        fb_preds.append(_build_region_pred(region))
    ids = _run(fb_preds, "FALLBACK_no_service")
    return ids

def _lookup_resource_ids_for_ec2(region: str, inst: Optional[str], days: int = 30, limit: int = 5) -> List[str]:
    rid = _resource_id_col()
    cost = _cost_col()
    if not rid or not cost:
        return []
    service_pred = _build_service_pred_any(["AmazonEC2", "EC2", "Amazon Elastic Compute Cloud"])
    date_pred    = _date_pred(days)
    notnull_pred = f" AND {rid} IS NOT NULL AND {rid} <> ''"
    inst_pred    = _build_ec2_inst_pred(inst) if inst else ""
    preds_A = [service_pred, _build_region_pred(region), inst_pred, date_pred, notnull_pred]
    preds_B = [service_pred, _build_region_pred(region), date_pred, notnull_pred]
    preds_C = [service_pred, date_pred, notnull_pred]

    def _run(preds: List[str]) -> List[str]:
        sql = f"""
        SELECT {rid} AS r_id, SUM(COALESCE({cost}, 0)) AS cost_usd
        FROM {ATHENA_TABLE}
        WHERE 1=1
          {''.join(p for p in preds if p)}
        GROUP BY 1
        ORDER BY cost_usd DESC
        LIMIT {limit}
        """
        if os.getenv("DEBUG_ENRICH", "").lower() in ("1", "true", "yes"):
            print("[enrich][EC2] SQL:\n", sql)
        df = run_athena(sql)
        if df is None or df.empty or "r_id" not in df.columns:
            return []
        return [str(x) for x in df["r_id"].dropna().tolist() if str(x).strip()]

    for preds in (preds_A, preds_B, preds_C):
        ids = _run(preds)
        if ids:
            return ids
    return []

def _lookup_resource_ids_for_s3(region: str, days: int = 30, limit: int = 5) -> List[str]:
    return _lookup_resource_ids_generic(
        service_aliases=["AmazonS3", "S3", "Amazon Simple Storage Service"],
        region=region,
        days=days,
        limit=limit,
        extra_regex_any=[],
    )

# -------------------------------------------------
# 5.4) Enrichment entrypoint (covers all major categories)
def enrich_recommendations_with_resource_ids(recs: List[Dict], days: int = 30) -> List[Dict]:
    """
    Populate rline_item_resource_id for all major services:
      - EC2 Right-size (instance-aware)
      - S3 Storage Optimization
      - EBS Optimization (volumes)
      - Snapshot Hygiene (EBS snapshots)
      - CloudFront Optimization (global)
      - Lambda Optimization
      - Generic LLM categories like 'AmazonS3 Optimization', etc.
    """
    PROFILES = {
        "s3": {
            "aliases": ["AmazonS3", "S3", "Amazon Simple Storage Service"],
            "regex":   [],
            "region_optional": False,
        },
        "ebs_vol": {
            "aliases": ["AmazonEBS", "EBS", "Amazon Elastic Block Store"],
            "regex":   [r"volume", r"ebs.*volume", r"vol-"],
            "region_optional": False,
        },
        "ebs_snap": {
            "aliases": ["AmazonEBS", "EBS", "Amazon Elastic Block Store"],
            "regex":   [r"snapshot", r"snap-"],
            "region_optional": False,
        },
        "cloudfront": {
            "aliases": ["AmazonCloudFront", "CloudFront"],
            "regex":   [],
            "region_optional": True,  # global service
        },
        "lambda": {
            "aliases": ["AWSLambda", "Lambda"],
            "regex":   [r"lambda", r"request", r"gb-second", r"duration"],
            "region_optional": False,
        },
        "ec2_generic": {
            "aliases": ["AmazonEC2", "EC2", "Amazon Elastic Compute Cloud"],
            "regex":   [r"boxusage", r"runinstances", r"on demand"],
            "region_optional": False,
        },
    }

    def _match_profile(cat_lc: str) -> Optional[str]:
        if cat_lc.startswith("s3"):
            return "s3"
        if "snapshot" in cat_lc:
            return "ebs_snap"
        if "ebs" in cat_lc:
            return "ebs_vol"
        if "cloudfront" in cat_lc:
            return "cloudfront"
        if "lambda" in cat_lc:
            return "lambda"
        if "ec2" in cat_lc:
            return "ec2_generic"
        # LLM-named categories like 'AmazonS3 Optimization'
        if "amazon s3" in cat_lc:
            return "s3"
        if "amazoncloudfront" in cat_lc or "cloudfront" in cat_lc:
            return "cloudfront"
        if "amazonebs" in cat_lc or "elastic block store" in cat_lc:
            return "ebs_vol"
        if "aws lambda" in cat_lc or "lambda" in cat_lc:
            return "lambda"
        if "amazon ec2" in cat_lc or "elastic compute cloud" in cat_lc:
            return "ec2_generic"
        return None

    out: List[Dict] = []
    for r in recs:
        r2 = dict(r)
        cat_lc = (r.get("category") or "").strip().lower()
        region = (r.get("region") or "").strip()
        ids: List[str] = []

        if cat_lc.startswith("ec2 right-size") or (cat_lc.startswith("ec2") and "right" in cat_lc):
            src_inst = _infer_source_instance_type(r.get("subtype") or "")
            ids = _lookup_resource_ids_for_ec2(region, src_inst, days=days) if src_inst else \
                  _lookup_resource_ids_generic(PROFILES["ec2_generic"]["aliases"], region, days, 5,
                                               PROFILES["ec2_generic"]["regex"], False)
        elif cat_lc.startswith("s3"):
            ids = _lookup_resource_ids_for_s3(region, days=days)
        else:
            key = _match_profile(cat_lc)
            if key:
                p = PROFILES[key]
                ids = _lookup_resource_ids_generic(
                    service_aliases=p["aliases"],
                    region=region,
                    days=days,
                    limit=5,
                    extra_regex_any=p["regex"],
                    region_optional=p["region_optional"],
                )
            else:
                # last-resort broad match
                ids = _lookup_resource_ids_generic(
                    service_aliases=["AmazonEC2","EC2","Amazon Elastic Compute Cloud",
                                     "AmazonS3","S3","Amazon Simple Storage Service",
                                     "AmazonEBS","EBS","Amazon Elastic Block Store",
                                     "AWSLambda","Lambda",
                                     "AmazonCloudFront","CloudFront"],
                    region=region,
                    days=days,
                    limit=5,
                    extra_regex_any=[],
                    region_optional=True,
                )

        r2["rline_item_resource_id"] = ",".join(ids) if ids else ""
        out.append(r2)
    return out

# -------------------------------------------------
# 5.5) S3 summary helpers
def _s3_key_join(*parts: str) -> str:
    return "/".join([p.strip("/") for p in parts if p])

def write_summary_json_to_s3(summary_obj: dict, run_id: str) -> str:
    key = _s3_key_join(RESULTS_PREFIX, "runs", run_id, "summary.json")
    body = json.dumps(summary_obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    s3.put_object(
        Bucket=RESULTS_BUCKET,
        Key=key,
        Body=body,
        ContentType="application/json",
        CacheControl="no-store, no-cache, must-revalidate"
    )
    return f"s3://{RESULTS_BUCKET}/{key}"

# -------------------------------------------------
# 6) Main handler (per-invocation run_id)
def handler(event=None, context=None):
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    now_iso = datetime.now(timezone.utc).isoformat()

    print(f"[cfg] DB={ATHENA_DB} TABLE={ATHENA_TABLE} OUT={ATHENA_OUTPUT} RECS_TABLE={ATHENA_RECS_TABLE} RUN_ID={run_id}")

    hourly = hourly_usage_breakdown()
    print(f"[athena] hourly rows: {len(hourly)}")

    # Heuristic recommendations
    recs: List[Dict] = []
    recs += recommend_ec2_rightsize(hourly)
    recs += recommend_s3_tiering(hourly)
    recs += recommend_snapshot_hygiene(hourly)

    # LLM augmentation (optional)
    svc_reg_cost: Dict[Tuple[str, str], float] = {}
    allowed_pairs: List[Tuple[str, str]] = []
    if USE_LLM:
        topn = query_cost_summary_topn(20)
        print(f"[athena] topN rows for LLM: {len(topn)}")
        for _, row in topn.iterrows():
            svc = str(row["service"]); reg = str(row["region"]); cost = float(row["cost_usd"])
            svc_reg_cost[(svc, reg)] = svc_reg_cost.get((svc, reg), 0.0) + cost
            allowed_pairs.append((svc, reg))
        try:
            llm_recs = ask_llm_for_recommendations_as_list(topn, allowed_pairs)
            recs.extend(llm_recs)
        except Exception as e:
            print(f"[llm] skipping due to error: {e}")

    # ---------- QC & capping for LLM ----------
    from collections import defaultdict
    ABS_CAP = LLM_ABS_CAP
    REL_CAP = LLM_REL_CAP_PCT / 100.0

    def _observed_spend_for_rec(r: dict) -> float:
        cat = r.get("category", ""); reg = r.get("region", "")
        services = CATEGORY_TO_SERVICE.get(cat, [])
        return sum(svc_reg_cost.get((svc, reg), 0.0) for svc in services)

    for r in recs:
        if r.get("source_note") == "llm":
            v = float(r.get("est_monthly_saving_usd", 0.0))
            if v > ABS_CAP:
                r["assumption"] = (r.get("assumption", "") + " | capped_abs").strip(" |")
                v = ABS_CAP
            base = _observed_spend_for_rec(r)
            if base > 0:
                limit = REL_CAP * base
                if v > limit:
                    r["assumption"] = (r.get("assumption", "") + f" | capped_to_{int(REL_CAP*100)}pct").strip(" |")
                    v = limit
            r["est_monthly_saving_usd"] = round(v, 2)

    # Deduplicate by (category, subtype, region)
    key_fn = lambda r: (r.get("category",""), r.get("subtype",""), r.get("region",""))
    dedup: Dict[Tuple[str,str,str], Dict] = {}
    for r in recs:
        k = key_fn(r)
        if k not in dedup or float(r.get("est_monthly_saving_usd",0)) > float(dedup[k].get("est_monthly_saving_usd",0)):
            dedup[k] = r
    recs = list(dedup.values())

    # Enrich with resource IDs from raw BEFORE stamping/IO (use dataset-aware lookback)
    recs = enrich_recommendations_with_resource_ids(recs, days=LOOKBACK_DAYS)

    # QC summaries
    by_src = defaultdict(float)
    by_cat = defaultdict(float)
    for r in recs:
        by_src[r.get("source_note","?")] += float(r.get("est_monthly_saving_usd", 0.0))
        by_cat[r.get("category","?")]    += float(r.get("est_monthly_saving_usd", 0.0))
    print("[QC] savings by source:", {k: round(v,2) for k,v in by_src.items()})
    print("[QC] savings by category:", {k: round(v,2) for k,v in by_cat.items()})

    one_time_total = sum(float(r.get("one_time_saving_usd", 0.0)) for r in recs)
    if one_time_total:
        print(f"[QC] one-time savings (not counted in monthly total): ${round(one_time_total,2)}")

    # Stamp run_id + created_at
    for r in recs:
        r["run_id"] = run_id
        r["created_at"] = now_iso

    # Write detailed rows (CSV) & ensure Athena table
    s3_prefix_uri = write_recommendations_csv_to_s3(recs, run_id=run_id)
    build_recommendations_table_if_needed(s3_prefix_uri, table_name=ATHENA_RECS_TABLE)

    total = round(sum(float(r.get("est_monthly_saving_usd", 0.0)) for r in recs), 2)

    # summary.json now includes ALL items
    summary_obj = {
        "run_id": run_id,
        "created_at": now_iso,
        "use_llm": USE_LLM,
        "counts": {
            "total_recommendations": len(recs),
            "by_category": {k: int(sum(1 for r in recs if r.get("category","")==k)) for k in by_cat.keys()},
            "by_source":   {k: int(sum(1 for r in recs if r.get("source_note","")==k)) for k in by_src.keys()},
        },
        "savings_usd": {
            "monthly_total": total,
            "by_category": {k: round(v, 2) for k, v in by_cat.items()},
            "by_source":   {k: round(v, 2) for k, v in by_src.items()},
        },
        "storage": {
            "athena_recs_table": ATHENA_RECS_TABLE,
            "rows_csv_prefix": s3_prefix_uri,
        },
        "items": recs,        # ALL items for UI
        "preview": recs[:5],  # convenience
    }

    summary_s3_uri = write_summary_json_to_s3(summary_obj, run_id=run_id)

    print(f"[done] wrote {len(recs)} recommendations, est_total=${total} to {s3_prefix_uri}")
    print(f"[done] summary.json -> {summary_s3_uri}")

    return {
        "count": len(recs),
        "est_total_monthly_saving_usd": total,
        "athena_recs_table": ATHENA_RECS_TABLE,
        "s3_prefix": s3_prefix_uri,
        "summary_s3_uri": summary_s3_uri,
        "use_llm": USE_LLM,
        "preview": recs[:3],
        "run_id": run_id,
    }

# -------------------------------------------------
# 8) Local run
if __name__ == "__main__":
    out = handler()
    print(json.dumps(out, indent=2))
