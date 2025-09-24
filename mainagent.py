# mainagent.py
# -------------------------------------------------
# 0) Load env before imports that read os.environ
import os, sys
from pathlib import Path

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
import pandas as pd
import boto3

# Required env
ATHENA_DB         = os.environ["ATHENA_DB"]
ATHENA_TABLE      = os.environ["ATHENA_TABLE"]         # e.g., raw or raw_cast
ATHENA_WORKGROUP  = os.environ.get("ATHENA_WORKGROUP", "primary")
ATHENA_OUTPUT     = os.environ["ATHENA_OUTPUT"]        # s3://.../athena/
RESULTS_BUCKET    = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX    = os.environ.get("RESULTS_PREFIX", "cost-agent")
ATHENA_RECS_TABLE = os.environ.get("ATHENA_RECS_TABLE", "recommendations")

# Optional Bedrock/LLM
USE_LLM          = os.environ.get("USE_LLM", "false").lower() == "true"
BEDROCK_REGION   = os.environ.get("BEDROCK_REGION", "us-east-1")
# Default to Nova Pro; overrideable in config.ini
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "amazon.nova-pro-v1:0")

# AWS clients
athena  = boto3.client("athena", region_name=os.environ.get("AWS_REGION", None))
bedrock = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION)

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
# 3) Small summary query for LLM (top N contributors)
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
    if df.empty:
        return df
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    return df

# -------------------------------------------------
# 4) Bedrock (Nova Pro) helpers via CONVERSE API

def _extract_json_array(text: str) -> str:
    """Extract a JSON array from arbitrary text."""
    if not text:
        raise ValueError("Empty model output")

    # ```json ... ```
    fence = re.search(r"```json\s*(.+?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()

    # First [...] span with balanced brackets
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

    # Last resort: newline-delimited JSON objects -> array
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

def _converse_text(
    user_text: str,
    system_text: str | None = None,
    max_tokens: int = 600,
    temperature: float = 0.2,
    top_p: float = 0.9,
) -> str:
    """
    Single-turn text response using Bedrock Converse.
    """
    messages = [{"role": "user", "content": [{"text": user_text}]}]
    kwargs = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
    }
    if system_text:
        kwargs["system"] = [{"text": system_text}]

    resp = bedrock.converse(**kwargs)
    out_msg = resp.get("output", {}).get("message", {})
    content = out_msg.get("content", [])
    for block in content:
        if "text" in block and isinstance(block["text"], str):
            return block["text"].strip()
    return ""

def _converse_json_array(
    instruction: str,
    system_text: str | None = None,
    max_tokens: int = 900,
    temperature: float = 0.1,
    top_p: float = 0.9,
) -> list:
    """
    Ask Nova Pro for a STRICT JSON array.
    Strategy:
      1) Try JSON responseFormat.
      2) If not pure JSON array, try to extract first balanced array.
      3) Retry without responseFormat but with ultra-strict prompt + few-shot.
    """
    def _read_text_from_converse(resp: dict) -> str:
        out_msg = resp.get("output", {}).get("message", {})
        txt = ""
        for block in out_msg.get("content", []):
            if "text" in block:
                txt += block["text"]
        return txt.strip()

    # ---------- Attempt 1: JSON responseFormat ----------
    messages = [{"role": "user", "content": [{"text": instruction}]}]
    kwargs = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
        },
        "responseFormat": {"type": "JSON"},  # honored on newer Nova runtimes
    }
    if system_text:
        kwargs["system"] = [{"text": system_text}]

    try:
        resp = bedrock.converse(**kwargs)
        txt = _read_text_from_converse(resp)

        # Try direct parse
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

        # Try extracting first [...] span
        arr = _extract_json_array(txt)
        return json.loads(arr)
    except Exception as e:
        last_err = e  # noqa: F841

    # ---------- Attempt 2: No JSON mode, stricter prompt + few-shot ----------
    strict_system = (
        (system_text + " ") if system_text else ""
    ) + "Return ONLY a JSON array. Do not add any prose, backticks, or explanations."

    few_shot = (
        "You must respond with a JSON array only.\n"
        "Valid example:\n"
        '[{"category":"EC2 Right-size","subtype":"m5.large→m7g.large",'
        '"recommendation":"Move to Graviton m7g.large for CPU-bound workloads",'
        '"estimated_saving_usd":123.45,"region":"us-east-1"}]'
    )

    messages2 = [
        {"role": "user", "content": [{"text": few_shot}]},
        {"role": "assistant", "content": [{"text":
            '[{"category":"EC2 Right-size","subtype":"m5.large→m7g.large","recommendation":"...",'
            '"estimated_saving_usd":100.0,"region":"us-east-1"}]'}]},
        {"role": "user", "content": [{"text": instruction}]},
    ]
    kwargs2 = {
        "modelId": BEDROCK_MODEL_ID,
        "messages": messages2,
        "inferenceConfig": {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": top_p,
            "stopSequences": ["\n\nExplanation:", "```"],
        },
        "system": [{"text": strict_system}],
    }

    resp2 = bedrock.converse(**kwargs2)
    txt2 = _read_text_from_converse(resp2)

    if not txt2.strip().startswith("["):
        print("[llm][debug] Non-array start; raw (first 1200 chars):\n", txt2[:1200])

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

# -------------------------------------------------
# 5) Bedrock LLM call (returns list[dict] with same schema as heuristics)
def ask_llm_for_recommendations_as_list(df_summary: pd.DataFrame) -> list[dict]:
    """Call Bedrock LLM (Nova Pro by default via Converse) and return list of rec dicts."""
    if df_summary.empty:
        return []

    summary_csv = df_summary.to_csv(index=False)

    system_text = (
        "You are a precise AWS FinOps assistant. "
        "Return ONLY a JSON array with no prose or code fences."
    )

    prompt_text = (
        "Given the following AWS cost breakdown (CSV), propose savings opportunities.\n"
        "Output MUST be a JSON array ONLY. Do not include backticks, explanations, or extra fields.\n"
        "Each array item MUST have exactly these keys: "
        "category, subtype, recommendation, estimated_saving_usd, region.\n"
        'Example: [{"category":"EC2 Right-size","subtype":"m5.large→m7g.large",'
        '"recommendation":"Move m5.large to m7g.large where supported",'
        '"estimated_saving_usd":120.0,"region":"us-east-1"}]\n'
        "\nCSV:\n"
        f"{summary_csv}\n"
        "\nReturn ONLY the JSON array."
    )

    try:
        items = _converse_json_array(
            instruction=prompt_text,
            system_text=system_text,
            max_tokens=900,
            temperature=0.1,
            top_p=0.9,
        )

        out: list[dict] = []
        for it in items:
            out.append(
                {
                    "category": it.get("category", "LLM Suggestion"),
                    "subtype": it.get("subtype", ""),
                    "region": it.get("region", ""),
                    "assumption": it.get("assumption", "LLM-derived heuristic"),
                    "metric": it.get("metric", ""),
                    "est_monthly_saving_usd": float(it.get("estimated_saving_usd", 0.0)),
                    "action_sql_hint": it.get("recommendation", ""),
                    "source_note": "llm",
                }
            )
        return out

    except Exception as e:
        print(f"[bedrock] converse failed or parse error: {e}")
        # Fallback so pipeline continues
        return [
            {
                "category": "S3 Tiering",
                "subtype": "Standard→IA (10-30%)",
                "region": df_summary.iloc[0]["region"] if "region" in df_summary.columns and not df_summary.empty else "",
                "assumption": "LLM fallback; review access patterns",
                "metric": "top service/region cold prefixes",
                "est_monthly_saving_usd": 25.0,
                "action_sql_hint": "Enable lifecycle to IA for low-GET prefixes; verify retrievals.",
                "source_note": "llm-fallback",
            }
        ]

# -------------------------------------------------
# 6) Main handler
def handler(event=None, context=None):
    print(f"[cfg] DB={ATHENA_DB} TABLE={ATHENA_TABLE} OUT={ATHENA_OUTPUT} RECS_TABLE={ATHENA_RECS_TABLE}")
    # Pull hourly breakdown (for heuristics)
    hourly = hourly_usage_breakdown()
    print(f"[athena] hourly rows: {len(hourly)}")

    # Heuristic recommendations
    recs: list[dict] = []
    recs += recommend_ec2_rightsize(hourly)
    recs += recommend_s3_tiering(hourly)
    recs += recommend_snapshot_hygiene(hourly)

    # Optional LLM augmentation (top-N summary to LLM)
    if USE_LLM:
        topn = query_cost_summary_topn(20)
        print(f"[athena] topN rows for LLM: {len(topn)}")
        try:
            llm_recs = ask_llm_for_recommendations_as_list(topn)
            recs.extend(llm_recs)
        except Exception as e:
            print(f"[llm] skipping due to error: {e}")

    # Write to S3 and ensure Athena external table exists
    s3_prefix_uri = write_recommendations_csv_to_s3(recs)
    build_recommendations_table_if_needed(s3_prefix_uri, table_name=ATHENA_RECS_TABLE)

    total = sum(float(r.get("est_monthly_saving_usd", 0.0)) for r in recs)
    print(f"[done] wrote {len(recs)} recommendations, est_total=${round(total,2)} to {s3_prefix_uri}")

    return {
        "count": len(recs),
        "est_total_monthly_saving_usd": round(total, 2),
        "athena_recs_table": ATHENA_RECS_TABLE,
        "s3_prefix": s3_prefix_uri,
        "use_llm": USE_LLM,
        "preview": recs[:3],
    }

# -------------------------------------------------
# 7) Local run
if __name__ == "__main__":
    out = handler()
    print(json.dumps(out, indent=2))
