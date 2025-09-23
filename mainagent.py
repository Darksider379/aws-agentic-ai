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
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620")

# AWS clients
athena  = boto3.client("athena")
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
# 4) Bedrock LLM call (returns list[dict] with same schema as heuristics)
def ask_llm_for_recommendations_as_list(df_summary: pd.DataFrame) -> list[dict]:
    if df_summary.empty:
        return []

    # Prepare minimal CSV for the prompt
    summary_csv = df_summary.to_csv(index=False)

    system = (
        "You are a cloud cost optimization assistant. "
        "Given the following AWS cost breakdown (CSV), propose savings opportunities. "
        "Return STRICT JSON array of objects with keys: "
        "category, subtype, recommendation, estimated_saving_usd, region."
    )
    user = f"CSV:\n{summary_csv}\n"

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 600,
        "system": system,
        "messages": [{"role": "user", "content": [{"type": "text", "text": user}]}],
    }

    try:
        resp = bedrock.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(resp["body"].read())

        text = ""
        for block in payload.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")

        # Try to parse JSON; if it isn't valid JSON, fallback gracefully
        llm_items = json.loads(text)
        out: list[dict] = []
        for it in llm_items:
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
        print(f"[bedrock] invoke failed or parse error: {e}")
        # Fallback suggestion so the pipeline still writes something
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
# 5) Main handler
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
# 6) Local run
if __name__ == "__main__":
    out = handler()
    print(json.dumps(out, indent=2))
