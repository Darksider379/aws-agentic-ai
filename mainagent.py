####config imports####
import os, sys
from pathlib import Path

def load_env_file(path: str = "config.ini"):
    try:
        from dotenv import load_dotenv
        if Path(path).exists():
            load_dotenv(dotenv_path=path, override=True)
            return
    except Exception:
        pass

    if Path(path).exists():
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

env_path = "config.ini"
load_env_file(env_path)

########main code########
import boto3, json
import pandas as pd

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
athena  = boto3.client("athena")

def query_cost_summary():
    sql = f"""
    SELECT line_item_product_code AS service,
           product_region AS region,
           SUM(line_item_blended_cost) AS cost_usd
    FROM {os.environ['ATHENA_TABLE']}
    GROUP BY 1,2
    ORDER BY cost_usd DESC
    LIMIT 20
    """
    # (reuse the run_athena function from before)
    return run_athena(sql)

def ask_llm_for_recommendations(df: pd.DataFrame) -> str:
    summary = df.to_csv(index=False)
    prompt = f"""
    You are a cloud cost optimization assistant.
    Here is a recent AWS cost breakdown:

    {summary}

    Please identify patterns where money can be saved.
    Suggest concrete actions (like right-sizing, S3 tiering, etc.)
    and give a rough monthly savings estimate in USD if possible.
    Format your answer as JSON with fields:
    category, recommendation, estimated_saving_usd, region.
    """

    resp = bedrock.invoke_model(
        modelId="anthropic.claude-v2",   # or another Bedrock LLM
        body=json.dumps({"prompt": prompt, "max_tokens_to_sample": 500}),
        contentType="application/json",
        accept="application/json"
    )
    out = json.loads(resp["body"].read())
    return out["completion"]  # model’s text

def handler(event=None, context=None):
    df = query_cost_summary()
    llm_json = ask_llm_for_recommendations(df)
    print("LLM recommendations:", llm_json)
    # TODO: parse JSON and write to S3/Athena table
