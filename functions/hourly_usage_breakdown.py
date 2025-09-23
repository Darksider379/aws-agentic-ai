# functions/hourly_usage_breakdown.py
import os
import pandas as pd
from .run_athena import run_athena

ATHENA_TABLE = os.environ["ATHENA_TABLE"]


def hourly_usage_breakdown() -> pd.DataFrame:
    """
    Returns hourly (or whatever your table granularity is) usage & cost rollup by
    service/usage_type/region. The function is schema-safe for your synthetic CUR.
    """
    sql = f"""
    SELECT line_item_usage_start_date AS ts,
           line_item_product_code     AS service,
           line_item_usage_type       AS usage_type,
           product_region             AS region,
           SUM(line_item_usage_amount) AS usage_amount,
           SUM(line_item_blended_cost) AS cost_usd
    FROM {ATHENA_TABLE}
    GROUP BY 1,2,3,4
    """
    df = run_athena(sql)
    if df.empty:
        return df
    df["usage_amount"] = pd.to_numeric(df["usage_amount"], errors="coerce").fillna(0.0)
    df["cost_usd"]     = pd.to_numeric(df["cost_usd"],     errors="coerce").fillna(0.0)
    return df
