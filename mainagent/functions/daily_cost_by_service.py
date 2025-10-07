# functions/daily_cost_by_service.py
import os
import pandas as pd
from .run_athena import run_athena

ATHENA_TABLE = os.environ["ATHENA_TABLE"]


def daily_cost_by_service() -> pd.DataFrame:
    """
    Daily cost rollup by service/region/env/app for anomaly/forecast summaries.
    """
    sql = f"""
    SELECT date_trunc('day', line_item_usage_start_date) AS day,
           line_item_product_code AS service,
           product_region         AS region,
           COALESCE(resource_tags_user_env,'') AS env,
           COALESCE(resource_tags_user_app,'') AS app,
           SUM(line_item_blended_cost) AS cost_usd
    FROM {ATHENA_TABLE}
    GROUP BY 1,2,3,4,5
    ORDER BY 1,2,3
    """
    df = run_athena(sql)
    if df.empty:
        return df
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
    return df
