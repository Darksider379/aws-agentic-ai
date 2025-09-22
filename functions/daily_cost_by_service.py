import pandas as pd
import boto3
import time
import os

class daily_cost_by_service():
    def daily_cost_by_service():
        sql = f"""
        SELECT date_trunc('day', line_item_usage_start_date) AS day,
            line_item_product_code AS service,
            product_region AS region,
            COALESCE(resource_tags_user_env,'') AS env,
            COALESCE(resource_tags_user_app,'') AS app,
            SUM(line_item_blended_cost) AS cost_usd
        FROM {ATHENA_TABLE}
        GROUP BY 1,2,3,4,5
        """
        df = run_athena(sql)
        if not df.empty:
            df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)
        return df