import pandas as pd
import boto3
import time
import os


class recommend_ec2_rightsize():
    def recommend_ec2_rightsize(df_hourly: pd.DataFrame) -> list:
        """
        Heuristic: where EC2 m5.large hours dominate, recommend switching to m7g.large
        Savings = hours * (m5 - m7g); assumes ARM compatibility and similar perf.
        """
        ec2 = df_hourly[(df_hourly["service"]=="AmazonEC2") & df_hourly["usage_type"].str.startswith("BoxUsage")]
        if ec2.empty: return []
        # Sum hours by flavor and region
        ec2["hours"] = ec2["usage_amount"]  # usage is hours
        by_type = ec2.groupby(["usage_type","region"], as_index=False)["hours"].sum()
        m5 = by_type[by_type["usage_type"].str.contains("m5.large")]
        recs = []
        for _, r in m5.iterrows():
            hours = float(r["hours"])
            region = r["region"]
            potential = max(0.0, (M5_LARGE - M7G_LARGE) * hours)
            if hours >= 100 and potential >= 5:  # simple thresholds to filter noise
                recs.append({
                    "category": "EC2 Right-size",
                    "subtype": "m5.large→m7g.large",
                    "region": region,
                    "assumption": "ARM compatible workload",
                    "metric": f"Hours:{int(hours)}",
                    "est_monthly_saving_usd": round(potential, 2),
                    "action_sql_hint": "Identify ASGs/LaunchTemplates using m5.large in this region; test Graviton."
                })
        return recs