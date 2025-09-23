# recommendations/recommend_ec2_rightsize.py
import os
import pandas as pd

M5_LARGE = float(os.environ.get("EC2_M5_LARGE", "0.096"))
M7G_LARGE = float(os.environ.get("EC2_M7G_LARGE", "0.084"))


def recommend_ec2_rightsize(df_hourly: pd.DataFrame) -> list[dict]:
    """
    Heuristic: where EC2 m5.large hours dominate, recommend switching to m7g.large.
    Savings = hours * (m5 - m7g); assumes ARM compatibility.
    """
    if df_hourly.empty:
        return []
    ec2 = df_hourly[
        (df_hourly["service"] == "AmazonEC2")
        & df_hourly["usage_type"].str.startswith("BoxUsage")
    ].copy()
    if ec2.empty:
        return []

    ec2["hours"] = ec2["usage_amount"]
    by_type = ec2.groupby(["usage_type", "region"], as_index=False)["hours"].sum()
    m5 = by_type[by_type["usage_type"].str.contains("m5.large")]
    recs: list[dict] = []

    for _, r in m5.iterrows():
        hours = float(r["hours"])
        region = r["region"]
        potential = max(0.0, (M5_LARGE - M7G_LARGE) * hours)
        if hours >= 100 and potential >= 5:  # de-noise thresholds
            recs.append(
                {
                    "category": "EC2 Right-size",
                    "subtype": "m5.large→m7g.large",
                    "region": region,
                    "assumption": "ARM compatible workload",
                    "metric": f"Hours:{int(hours)}",
                    "est_monthly_saving_usd": round(potential, 2),
                    "action_sql_hint": "Identify ASGs/LaunchTemplates using m5.large; test Graviton in this region.",
                    "source_note": "heuristics-v1",
                }
            )
    return recs
