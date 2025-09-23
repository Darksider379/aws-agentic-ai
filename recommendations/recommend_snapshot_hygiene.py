# recommendations/recommend_snapshot_hygiene.py
import pandas as pd


def recommend_snapshot_hygiene(df_hourly: pd.DataFrame) -> list[dict]:
    """
    Heuristic: If EBS Snapshot cost is material, suggest cutting ~20% via retention/dedupe.
    """
    if df_hourly.empty:
        return []

    ebs = df_hourly[
        (df_hourly["service"] == "AmazonEBS")
        & (df_hourly["usage_type"] == "EBS:SnapshotUsage")
    ].copy()
    if ebs.empty:
        return []

    by_region = ebs.groupby("region", as_index=False)["cost_usd"].sum()

    recs: list[dict] = []
    for _, r in by_region.iterrows():
        monthly_cost = float(r["cost_usd"])
        if monthly_cost >= 10:  # noise filter
            recs.append(
                {
                    "category": "EBS Snapshots",
                    "subtype": "Retention policy",
                    "region": r["region"],
                    "assumption": "Reduce redundant/old snapshots by ~20%",
                    "metric": f"Monthly snapshot cost:${round(monthly_cost,2)}",
                    "est_monthly_saving_usd": round(monthly_cost * 0.20, 2),
                    "action_sql_hint": "Apply lifecycle retention; prune orphaned snapshots; review fast-snapshot-restore usage.",
                    "source_note": "heuristics-v1",
                }
            )
    return recs
