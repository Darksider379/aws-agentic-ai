class recommend_snapshot_hygiene():
    def recommend_snapshot_hygiene(df_hourly: pd.DataFrame) -> list:
        """
        Heuristic: snapshot usage seen (EBS:SnapshotUsage) -> suggest lifecycle/retention tightening.
        """
        ebs = df_hourly[(df_hourly["service"]=="AmazonEBS") & (df_hourly["usage_type"]=="EBS:SnapshotUsage")]
        if ebs.empty: return []
        by_region = ebs.groupby("region", as_index=False)["cost_usd"].sum()
        recs = []
        for _, r in by_region.iterrows():
            cost = float(r["cost_usd"])
            if cost >= 10:
                recs.append({
                    "category": "EBS Snapshots",
                    "subtype": "Retention policy",
                    "region": r["region"],
                    "assumption": "Can reduce redundant snapshots by ~20%",
                    "metric": f"Monthly snapshot cost:${round(cost,2)}",
                    "est_monthly_saving_usd": round(cost * 0.20, 2),
                    "action_sql_hint": "Enable lifecycle; dedupe old snapshots; evaluate fast-snapshot restore usage."
                })
        return recs
