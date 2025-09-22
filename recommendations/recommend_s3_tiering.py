class recommend_s3_tiering():
    def recommend_s3_tiering(df_hourly: pd.DataFrame) -> list:
        """
        Heuristic: if Requests/GB is low, move portion to IA.
        Convert TimedStorage-ByteHrs to GB-Month approximate; assume 30% can go to IA.
        """
        s3 = df_hourly[df_hourly["service"]=="AmazonS3"]
        if s3.empty: return []
        # Aggregate monthly storage and requests by region
        s3["month"] = pd.to_datetime(s3["ts"]).dt.to_period("M")
        storage = s3[s3["usage_type"]=="TimedStorage-ByteHrs"].groupby(["region","month"], as_index=False)["usage_amount"].sum()
        reqs    = s3[s3["usage_type"]=="Requests-Tier1"].groupby(["region","month"], as_index=False)["usage_amount"].sum()
        # Convert ByteHrs -> GB-Month: GB = bytes / 1024^3; Month ≈ ByteHrs / (1024^3 * 720)
        storage["gb_month"] = storage["usage_amount"].astype(float) / (1024**3 * 720.0)
        merged = pd.merge(storage, reqs, on=["region","month"], how="left", suffixes=("","_req"))
        merged["usage_amount_req"] = pd.to_numeric(merged["usage_amount_req"], errors="coerce").fillna(0.0)
        recs = []
        for _, r in merged.groupby("region").tail(1).iterrows():  # latest month per region
            gbm = float(r["gb_month"])
            req = float(r["usage_amount_req"])
            if gbm <= 0: continue
            requests_per_gb = req / (gbm * 1_000_000_000 / (1024**3))  # rough: requests per GB-month (units differ; good enough for heuristic)
            # If requests/GB is low, cold data likely → tier 30% to IA
            if requests_per_gb < 1.0:  # tune
                move_gb = gbm * 0.30
                base = move_gb * S3_STD
                tgt  = move_gb * S3_IA + (move_gb * S3_RETR) * (S3_STD - S3_IA)  # retrieval penalty very rough
                saving = max(0.0, base - tgt)
                if saving >= 5:
                    recs.append({
                        "category": "S3 Tiering",
                        "subtype": "Standard→IA (30%)",
                        "region": r["region"],
                        "assumption": f"~{int(S3_RETR*100)}% monthly retrieval",
                        "metric": f"GB-Month move:{int(move_gb)}",
                        "est_monthly_saving_usd": round(saving, 2),
                        "action_sql_hint": "Target buckets with low GET rate; enable lifecycle to IA."
                    })
        return recs