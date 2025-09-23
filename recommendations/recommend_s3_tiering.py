# recommendations/recommend_s3_tiering.py
import os
import pandas as pd

S3_STD  = float(os.environ.get("S3_STANDARD_PER_GB_MONTH", "0.023"))
S3_IA   = float(os.environ.get("S3_IA_PER_GB_MONTH", "0.0125"))
S3_RETR = float(os.environ.get("S3_DEFAULT_RETRIEVAL_RATE", "0.10"))  # 10% retrieval assumption


def recommend_s3_tiering(df_hourly: pd.DataFrame) -> list[dict]:
    """
    Heuristic: If Requests per GB-month is low, move ~30% to IA.
    Converts TimedStorage-ByteHrs → GB-Month approx (ByteHrs / (GiB * 720)).
    """
    if df_hourly.empty:
        return []

    s3 = df_hourly[df_hourly["service"] == "AmazonS3"].copy()
    if s3.empty:
        return []

    s3["month"] = pd.to_datetime(s3["ts"]).dt.to_period("M")

    storage = (
        s3[s3["usage_type"] == "TimedStorage-ByteHrs"]
        .groupby(["region", "month"], as_index=False)["usage_amount"]
        .sum()
    )
    reqs = (
        s3[s3["usage_type"] == "Requests-Tier1"]
        .groupby(["region", "month"], as_index=False)["usage_amount"]
        .sum()
        .rename(columns={"usage_amount": "requests"})
    )

    storage["gb_month"] = storage["usage_amount"].astype(float) / (1024**3 * 720.0)
    merged = storage.merge(reqs, on=["region", "month"], how="left")
    merged["requests"] = pd.to_numeric(merged["requests"], errors="coerce").fillna(0.0)

    recs: list[dict] = []
    # use most recent month per region
    latest = merged.sort_values("month").groupby("region", as_index=False).tail(1)

    for _, r in latest.iterrows():
        region = r["region"]
        gbm = float(r["gb_month"])
        req = float(r["requests"])
        if gbm <= 0:
            continue

        # rough proxy for coldness
        requests_per_gbm = req / max(gbm, 1e-9)
        if requests_per_gbm < 1.0:  # tune threshold as needed
            move_gb = gbm * 0.30
            base_cost = move_gb * S3_STD
            ia_cost   = move_gb * S3_IA + move_gb * S3_RETR * (S3_STD - S3_IA)
            saving = max(0.0, base_cost - ia_cost)
            if saving >= 5:
                recs.append(
                    {
                        "category": "S3 Tiering",
                        "subtype": "Standard→IA (30%)",
                        "region": region,
                        "assumption": f"~{int(S3_RETR*100)}% monthly retrieval",
                        "metric": f"GB-Month move:{int(move_gb)}",
                        "est_monthly_saving_usd": round(saving, 2),
                        "action_sql_hint": "Enable lifecycle to IA for cold prefixes/buckets; verify retrieval patterns.",
                        "source_note": "heuristics-v1",
                    }
                )
    return recs
