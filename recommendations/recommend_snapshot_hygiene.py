# recommendations/recommend_snapshot_hygiene.py
import pandas as pd

def recommend_snapshot_hygiene(hourly_df: pd.DataFrame) -> list[dict]:
    """
    Identify old EBS snapshots and compute one-time savings for deletion.
    Expected (adapt as needed): columns region, snapshot_id, size_gb, age_days, price_per_gb_month
    """
    if hourly_df is None or hourly_df.empty:
        return []

    snaps = extract_snapshot_view(hourly_df)
    if snaps.empty:
        return []

    # Obsolete threshold (e.g., > 90 days old)
    obsolete = snaps[snaps["age_days"] > 90].copy()
    if obsolete.empty:
        return []


    obsolete["one_time_saving_usd"] = obsolete["size_gb"] * obsolete["price_per_gb_month"]

    grp = obsolete.groupby(["region"], dropna=False).agg(
        count=("snapshot_id", "nunique"),
        size_gb=("size_gb", "sum"),
        one_time=("one_time_saving_usd", "sum"),
    ).reset_index()

    recs: list[dict] = []
    for _, row in grp.iterrows():
        region = row["region"] if pd.notna(row["region"]) else ""
        if row["one_time"] <= 0:
            continue
        recs.append({
            "category": "EBS Snapshot Hygiene",
            "subtype": "delete_old",
            "region": region,
            "assumption": "one-time deletion of obsolete snapshots >90d",
            "metric": f"count:{int(row['count'])} size_gb:{int(row['size_gb'])}",
            "est_monthly_saving_usd": 0.0,  # not monthly recurring
            "one_time_saving_usd": round(float(row["one_time"]), 2),
            "action_sql_hint": "Delete snapshots older than 90d not referenced by AMIs/volumes; keep policy exceptions.",
            "source_note": "heuristics-v1",
        })
    return recs

def extract_snapshot_view(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal snapshot view:
      region, snapshot_id, size_gb, age_days, price_per_gb_month
    """
    df = hourly_df.copy()
    # Filter: if you have a service/type column, narrow to EBS snapshots here
    if "region" not in df.columns:
        df["region"] = "us-east-1"
    if "snapshot_id" not in df.columns:
        # if you have no snapshot granularity, don't emit any recs
        return pd.DataFrame(columns=["region", "snapshot_id", "size_gb", "age_days", "price_per_gb_month"])

    # price per GB-month default (illustrative; replace with lookup)
    if "price_per_gb_month" not in df.columns:
        df["price_per_gb_month"] = 0.05  # $/GB-month

    # required columns
    for k, default in [("size_gb", 0.0), ("age_days", 0.0)]:
        if k not in df.columns:
            df[k] = default
        df[k] = pd.to_numeric(df[k], errors="coerce").fillna(0.0)

    keep = ["region", "snapshot_id", "size_gb", "age_days", "price_per_gb_month"]
    return df[keep]
