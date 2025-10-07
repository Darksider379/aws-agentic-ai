# recommendations/recommend_s3_tiering.py
import pandas as pd

# Assumptive regional price fallbacks (illustrative – replace with your lookup if you have one)
# Tuple = (STANDARD, IA) in $/GB-month
PRICE_FALLBACKS = {
    "us-east-1": (0.023, 0.0125),
    "us-west-2": (0.023, 0.0125),
    "eu-west-1": (0.0245, 0.0135),
}

def recommend_s3_tiering(hourly_df: pd.DataFrame) -> list[dict]:
    """
    Produce S3 tiering recommendations using GB-month math ONLY
    (no hourly multipliers; S3 is billed per GB-month).

    Expects an 'hourly_df' with some S3-related rows. This function:
      1) Extracts a minimal S3 view (region, ts/month, storage_gb, current/target prices)
      2) Computes monthly savings = storage_gb * (price_current - price_target)
      3) Aggregates by (region, month) and emits recommendations
    """
    if hourly_df is None or hourly_df.empty:
        return []

    s3 = extract_s3_view(hourly_df)
    if s3.empty:
        return []

    # --- Month derivation (silence tz warning and handle epoch/ISO gracefully) ---
    if "ts" in s3.columns and s3["ts"].notna().any():
        # try epoch seconds first
        dt = pd.to_datetime(s3["ts"], unit="s", utc=True, errors="coerce")
        if dt.isna().all():
            # fallback to ISO like 2025-09-01T12:34:56Z
            dt = pd.to_datetime(s3["ts"], utc=True, format="%Y-%m-%dT%H:%M:%SZ", errors="coerce")
        # strip timezone before converting to Period to avoid warning
        s3["month"] = dt.dt.tz_localize(None).dt.to_period("M")
    elif "month" not in s3.columns:
        s3["month"] = pd.PeriodIndex(pd.to_datetime("today"), freq="M")

    # --- Ensure region present ---
    if "region" not in s3.columns:
        s3["region"] = "us-east-1"
    s3["region"] = s3["region"].fillna("us-east-1").astype(str)

    # --- Ensure price columns exist; fill from fallbacks where missing ---
    for col in ("price_per_gb_month_current", "price_per_gb_month_target"):
        if col not in s3.columns:
            s3[col] = pd.NA

    def _fill_prices(row):
        region = str(row.get("region", "us-east-1"))
        std, ia = PRICE_FALLBACKS.get(region, PRICE_FALLBACKS["us-east-1"])
        cur_p = row.get("price_per_gb_month_current")
        tgt_p = row.get("price_per_gb_month_target")
        cur_p = float(std if pd.isna(cur_p) else cur_p)
        tgt_p = float(ia  if pd.isna(tgt_p) else tgt_p)
        return pd.Series({
            "price_per_gb_month_current": cur_p,
            "price_per_gb_month_target": tgt_p
        })

    s3[["price_per_gb_month_current", "price_per_gb_month_target"]] = s3.apply(_fill_prices, axis=1)

    # --- Compute monthly savings (NO hourly factor) ---
    s3["storage_gb"] = pd.to_numeric(s3.get("storage_gb", 0.0), errors="coerce").fillna(0.0)
    delta = (s3["price_per_gb_month_current"] - s3["price_per_gb_month_target"]).clip(lower=0)
    s3["est_monthly_saving_usd"] = (s3["storage_gb"] * delta).fillna(0.0)

    # --- Aggregate and build recs ---
    grp = (
        s3.groupby(["region", "month"], dropna=False)[["est_monthly_saving_usd", "storage_gb"]]
          .sum()
          .reset_index()
    )

    recs: list[dict] = []
    for _, row in grp.iterrows():
        region = row["region"] if pd.notna(row["region"]) else ""
        saving = float(row["est_monthly_saving_usd"])
        if saving <= 0:
            continue
        recs.append({
            "category": "S3 Tiering",
            "subtype": "Standard→IA",
            "region": region,
            "assumption": "Move cold prefixes from STANDARD to IA; confirm retrieval patterns",
            "metric": f"month:{row['month']} storage_gb:{round(float(row['storage_gb']),2)}",
            "est_monthly_saving_usd": round(saving, 2),
            "action_sql_hint": "Enable lifecycle to IA for low-GET prefixes; verify retrievals.",
            "source_note": "heuristics-v1",
        })
    return recs

def extract_s3_view(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract a minimal S3 storage view from the broader hourly DataFrame.
    Tries to find S3 rows and produce columns:
      ts (or month), region, storage_gb, price_per_gb_month_current, price_per_gb_month_target

    If only GB-hour is present, converts to GB-month via sum(GB-hour)/730.
    """
    df = hourly_df.copy()

    # Try to filter to S3-like rows using a 'service' column if present
    if "service" in df.columns:
        s = df["service"].astype(str)
        df = df[s.str.contains("Storage", case=False, na=False) | s.str.contains("S3", case=False, na=False)]

    # region best-effort
    if "region" not in df.columns:
        for col in ("product_region", "line_item_product_region", "aws_region"):
            if col in df.columns:
                df["region"] = df[col]
                break
    df["region"] = df.get("region", "us-east-1").fillna("us-east-1")

    # storage_gb derivation
    if "storage_gb" not in df.columns:
        if "storage_gb_hour" in df.columns:
            # convert GB-hour → GB-month
            df["storage_gb"] = pd.to_numeric(df["storage_gb_hour"], errors="coerce").fillna(0.0) / 730.0
        elif "usage_quantity" in df.columns and "unit" in df.columns:
            # if unit indicates GB-Hours
            mask = df["unit"].astype(str).str.contains("GB-Hours?|GBH", case=False, na=False)
            gb_hours = pd.to_numeric(df.loc[mask, "usage_quantity"], errors="coerce").fillna(0.0)
            df["storage_gb"] = 0.0
            df.loc[mask, "storage_gb"] = gb_hours / 730.0
        else:
            df["storage_gb"] = 0.0

    # timestamp passthrough if present
    if "ts" not in df.columns:
        # leave missing; caller handles month fallback
        df["ts"] = pd.NA

    # Ensure price columns exist (may be filled via fallbacks later)
    for k in ("price_per_gb_month_current", "price_per_gb_month_target"):
        if k not in df.columns:
            df[k] = pd.NA

    keep = ["ts", "month", "region", "storage_gb", "price_per_gb_month_current", "price_per_gb_month_target"]
    for k in keep:
        if k not in df.columns:
            df[k] = pd.NA
    return df[keep]
