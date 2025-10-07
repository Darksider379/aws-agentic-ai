# recommendations/recommend_ec2_rightsize.py
import re
import pandas as pd

# Simple on-demand price cache; replace with your pricing source
ONDEMAND_PRICES = {
    "m5.large": 0.096,
    "m7g.large": 0.076,
}

EC2_SERVICE_HINTS = ("AmazonEC2", "EC2")

# regex to pull instance type from usage strings like:
#   "BoxUsage:m5.large", "RunInstances:m5.xlarge", "SpotUsage:m5.large",
#   or even "...:m5.2xlarge"
INSTANCE_FROM_USAGE_RE = re.compile(
    r"(?:(?:Box|Run|Spot)Usage:)?([a-z]\d[\w-]*\.[a-z0-9]+)", re.IGNORECASE
)

def price_per_hour(region: str, instance_type: str) -> float:
    # TODO: use proper regional pricing. For now, fall back if unknown.
    return ONDEMAND_PRICES.get(instance_type, 0.1)

def _infer_region(df: pd.DataFrame) -> pd.Series:
    """
    Try common CUR fields to infer region. Fallback to 'us-east-1'.
    If AZ like us-east-1a is present, strip the last letter.
    """
    region = None
    for col in ("product_region", "line_item_product_region", "region", "aws_region"):
        if col in df.columns:
            region = df[col]
            break
    if region is None and "line_item_availability_zone" in df.columns:
        az = df["line_item_availability_zone"].astype(str)
        region = az.str.replace(r"[a-z]$", "", regex=True)  # us-east-1a -> us-east-1
    if region is None:
        region = pd.Series(["us-east-1"] * len(df), index=df.index)
    return region.fillna("us-east-1")

def _infer_instance_type(df: pd.DataFrame) -> pd.Series:
    """
    Infer instance type from the richest available source:
    - product_instance_type (CUR)
    - line_item_usage_type like 'BoxUsage:m5.large'
    - generic 'instance_type'
    Returns a Series of strings (may contain NaN if not inferable).
    """
    # 1) Direct CUR column
    if "product_instance_type" in df.columns:
        s = df["product_instance_type"].astype(str).str.strip()
        s = s.where(~s.eq("") & ~s.eq("nan"), None)
        if s.notna().any():
            return s

    # 2) usage_type like 'BoxUsage:m5.large'
    for col in ("line_item_usage_type", "usage_type"):
        if col in df.columns:
            def pull(val):
                if pd.isna(val):
                    return None
                m = INSTANCE_FROM_USAGE_RE.search(str(val))
                return m.group(1) if m else None
            s = df[col].apply(pull)
            if s.notna().any():
                return s

    # 3) generic column
    if "instance_type" in df.columns:
        s = df["instance_type"].astype(str).str.strip()
        s = s.where(~s.eq("") & ~s.eq("nan"), None)
        if s.notna().any():
            return s

    # not found
    return pd.Series([None] * len(df), index=df.index)

def _infer_hours(df: pd.DataFrame) -> pd.Series:
    """
    Use CUR usage amount if present (Hrs). Otherwise assume per-hour rows => 1 hour per row.
    """
    for col in ("line_item_usage_amount", "usage_amount", "usagequantity", "usage_quantity"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            # Heuristic: if values look like hours already, use them.
            # If you later detect rates in seconds, divide accordingly.
            return s
    return pd.Series([1.0] * len(df), index=df.index)

def _is_ec2_subset(df: pd.DataFrame) -> pd.Series:
    """
    Try to filter to EC2 compute usage. Very permissive; adjust as needed.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    # Narrow by product code/service if available
    if "line_item_product_code" in df.columns:
        m = df["line_item_product_code"].astype(str)
        mask &= (
            m.str.contains("AmazonEC2", case=False, na=False) |
            m.str.contains("EC2", case=False, na=False)
        )
    if "service" in df.columns:
        s = df["service"].astype(str)
        mask &= (
            s.str.contains("AmazonEC2", case=False, na=False) |
            s.str.contains("EC2", case=False, na=False)
        )
    # If usage_type present, prefer EC2 run/box usage
    if "line_item_usage_type" in df.columns:
        ut = df["line_item_usage_type"].astype(str)
        mask &= ut.str.contains("BoxUsage|RunInstances|SpotUsage", case=False, na=False)
    return mask

def extract_ec2_view(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a minimal EC2 rightsizing view:
      Required output columns: region, instance_id, from_type, to_type, hours
    We infer these from common CUR columns. Rows lacking 'from_type' are dropped.
    """
    df = hourly_df.copy()

    # Filter to EC2-like usage
    df = df[_is_ec2_subset(df)]
    if df.empty:
        return pd.DataFrame(columns=["region", "instance_id", "from_type", "to_type", "hours"])

    # Region
    region = _infer_region(df)

    # Instance type
    from_type = _infer_instance_type(df)

    # Drop rows we cannot classify
    keep_mask = from_type.notna()
    df = df[keep_mask].copy()
    if df.empty:
        return pd.DataFrame(columns=["region", "instance_id", "from_type", "to_type", "hours"])

    from_type = from_type[keep_mask]
    region = region[keep_mask]

    # Hours
    hours = _infer_hours(df)

    # Instance id (best-effort)
    if "resource_id" in df.columns:
        iid = df["resource_id"].astype(str)
    elif "instance_id" in df.columns:
        iid = df["instance_id"].astype(str)
    elif "line_item_resource_id" in df.columns:
        iid = df["line_item_resource_id"].astype(str)
    else:
        # Fallback synthetic id
        iid = (region.astype(str) + "|" + from_type.astype(str) + "|" + df.index.astype(str))

    # Target type heuristic: suggest Graviton within same size if possible
    def to_graviton(t: str) -> str:
        if not isinstance(t, str) or "." not in t:
            return t
        fam, size = t.split(".", 1)
        # common x86 -> ARM swaps
        fam = fam.replace("m5", "m7g").replace("c5", "c7g").replace("r5", "r7g")
        return f"{fam}.{size}"

    to_type = from_type.astype(str).apply(to_graviton)

    out = pd.DataFrame({
        "region": region.astype(str).fillna("us-east-1"),
        "instance_id": iid.astype(str),
        "from_type": from_type.astype(str),
        "to_type": to_type.astype(str),
        "hours": pd.to_numeric(hours, errors="coerce").fillna(0.0),
    })
    # Remove zeros
    out = out[out["hours"] > 0]
    return out.reset_index(drop=True)

def recommend_ec2_rightsize(hourly_df: pd.DataFrame) -> list[dict]:
    """
    Normalize EC2 monthly savings:
      saving = max(0, old$/h - new$/h) * SUM(instance-hours)
    Clamp per-instance monthly delta to avoid runaway totals.
    """
    if hourly_df is None or hourly_df.empty:
        return []

    ec2 = extract_ec2_view(hourly_df)
    if ec2.empty:
        return []

    grp_cols = ["region", "from_type", "to_type"]
    agg = ec2.groupby(grp_cols, dropna=False).agg(
        hours_sum=("hours", "sum"),
        instances=("instance_id", "nunique")
    ).reset_index()

    recs: list[dict] = []
    for _, row in agg.iterrows():
        region = row["region"]
        from_t = row["from_type"]
        to_t   = row["to_type"]
        hours  = float(row["hours_sum"])

        old_p = price_per_hour(region, from_t)
        new_p = price_per_hour(region, to_t)
        delta_per_hour = max(0.0, old_p - new_p)

        # monthly saving = delta_per_hour * instance-hours (already aggregated)
        saving = delta_per_hour * hours

        # Clamp: per-instance monthly delta (assume ~730 h/mo)
        per_inst_cap = min(delta_per_hour * 730.0, 2000.0)  # $2k/mo cap
        approx_instances = max(1.0, float(row["instances"]))
        saving = min(saving, per_inst_cap * approx_instances)

        if saving <= 0:
            continue

        recs.append({
            "category": "EC2 Right-size",
            "subtype": f"{from_t}→{to_t}",
            "region": region,
            "assumption": "ARM compatible workload" if ("m7g" in to_t or "c7g" in to_t or "r7g" in to_t) else "Lower-cost family target",
            "metric": f"Hours:{int(hours)} insts:{int(approx_instances)}",
            "est_monthly_saving_usd": round(float(saving), 2),
            "action_sql_hint": f"Identify ASGs/LaunchTemplates using {from_t}; test {to_t} in {region}.",
            "source_note": "heuristics-v1",
        })
    return recs
