# recommendations/build_recommendations_table_if_needed.py
import os
from functions.run_athena import run_athena

ATHENA_DB         = os.environ["ATHENA_DB"]
DEFAULT_RECS_TBL  = os.environ.get("ATHENA_RECS_TABLE", "recommendations_v2")
RESULTS_BUCKET    = os.environ["RESULTS_BUCKET"]
RESULTS_PREFIX    = os.environ.get("RESULTS_PREFIX", "cost-agent-v2")

# Keep this schema EXACTLY in the same order as CSV_COLUMNS in write_recommendations_csv_to_s3.py
DDL_COLUMNS = """
  `run_id` string,
  `created_at` string,
  `category` string,
  `subtype` string,
  `region` string,
  `assumption` string,
  `metric` string,
  `est_monthly_saving_usd` string,
  `one_time_saving_usd` string,
  `action_sql_hint` string,
  `source_note` string,
  `rline_item_resource_id` string
""".strip()

def _prefix() -> str:
    return f"s3://{RESULTS_BUCKET}/{RESULTS_PREFIX.strip('/')}/recommendations/"

def _table_exists(table: str) -> bool:
    """
    Robust check that honors the passed table name.
    We try two SHOW variants because Athena/Hive can differ by engine:
      - SHOW TABLES IN <db> LIKE '<name>'
      - SHOW TABLES LIKE '<name>'
    """
    queries = [
        f"SHOW TABLES IN {ATHENA_DB} LIKE '{table}'",
        f"SHOW TABLES LIKE '{table}'"
    ]
    for q in queries:
        try:
            df = run_athena(q)
            if df is not None and not df.empty:
                return True
        except Exception:
            # try the next variant
            continue
    return False

def _has_column(table: str, colname: str) -> bool:
    try:
        df = run_athena(f"SHOW COLUMNS IN {ATHENA_DB}.{table}")
        if df is None or df.empty:
            return False
        cname = "column" if "column" in df.columns else df.columns[0]
        cols = [str(x).strip().lower() for x in df[cname].tolist()]
        return colname.lower() in cols
    except Exception:
        return False

def build_recommendations_table_if_needed(s3_prefix_uri: str, table_name: str = None):
    """
    Creates the table if missing (pointing to the recommendations/ prefix),
    and ensures rline_item_resource_id exists. Uses OpenCSVSerde; column order matters.
    """
    table = table_name or DEFAULT_RECS_TBL
    loc = _prefix()

    # Safe CREATE that won't error on re-runs
    if not _table_exists(table):
        ddl = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_DB}.{table}(
          {DDL_COLUMNS}
        )
        ROW FORMAT SERDE
          'org.apache.hadoop.hive.serde2.OpenCSVSerde'
        WITH SERDEPROPERTIES (
          'escapeChar'='\\\\',
          'quoteChar'='\"',
          'separatorChar'=','
        )
        STORED AS INPUTFORMAT
          'org.apache.hadoop.mapred.TextInputFormat'
        OUTPUTFORMAT
          'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
        LOCATION '{loc}'
        TBLPROPERTIES ('skip.header.line.count'='1')
        """
        run_athena(ddl, expect_result=False)
        return

    # If table exists, ensure the resource id column is present
    if not _has_column(table, "rline_item_resource_id"):
        run_athena(
            f"ALTER TABLE {ATHENA_DB}.{table} ADD COLUMNS (rline_item_resource_id string)",
            expect_result=False,
        )
