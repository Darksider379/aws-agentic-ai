# recommendations/build_recommendations_table_if_needed.py
from functions.run_athena import run_athena


def build_recommendations_table_if_needed(s3_prefix_uri: str, table_name: str = None):
    """
    Creates an Athena external table over the recommendations CSV prefix if it doesn't exist.
    s3_prefix_uri should be like: s3://bucket/cost-agent/recommendations/
    """
    tbl = table_name or "recommendations"
    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {tbl} (
      run_ts timestamp,
      category string,
      subtype string,
      region string,
      assumption string,
      metric string,
      est_monthly_saving_usd double,
      source_note string
    )
    ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.OpenCSVSerde'
    WITH SERDEPROPERTIES ('separatorChar' = ',', 'quoteChar' = '"', 'escapeChar'='\\\\')
    LOCATION '{s3_prefix_uri}'
    TBLPROPERTIES ('skip.header.line.count'='1')
    """
    run_athena(ddl, expect_result=False)
