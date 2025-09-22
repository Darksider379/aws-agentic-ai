class build_recommendations_table_if_needed():
    def build_recommendations_table_if_needed(s3_path: str):
        # Create an external table if not exists (CSV)
        ddl = f"""
        CREATE EXTERNAL TABLE IF NOT EXISTS {ATHENA_RECS_TABLE} (
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
        LOCATION '{s3_path}'
        TBLPROPERTIES ('skip.header.line.count'='1')
        """
        run_athena(ddl)