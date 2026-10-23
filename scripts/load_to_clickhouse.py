from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
import urllib.error

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ClickHouseLoader")

DEFAULT_CLICKHOUSE_URL = "http://clickhouse:8123"
DEFAULT_CLICKHOUSE_USER = os.getenv("CLICKHOUSE_USER", "")
DEFAULT_CLICKHOUSE_PASSWORD = os.getenv("CLICKHOUSE_PASSWORD", "")
DEFAULT_BATCH_SUMMARY_HDFS = "hdfs://namenode:9000/aircraft/checkpoints/batch_summary"
DEFAULT_PREDICTIONS_HDFS = "hdfs://namenode:9000/aircraft/predictions/rul_predictions"

CLICKHOUSE_DDL_STATEMENTS = [
    "CREATE DATABASE IF NOT EXISTS aircraft",
    """CREATE TABLE IF NOT EXISTS aircraft.engine_batch_summary (
        dataset LowCardinality(String),
        unit UInt32,
        reading_count UInt64,
        cycle_count UInt64,
        min_cycle Float32,
        max_cycle Float32,
        min_rul Float32,
        max_rul Float32,
        avg_rul Float32,
        created_at DateTime DEFAULT now()
    ) ENGINE = ReplacingMergeTree(created_at) ORDER BY (dataset, unit)""",
    """CREATE TABLE IF NOT EXISTS aircraft.engine_rul_predictions (
        dataset LowCardinality(String),
        split LowCardinality(String),
        unit UInt32,
        cycle UInt32,
        actual_rul Nullable(Float32),
        predicted_rul Float32,
        absolute_error Nullable(Float32),
        predicted_at DateTime DEFAULT now()
    ) ENGINE = MergeTree() PARTITION BY toYYYYMM(predicted_at) ORDER BY (dataset, unit, cycle, predicted_at)""",
    """CREATE TABLE IF NOT EXISTS aircraft.maintenance_alerts (
        alert_time DateTime DEFAULT now(),
        unit UInt32,
        severity LowCardinality(String),
        alert_type LowCardinality(String),
        max_sensor_T30 Float32,
        min_rul Float32,
        diagnosis String
    ) ENGINE = MergeTree() PARTITION BY toYYYYMM(alert_time) ORDER BY (unit, alert_time)""",
    """CREATE TABLE IF NOT EXISTS aircraft.engine_stream_metrics (
        window_start DateTime,
        window_end DateTime,
        unit UInt32,
        reading_count UInt32,
        avg_sensor_T24 Float32,
        avg_sensor_T30 Float32,
        max_sensor_T30 Float32,
        min_rul Float32,
        created_at DateTime DEFAULT now()
    ) ENGINE = MergeTree() PARTITION BY toYYYYMM(window_start) ORDER BY (unit, window_start)""",
]


def ensure_clickhouse_schema(url: str, user: str, password: str) -> None:
    for ddl in CLICKHOUSE_DDL_STATEMENTS:
        req = urllib.request.Request(
            url=f"{url}/",
            data=ddl.encode("utf-8"),
            headers={
                "X-ClickHouse-User": user,
                "X-ClickHouse-Key": password,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as resp:
            resp.read()
    logger.info("✔ Verified ClickHouse database 'aircraft' and all analytics tables.")


def post_json_to_clickhouse(
    url: str,
    table_name: str,
    json_lines: list[str],
    user: str = DEFAULT_CLICKHOUSE_USER,
    password: str = DEFAULT_CLICKHOUSE_PASSWORD,
) -> bool:
    if not json_lines:
        logger.info(f"No records to insert into {table_name}.")
        return True

    payload = "\n".join(json_lines).encode("utf-8")
    query_url = f"{url}/?query=INSERT+INTO+{table_name}+FORMAT+JSONEachRow"

    try:
        req = urllib.request.Request(
            url=query_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-ClickHouse-User": user,
                "X-ClickHouse-Key": password,
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15.0) as response:
            if response.status == 200:
                logger.info(f"✔ Successfully inserted {len(json_lines):,} rows into ClickHouse table: {table_name}")
                return True
            else:
                logger.error(f"✖ ClickHouse error status: {response.status}")
                return False
    except Exception as e:
        logger.error(f"✖ Failed to write to ClickHouse table {table_name}: {e}")
        return False


def load_batch_summary(spark: SparkSession, hdfs_path: str, ch_url: str, user: str, password: str) -> None:
    logger.info(f"Loading Batch Summary from {hdfs_path}...")
    try:
        df = spark.read.parquet(hdfs_path)
    except Exception as e:
        logger.warning(f"Could not read batch summary from {hdfs_path}: {e}")
        return

    rows = df.collect()
    json_lines = []
    for r in rows:
        record = {
            "dataset": str(r.dataset),
            "unit": int(r.unit),
            "reading_count": int(r.reading_count),
            "cycle_count": int(r.cycle_count),
            "min_cycle": float(r.min_cycle),
            "max_cycle": float(r.max_cycle),
            "min_rul": float(r.min_rul) if r.min_rul is not None else 0.0,
            "max_rul": float(r.max_rul) if r.max_rul is not None else 0.0,
            "avg_rul": float(r.avg_rul) if r.avg_rul is not None else 0.0,
        }
        json_lines.append(json.dumps(record))

    post_json_to_clickhouse(ch_url, "aircraft.engine_batch_summary", json_lines, user=user, password=password)


def load_predictions(spark: SparkSession, hdfs_path: str, ch_url: str, user: str, password: str, chunk_size: int = 50000) -> None:
    logger.info(f"Loading Predictions from {hdfs_path}...")
    try:
        df = spark.read.parquet(hdfs_path)
    except Exception as e:
        logger.warning(f"Could not read predictions from {hdfs_path}: {e}")
        return

    has_actual = "actual_rul" in df.columns
    has_error = "absolute_error" in df.columns
    has_split = "split" in df.columns
    has_dataset = "dataset" in df.columns

    formatted_df = df.select(
        (F.col("dataset") if has_dataset else F.lit("N-CMAPSS_DS01-005")).alias("dataset"),
        (F.col("split") if has_split else F.lit("test")).alias("split"),
        F.col("unit").cast("int").alias("unit"),
        F.col("cycle").cast("int").alias("cycle"),
        (F.col("actual_rul").cast("float") if has_actual else F.lit(None)).alias("actual_rul"),
        F.col("predicted_rul").cast("float").alias("predicted_rul"),
        (F.col("absolute_error").cast("float") if has_error else F.lit(None)).alias("absolute_error"),
    )

    total_rows = formatted_df.count()
    logger.info(f"Total prediction records found: {total_rows:,}")

    batch_rows = formatted_df.limit(chunk_size).collect()
    json_lines = []
    for r in batch_rows:
        rec = {
            "dataset": str(r.dataset),
            "split": str(r.split),
            "unit": int(r.unit),
            "cycle": int(r.cycle),
            "actual_rul": float(r.actual_rul) if r.actual_rul is not None else None,
            "predicted_rul": float(r.predicted_rul),
            "absolute_error": float(r.absolute_error) if r.absolute_error is not None else None,
        }
        json_lines.append(json.dumps(rec))

    post_json_to_clickhouse(ch_url, "aircraft.engine_rul_predictions", json_lines, user=user, password=password)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Spark HDFS results to ClickHouse.")
    parser.add_argument("--clickhouse-url", default=DEFAULT_CLICKHOUSE_URL)
    parser.add_argument("--clickhouse-user", default=DEFAULT_CLICKHOUSE_USER)
    parser.add_argument("--clickhouse-password", default=DEFAULT_CLICKHOUSE_PASSWORD)
    parser.add_argument("--summary-input", default=DEFAULT_BATCH_SUMMARY_HDFS)
    parser.add_argument("--predictions-input", default=DEFAULT_PREDICTIONS_HDFS)
    parser.add_argument("--master", default="spark://spark-master:7077")
    args = parser.parse_args()

    if not args.clickhouse_user or not args.clickhouse_password:
        raise SystemExit(
            "ClickHouse credentials missing: set CLICKHOUSE_USER and CLICKHOUSE_PASSWORD "
            "environment variables (or pass --clickhouse-user/--clickhouse-password)."
        )

    spark = (
        SparkSession.builder
        .appName("ExportToClickHouse")
        .master(args.master)
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        ensure_clickhouse_schema(args.clickhouse_url, args.clickhouse_user, args.clickhouse_password)
        load_batch_summary(spark, args.summary_input, args.clickhouse_url, args.clickhouse_user, args.clickhouse_password)
        load_predictions(spark, args.predictions_input, args.clickhouse_url, args.clickhouse_user, args.clickhouse_password)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
