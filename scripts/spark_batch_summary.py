from __future__ import annotations

import argparse

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_INPUT = (
    "hdfs://namenode:9000/"
    "aircraft/processed/"
    "N-CMAPSS_DS01-005/dev.parquet"
)

DEFAULT_OUTPUT = (
    "hdfs://namenode:9000/"
    "aircraft/checkpoints/batch_summary"
)


def build_summary(df: DataFrame) -> DataFrame:
    """
    Compute aggregate statistics for each engine unit within each dataset.
    """

    summary = (
        df.groupBy("dataset", "unit")
        .agg(
            F.count("*").alias("reading_count"),
            F.countDistinct("cycle").alias("cycle_count"),
            F.min("cycle").alias("min_cycle"),
            F.max("cycle").alias("max_cycle"),
            F.min("rul").alias("min_rul"),
            F.max("rul").alias("max_rul"),
            F.avg("rul").alias("avg_rul"),
        )
        .withColumn(
            "avg_rul",
            F.round(F.col("avg_rul"), 4),
        )
        .orderBy("dataset", "unit")
    )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create aircraft RUL batch summary with Spark."
    )

    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input Parquet path in HDFS.",
    )

    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Output Parquet path in HDFS.",
    )

    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("AircraftBatchSummary")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("=== Aircraft Batch Summary ===")
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")

        df = spark.read.parquet(args.input)

        required_columns = {
            "dataset",
            "unit",
            "cycle",
            "rul",
        }

        missing_columns = required_columns.difference(df.columns)

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{sorted(missing_columns)}"
            )

        print(f"Input columns: {len(df.columns)}")
        print("Building summary...")

        summary_df = build_summary(df)

        summary_count = summary_df.count()

        (
            summary_df.write
            .mode("overwrite")
            .parquet(args.output)
        )

        print(f"Summary rows: {summary_count:,}")

        print("Summary preview:")
        summary_df.show(20, truncate=False)

        print("Batch summary completed successfully.")

    finally:
        spark.stop()


if __name__ == "__main__":
    main()