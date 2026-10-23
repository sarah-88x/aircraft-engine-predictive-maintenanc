from __future__ import annotations

import argparse
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, FloatType, IntegerType, LongType, NumericType


DEFAULT_INPUT = (
    "hdfs://namenode:9000/"
    "aircraft/processed/"
    "N-CMAPSS_DS01-005/dev.parquet"
)

DEFAULT_MODEL_DIR = (
    "hdfs://namenode:9000/"
    "aircraft/models"
)


def run_compatibility_check(
    spark: SparkSession,
    input_path: str,
    model_dir: str,
) -> bool:
    print("=" * 70)
    print("          SPARK ML COMPATIBILITY & SCHEMA VERIFICATION          ")
    print("=" * 70)
    print(f"[*] Checking Input Dataset: {input_path}")
    print(f"[*] Target Model Directory: {model_dir}")
    print("-" * 70)

    all_passed = True

    # 1. Check Data Readability
    try:
        df = spark.read.parquet(input_path)
        row_count = df.count()
        col_count = len(df.columns)
        print(f"[✔ PASS] Successfully read Parquet from HDFS.")
        print(f"         Total Rows   : {row_count:,}")
        print(f"         Total Columns: {col_count}")
    except Exception as e:
        print(f"[✖ FAIL] Unable to read Parquet data from HDFS: {e}")
        return False

    if row_count == 0:
        print("[✖ FAIL] Dataset is empty (0 rows).")
        return False

    # 2. Check Essential Identifier Columns
    required_id_cols = {"unit", "cycle"}
    missing_id_cols = required_id_cols.difference(df.columns)
    if missing_id_cols:
        print(f"[✖ FAIL] Missing identifier columns: {missing_id_cols}")
        all_passed = False
    else:
        print(f"[✔ PASS] Identifier columns present: {sorted(required_id_cols)}")

    # 3. Check Target Label ('rul') - Strict Validation (No Mocking)
    if "rul" not in df.columns:
        print("[✖ FAIL] CRITICAL: Target column 'rul' does NOT exist!")
        print("         ML training requires true ground-truth RUL labels.")
        all_passed = False
    else:
        rul_type = dict(df.dtypes).get("rul")
        rul_stats = df.select(
            F.count("rul").alias("non_null_count"),
            F.count(F.when(F.col("rul").isNull(), 1)).alias("null_count"),
            F.count(F.when(F.isnan("rul"), 1)).alias("nan_count"),
            F.min("rul").alias("min_val"),
            F.max("rul").alias("max_val"),
            F.avg("rul").alias("avg_val"),
        ).collect()[0]

        if rul_stats["null_count"] > 0 or rul_stats["nan_count"] > 0:
            print(f"[✖ FAIL] 'rul' column contains invalid values:")
            print(f"         Nulls: {rul_stats['null_count']}, NaNs: {rul_stats['nan_count']}")
            all_passed = False
        elif rul_stats["non_null_count"] == 0:
            print("[✖ FAIL] 'rul' column has zero valid entries!")
            all_passed = False
        else:
            print(f"[✔ PASS] Target column 'rul' verified:")
            print(f"         Type: {rul_type} | Min: {rul_stats['min_val']:.2f} | Max: {rul_stats['max_val']:.2f} | Avg: {rul_stats['avg_val']:.2f}")

    # 4. Check Engine 'unit' distribution for Data-Leakage Free Split
    if "unit" in df.columns:
        units = [row.unit for row in df.select("unit").distinct().orderBy("unit").collect()]
        unit_count = len(units)
        print(f"[*] Engine Units Detected: {unit_count} units -> {units}")
        if unit_count < 3:
            print(f"[⚠ WARN] Only {unit_count} engine units found. Strict 3-way split (Train/Val/Test)")
            print("         requires at least 3 distinct units to avoid leakage.")
            all_passed = False
        else:
            print(f"[✔ PASS] Sufficient engine units ({unit_count}) for strict Unit-based partitioning.")

    # 5. Check Feature Columns
    excluded_cols = {"unit", "cycle", "rul", "dataset", "split", "source_row"}
    feature_cols = [
        col_name for col_name, dtype in df.dtypes
        if col_name not in excluded_cols
        and dtype in {"double", "float", "int", "bigint", "smallint", "tinyint"}
    ]

    if not feature_cols:
        print("[✖ FAIL] No numeric sensor or operational features found for VectorAssembler!")
        all_passed = False
    else:
        print(f"[✔ PASS] Detected {len(feature_cols)} numeric feature columns for ML modeling.")
        sensor_cols = [c for c in feature_cols if c.startswith("sensor_")]
        setting_cols = [c for c in feature_cols if c.startswith("W_") or c.startswith("T_")]
        v_sensor_cols = [c for c in feature_cols if c.startswith("virtual_sensor_")]
        print(f"         - Physical Sensors   : {len(sensor_cols)}")
        print(f"         - Operational / State: {len(setting_cols)}")
        print(f"         - Virtual Sensors    : {len(v_sensor_cols)}")

    # 6. Check for Nulls across Feature Sample
    sample_df = df.limit(5000)
    null_exprs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c) for c in feature_cols[:10]]
    sample_nulls = sample_df.agg(*null_exprs).collect()[0].asDict()
    total_nulls_in_sample = sum(sample_nulls.values())

    if total_nulls_in_sample > 0:
        print(f"[⚠ WARN] Sample check revealed {total_nulls_in_sample} nulls in first 10 features.")
        print("         Pipeline will use handleInvalid='skip' in VectorAssembler.")
    else:
        print("[✔ PASS] Feature sample shows 0 nulls. Clean schema for VectorAssembler.")

    print("-" * 70)
    if all_passed:
        print("[✔ RESULT] COMPATIBILITY CHECK PASSED! Data is 100% ready for ml_train.py")
    else:
        print("[✖ RESULT] COMPATIBILITY ISSUES FOUND. Please resolve errors before training.")
    print("=" * 70)

    return all_passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify HDFS Parquet dataset compatibility with Spark MLlib models."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input Parquet path in HDFS.",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help="Target model directory in HDFS.",
    )
    parser.add_argument(
        "--master",
        default="spark://spark-master:7077",
        help="Spark master URL.",
    )

    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("AircraftMLCompatibilityVerification")
        .master(args.master)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    try:
        passed = run_compatibility_check(
            spark=spark,
            input_path=args.input,
            model_dir=args.model_dir,
        )
        if not passed:
            sys.exit(1)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
