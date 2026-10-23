from __future__ import annotations

import argparse
import random
import sys
import time

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.regression import (
    GBTRegressor,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_INPUT = (
    "hdfs://namenode:9000/"
    "aircraft/processed/"
    "N-CMAPSS_DS01-005/dev.parquet"
)

DEFAULT_MODEL_OUTPUT = (
    "hdfs://namenode:9000/"
    "aircraft/models/rul_model"
)


def split_data_by_unit(
    df: DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[DataFrame, DataFrame, DataFrame, dict[str, list]]:
    """
    Split dataset strictly by engine unit ID to prevent Data Leakage.
    Ensures that readings from the same engine unit do not overlap across train, validation, and test splits.
    """
    distinct_units = [
        row.unit
        for row in df.select("unit").distinct().orderBy("unit").collect()
    ]

    total_units = len(distinct_units)
    if total_units < 3:
        raise ValueError(
            f"❌ Detected engine units ({total_units}) are insufficient for strict three-way splitting. "
            "At least 3 engine units are required to perform a data-leakage-free Train/Val/Test split."
        )

    # Randomly shuffle engine units using a fixed seed for determinism
    rng = random.Random(seed)
    shuffled_units = distinct_units.copy()
    rng.shuffle(shuffled_units)

    # Compute unit allocation counts per split
    n_train = max(1, int(round(total_units * train_ratio)))
    n_val = max(1, int(round(total_units * val_ratio)))
    
    # Guarantee at least one engine unit is allocated for testing
    if n_train + n_val >= total_units:
        n_train = total_units - 2
        n_val = 1

    train_units = sorted(shuffled_units[:n_train])
    val_units = sorted(shuffled_units[n_train : n_train + n_val])
    test_units = sorted(shuffled_units[n_train + n_val :])

    # Strict disjointness assertions to eliminate data leakage vectors
    assert set(train_units).isdisjoint(set(val_units)), "Data Leakage: Overlap detected between Train and Val sets!"
    assert set(train_units).isdisjoint(set(test_units)), "Data Leakage: Overlap detected between Train and Test sets!"
    assert set(val_units).isdisjoint(set(test_units)), "Data Leakage: Overlap detected between Val and Test sets!"

    train_df = df.filter(F.col("unit").isin(train_units))
    val_df = df.filter(F.col("unit").isin(val_units))
    test_df = df.filter(F.col("unit").isin(test_units))

    unit_split_info = {
        "train": train_units,
        "val": val_units,
        "test": test_units,
    }

    return train_df, val_df, test_df, unit_split_info


def select_feature_columns(df: DataFrame) -> list[str]:
    """
    Extract numeric sensor and operational feature columns,
    excluding identifiers and target labels to prevent model distortion.
    """
    excluded = {
        "unit",
        "cycle",
        "rul",
        "dataset",
        "split",
        "source_row",
    }

    numeric_types = {
        "double",
        "float",
        "int",
        "bigint",
        "smallint",
        "tinyint",
    }

    feature_cols = [
        col_name
        for col_name, dtype in df.dtypes
        if col_name not in excluded and dtype in numeric_types
    ]

    feature_cols.sort()
    return feature_cols


def print_comparison_table(results: list[dict]) -> None:
    """Print a formatted benchmark evaluation comparison table to the console."""
    print("\n" + "=" * 76)
    print("                    SPARK MLLIB MODEL BENCHMARK RESULTS                    ")
    print("=" * 76)
    header = f"| {'Model Name':<24} | {'Val RMSE':<12} | {'Val MAE':<12} | {'Val R2':<12} |"
    print(header)
    print("-" * 76)
    for r in results:
        row_str = (
            f"| {r['name']:<24} "
            f"| {r['rmse']:<12.4f} "
            f"| {r['mae']:<12.4f} "
            f"| {r['r2']:<12.4f} |"
        )
        print(row_str)
    print("=" * 76 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train & Compare Spark MLlib regression models for Aircraft RUL."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input Parquet path in HDFS.",
    )
    parser.add_argument(
        "--output-model",
        default=DEFAULT_MODEL_OUTPUT,
        help="Destination path in HDFS to save the best trained PipelineModel.",
    )
    parser.add_argument(
        "--master",
        default="spark://spark-master:7077",
        help="Spark master URL.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Proportion of engine units for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of engine units for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of engine units for testing.",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Optional sampling fraction (e.g. 0.1 for 10% sample) to speed up training on large datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic partitioning.",
    )

    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("AircraftRULModelTraining")
        .master(args.master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        print("\n" + "#" * 70)
        print("          STARTING PREDICTIVE MAINTENANCE ML TRAINING PIPELINE          ")
        print("#" * 70)
        print(f"[*] Input Data Path : {args.input}")
        print(f"[*] Best Model Target: {args.output_model}")
        print(f"[*] Spark Master     : {args.master}")
        print(f"[*] Sample Ratio     : {args.sample_ratio}")

        # 1. Read input dataset from HDFS Parquet storage
        df = spark.read.parquet(args.input)

        # 2. Strict validation of target column (absolute prohibition of synthetic RUL fabrication)
        if "rul" not in df.columns:
            raise ValueError(
                "❌ [Critical Data Error] Target column 'rul' is missing from the input dataset!\n"
                "Training models without authentic ground-truth reference labels is strictly prohibited.\n"
                "Execution halted to protect model credibility; no synthetic values will be fabricated."
            )

        valid_rul_count = df.filter(F.col("rul").isNotNull() & (~F.isnan("rul"))).count()
        if valid_rul_count == 0:
            raise ValueError(
                "❌ [Label Error] Target column 'rul' is completely empty or lacks valid numeric values!"
            )

        # Filter out any malformed rows lacking valid labels
        df = df.filter(F.col("rul").isNotNull() & (~F.isnan("rul")))

        # 3. Identify and isolate feature columns (sensors and operational variables)
        feature_cols = select_feature_columns(df)
        print(f"[*] Selected Feature Columns ({len(feature_cols)} total):")
        print(f"    Sample features: {feature_cols[:6]} ... {feature_cols[-3:]}")

        # 4. Partition dataset strictly by engine unit ID to maintain anti-data-leakage constraints
        print("\n[*] Splitting dataset strictly by engine 'unit' to prevent Data Leakage...")
        train_df, val_df, test_df, split_info = split_data_by_unit(
            df=df,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        # Apply optional sampling fraction to accelerate training and mitigate memory overhead
        if args.sample_ratio < 1.0:
            print(f"[*] Applying sample ratio of {args.sample_ratio:.2f} on training set...")
            train_df = train_df.sample(withReplacement=False, fraction=args.sample_ratio, seed=args.seed)

        n_train = train_df.count()
        n_val = val_df.count()
        n_test = test_df.count()

        print(f"    -> [Train Set] Units: {split_info['train']} | Rows: {n_train:,}")
        print(f"    -> [Val Set]   Units: {split_info['val']} | Rows: {n_val:,}")
        print(f"    -> [Test Set]  Units: {split_info['test']} | Rows: {n_test:,}")

        # 5. Build core feature engineering pipeline stages
        assembler = VectorAssembler(
            inputCols=feature_cols,
            outputCol="raw_features",
            handleInvalid="skip",
        )

        scaler = StandardScaler(
            inputCol="raw_features",
            outputCol="features",
            withMean=True,
            withStd=True,
        )

        # 6. Define candidate regression models with memory- and performance-optimized hyperparameters
        candidate_models = [
            (
                "Linear Regression",
                LinearRegression(
                    featuresCol="features",
                    labelCol="rul",
                    maxIter=50,
                    regParam=0.1,
                    elasticNetParam=0.2,
                ),
            ),
            (
                "Random Forest",
                RandomForestRegressor(
                    featuresCol="features",
                    labelCol="rul",
                    numTrees=25,
                    maxDepth=6,
                    seed=args.seed,
                ),
            ),
            (
                "GBT Regressor",
                GBTRegressor(
                    featuresCol="features",
                    labelCol="rul",
                    maxIter=25,
                    maxDepth=5,
                    seed=args.seed,
                ),
            ),
        ]

        # 7. Initialize regression performance evaluators
        evaluator_rmse = RegressionEvaluator(
            labelCol="rul",
            predictionCol="prediction",
            metricName="rmse",
        )
        evaluator_mae = RegressionEvaluator(
            labelCol="rul",
            predictionCol="prediction",
            metricName="mae",
        )
        evaluator_r2 = RegressionEvaluator(
            labelCol="rul",
            predictionCol="prediction",
            metricName="r2",
        )

        model_results = []
        trained_pipelines = {}

        # 8. Train and evaluate each candidate model iteratively
        for model_name, estimator in candidate_models:
            print(f"\n[*] Training {model_name}...")
            start_time = time.time()

            pipeline = Pipeline(stages=[assembler, scaler, estimator])
            fitted_pipeline = pipeline.fit(train_df)
            elapsed = time.time() - start_time

            # Evaluate model performance on the Validation Set
            val_preds = fitted_pipeline.transform(val_df)
            rmse = evaluator_rmse.evaluate(val_preds)
            mae = evaluator_mae.evaluate(val_preds)
            r2 = evaluator_r2.evaluate(val_preds)

            print(f"    Completed in {elapsed:.1f}s | RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            model_results.append({
                "name": model_name,
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "pipeline": fitted_pipeline,
            })
            trained_pipelines[model_name] = fitted_pipeline

        # 9. Print comprehensive model comparison results table
        print_comparison_table(model_results)

        # 10. Select the best performing model based on validation RMSE and persist it
        best_candidate = min(model_results, key=lambda x: x["rmse"])
        best_name = best_candidate["name"]
        best_pipeline = best_candidate["pipeline"]

        print(f"[🏆 WINNER] Best Model: '{best_name}' with lowest Validation RMSE: {best_candidate['rmse']:.4f}")

        # Evaluate the winning model on the completely independent Test Set (Out-of-Sample)
        print("\n[*] Evaluating Best Model on unseen Test Engines (Out-of-Sample)...")
        test_preds = best_pipeline.transform(test_df)
        test_rmse = evaluator_rmse.evaluate(test_preds)
        test_mae = evaluator_mae.evaluate(test_preds)
        test_r2 = evaluator_r2.evaluate(test_preds)

        print(f"    -> Test RMSE: {test_rmse:.4f}")
        print(f"    -> Test MAE : {test_mae:.4f}")
        print(f"    -> Test R2  : {test_r2:.4f}")

        # Persist the complete PipelineModel bundle to HDFS storage
        print(f"\n[*] Saving Best Model Pipeline to HDFS: {args.output_model} ...")
        best_pipeline.write().overwrite().save(args.output_model)
        print("    ✔ Model successfully saved with Overwrite mode.")

        # Display a concise sample of test set predictions
        print("\n[*] Sample Predictions from Test Set:")
        test_preds.select(
            "unit",
            "cycle",
            F.round("rul", 2).alias("actual_rul"),
            F.round("prediction", 2).alias("predicted_rul"),
            F.round(F.abs(F.col("rul") - F.col("prediction")), 2).alias("error"),
        ).show(10, truncate=False)

        print("\n" + "=" * 70)
        print(" ML TRAINING PIPELINE FINISHED SUCCESSFULLY! ")
        print("=" * 70)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()