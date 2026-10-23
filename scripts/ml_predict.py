from __future__ import annotations

import argparse
import sys

from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


DEFAULT_MODEL_PATH = (
    "hdfs://namenode:9000/"
    "aircraft/models/rul_model"
)

DEFAULT_INPUT = (
    "hdfs://namenode:9000/"
    "aircraft/processed/"
    "N-CMAPSS_DS01-005/test.parquet"
)

DEFAULT_OUTPUT = (
    "hdfs://namenode:9000/"
    "aircraft/predictions/rul_predictions"
)


def run_prediction(
    spark: SparkSession,
    model_path: str,
    input_path: str,
    output_path: str,
) -> DataFrame:
    print("\n" + "#" * 70)
    print("            PREDICTIVE MAINTENANCE: RUL INFERENCE PIPELINE            ")
    print("#" * 70)
    print(f"[*] Loading Saved Model from: {model_path}")
    print(f"[*] Input Data to Predict   : {input_path}")
    print(f"[*] Predictions Output Path : {output_path}")

    # 1. Load the pre-trained Spark MLlib PipelineModel from HDFS
    try:
        model = PipelineModel.load(model_path)
        print("✔ PipelineModel loaded successfully from HDFS.")
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load model from HDFS path ({model_path}): {e}\n"
            "Ensure that the training script ml_train.py has been executed successfully first."
        )

    # 2. Read test or production inference input dataset from Parquet
    df = spark.read.parquet(input_path)
    total_rows = df.count()
    print(f"✔ Read {total_rows:,} records from input Parquet.")

    # 3. Perform distributed batch inference using the loaded model
    print("[*] Generating RUL predictions...")
    predictions = model.transform(df)

    has_ground_truth = "rul" in df.columns

    # 4. Compute evaluation metrics if ground-truth labels are available in the dataset
    if has_ground_truth:
        print("[*] Ground-truth 'rul' detected. Computing evaluation metrics...")
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

        test_rmse = evaluator_rmse.evaluate(predictions)
        test_mae = evaluator_mae.evaluate(predictions)
        test_r2 = evaluator_r2.evaluate(predictions)

        print("-" * 50)
        print(f"  Test Evaluation RMSE: {test_rmse:.4f}")
        print(f"  Test Evaluation MAE : {test_mae:.4f}")
        print(f"  Test Evaluation R2  : {test_r2:.4f}")
        print("-" * 50)

        # Prepare output schema including error calculation against actual values
        output_df = predictions.select(
            "unit",
            "cycle",
            F.round("rul", 2).alias("actual_rul"),
            F.round("prediction", 2).alias("predicted_rul"),
            F.round(F.abs(F.col("rul") - F.col("prediction")), 2).alias("absolute_error"),
            *[c for c in ["dataset", "split"] if c in df.columns],
        )
    else:
        print("[*] No 'rul' label found. Operating in pure Production Inference mode.")
        output_df = predictions.select(
            "unit",
            "cycle",
            F.round("prediction", 2).alias("predicted_rul"),
            *[c for c in ["dataset", "split"] if c in df.columns],
        )

    # 5. Persist prediction results back to Hadoop HDFS
    print(f"[*] Saving predictions to HDFS: {output_path} ...")
    output_df.write.mode("overwrite").parquet(output_path)
    print("✔ Predictions saved successfully.")

    # 6. Display a preview sample of the inference results
    print("\n[*] Sample Predictions Preview:")
    output_df.show(20, truncate=False)

    # 7. Aggregate engine-level summary (lowest predicted RUL indicates upcoming maintenance urgency)
    print("[*] Critical Engine Maintenance Summary (Lowest Predicted RUL per Engine):")
    engine_summary = (
        output_df.groupBy("unit")
        .agg(
            F.max("cycle").alias("latest_cycle"),
            F.min("predicted_rul").alias("min_predicted_rul"),
            F.round(F.avg("predicted_rul"), 2).alias("avg_predicted_rul"),
        )
        .orderBy("min_predicted_rul")
    )
    engine_summary.show(truncate=False)

    return output_df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RUL batch inference with trained Spark MLlib model."
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to saved PipelineModel in HDFS.",
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help="Input Parquet data for inference.",
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="HDFS destination path for prediction results.",
    )
    parser.add_argument(
        "--master",
        default="spark://spark-master:7077",
        help="Spark master URL.",
    )

    args = parser.parse_args()

    spark = (
        SparkSession.builder
        .appName("AircraftRULModelInference")
        .master(args.master)
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.hadoop.dfs.client.use.datanode.hostname", "false")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    try:
        run_prediction(
            spark=spark,
            model_path=args.model_path,
            input_path=args.input,
            output_path=args.output,
        )
        print("\n" + "=" * 70)
        print(" PREDICTION PIPELINE COMPLETED SUCCESSFULLY! ")
        print("=" * 70)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()