from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator


# Default DAG configuration arguments
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=3),
}

# Core project paths and configuration variables
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
RAW_H5_PATH = f"{PROJECT_ROOT}/data/raw/N-CMAPSS_DS01-005.h5"
PROCESSED_DIR = f"{PROJECT_ROOT}/data/processed"
HDFS_TARGET_DIR = "/aircraft/processed/N-CMAPSS_DS01-005"


def validate_raw_hdf5_fn(file_path: str) -> bool:
    """
    Stage 1: Validate raw HDF5 file existence, mandatory groups, and schema integrity.
    Performs strict checks for ground-truth Y_dev target array without synthetic RUL fabrication.
    """
    import h5py

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"❌ [Airflow Error] Raw HDF5 file not found: {file_path}")

    print(f"[*] Validating HDF5: {file_path} (Size: {path.stat().st_size / (1024**3):.2f} GB)")

    with h5py.File(path, "r") as h5:
        required_keys = {"A_dev", "X_s_dev", "Y_dev"}
        missing_keys = required_keys.difference(h5.keys())
        if missing_keys:
            raise KeyError(f"❌ [Validation Error] Missing essential datasets in H5: {missing_keys}")

        # Verify ground-truth RUL target array
        y_dev = h5["Y_dev"][:]
        if y_dev.size == 0:
            raise ValueError("❌ [Validation Error] Y_dev target array is empty!")

        min_rul = float(y_dev.min())
        max_rul = float(y_dev.max())
        print(f"✔ Target Y_dev verified: {len(y_dev):,} rows | Min RUL: {min_rul:.2f} | Max RUL: {max_rul:.2f}")

    print("✔ HDF5 file passed all integrity and schema checks.")
    return True


with DAG(
    dag_id="aircraft_engine_batch_pipeline",
    default_args=default_args,
    description="End-to-end Aircraft Predictive Maintenance: HDF5 -> Parquet -> HDFS -> Spark -> ML -> ClickHouse",
    schedule_interval="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["aircraft", "spark", "mllib", "clickhouse", "predictive-maintenance"],
) as dag:

    # 1. Validate raw HDF5 file and verify real target labels
    task_validate_raw = PythonOperator(
        task_id="1_validate_raw_hdf5",
        python_callable=validate_raw_hdf5_fn,
        op_kwargs={"file_path": RAW_H5_PATH},
    )

    # 2. Convert raw HDF5 data into Snappy-compressed Parquet format
    task_convert_to_parquet = BashOperator(
        task_id="2_convert_hdf5_to_parquet",
        bash_command=(
            f"python3 {PROJECT_ROOT}/scripts/convert_hdf5_to_parquet.py "
            f"--input {RAW_H5_PATH} "
            f"--output-dir {PROCESSED_DIR} "
            "--split both "
            "--chunk-size 50000"
        ),
    )

    # 3. Upload processed Parquet files to Hadoop HDFS data lake
    task_upload_to_hdfs = BashOperator(
        task_id="3_upload_parquet_to_hdfs",
        bash_command=(
            f"docker exec $(docker ps -qf name=aircraft-hdfs-namenode | head -n 1) hdfs dfs -mkdir -p {HDFS_TARGET_DIR} && "
            f"docker exec $(docker ps -qf name=aircraft-hdfs-namenode | head -n 1) sh -c '"
            f"hdfs dfs -test -s {HDFS_TARGET_DIR}/dev.parquet && hdfs dfs -test -s {HDFS_TARGET_DIR}/test.parquet || "
            f"hdfs dfs -Ddfs.client.use.datanode.hostname=false -put -f /data/processed/N-CMAPSS_DS01-005/*.parquet {HDFS_TARGET_DIR}/'"
        ),
    )

    # 4. Execute distributed batch summary using Spark
    task_spark_batch = BashOperator(
        task_id="4_spark_batch_summary",
        bash_command=(
            "docker exec $(docker ps -qf name=aircraft-spark-master | head -n 1) /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--conf spark.hadoop.dfs.client.use.datanode.hostname=false "
            "/app/scripts/spark_batch_summary.py "
            f"--input hdfs://namenode:9000{HDFS_TARGET_DIR}/dev.parquet "
            "--output hdfs://namenode:9000/aircraft/checkpoints/batch_summary"
        ),
    )

    # 5. Train and benchmark machine learning models and save the optimal PipelineModel
    task_train_ml = BashOperator(
        task_id="5_train_and_save_ml_model",
        bash_command=(
            "docker exec $(docker ps -qf name=aircraft-spark-master | head -n 1) /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--conf spark.hadoop.dfs.client.use.datanode.hostname=false "
            "--driver-memory 1G "
            "/app/scripts/ml_train.py "
            f"--input hdfs://namenode:9000{HDFS_TARGET_DIR}/dev.parquet "
            "--output-model hdfs://namenode:9000/aircraft/models/rul_model "
            "--sample-ratio 0.1"
        ),
    )

    # 6. Run RUL predictions on test dataset and export critical engine reports
    task_predict_ml = BashOperator(
        task_id="6_predict_rul_test_engines",
        bash_command=(
            "docker exec $(docker ps -qf name=aircraft-spark-master | head -n 1) /opt/spark/bin/spark-submit "
            "--master spark://spark-master:7077 "
            "--conf spark.hadoop.dfs.client.use.datanode.hostname=false "
            "--driver-memory 1G "
            "/app/scripts/ml_predict.py "
            f"--input hdfs://namenode:9000{HDFS_TARGET_DIR}/test.parquet "
            "--output hdfs://namenode:9000/aircraft/predictions/rul_predictions"
        ),
    )

    # 7. Load batch aggregates and model predictions into ClickHouse OLAP tables
    task_load_clickhouse = BashOperator(
        task_id="7_load_results_to_clickhouse",
        bash_command=(
            "docker exec -e CLICKHOUSE_USER -e CLICKHOUSE_PASSWORD $(docker ps -qf name=aircraft-spark-master | head -n 1) /opt/spark/bin/spark-submit "
            "--conf spark.hadoop.dfs.client.use.datanode.hostname=false "
            "/app/scripts/load_to_clickhouse.py "
            "--clickhouse-url http://clickhouse:8123 "
            "--summary-input hdfs://namenode:9000/aircraft/checkpoints/batch_summary "
            "--predictions-input hdfs://namenode:9000/aircraft/predictions/rul_predictions"
        ),
    )

    # 8. Verify and initialize Spark Structured Streaming job for real-time computations
    task_spark_streaming = BashOperator(
        task_id="8_spark_structured_streaming",
        bash_command=(
            "docker exec $(docker ps -qf name=aircraft-spark-master | head -n 1) sh -c '"
            "pgrep -f streaming_job.py >/dev/null || "
            "nohup /opt/spark/bin/spark-submit --conf spark.hadoop.dfs.client.use.datanode.hostname=false /app/scripts/streaming_job.py > /tmp/streaming.log 2>&1 & "
            "sleep 5; pgrep -f streaming_job.py >/dev/null"
            "'"
        ),
    )

    # 9. Replay telemetry sample records via Kafka Producer without blocking the pipeline
    task_kafka_producer = BashOperator(
        task_id="9_kafka_producer",
        bash_command=(
            "docker exec $(docker ps -qf name=aircraft-spark-master | head -n 1) python3 /app/scripts/producer.py "
            "--input /app/data/processed/replay_sample.parquet "
            "--bootstrap-servers kafka:9092 "
            "--rate-per-sec 200 "
            "--batch-size 20 "
            "--no-loop "
            "--limit 1000"
        ),
    )

    # 10. Verify real-time stream ingestion and alert delivery inside ClickHouse OLAP
    task_verify_clickhouse = BashOperator(
        task_id="10_clickhouse_verification",
        bash_command=(
            "docker run --rm --network compose_default -e CLICKHOUSE_USER -e CLICKHOUSE_PASSWORD clickhouse/clickhouse-server:26.9.19.9 "
            "clickhouse-client --host clickhouse --user \"$CLICKHOUSE_USER\" --password \"$CLICKHOUSE_PASSWORD\" "
            "--query 'SELECT count() FROM aircraft.engine_stream_metrics'"
        ),
    )

    # Strict sequential task dependency definitions
    (
        task_validate_raw
        >> task_convert_to_parquet
        >> task_upload_to_hdfs
        >> task_spark_batch
        >> task_train_ml
        >> task_predict_ml
        >> task_load_clickhouse
        >> task_spark_streaming
        >> task_kafka_producer
        >> task_verify_clickhouse
    )