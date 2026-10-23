# ✈️ End-to-End Aircraft Engine Predictive Maintenance Big Data Platform

[![Big Data Architecture](https://img.shields.io/badge/Architecture-Distributed%20Batch%20%26%20Streaming-blue)](https://github.com/sarah-88x/aircraft-engine-predictive-maintenanc)
[![Apache Spark](https://img.shields.io/badge/Spark-v4.1.3-orange?logo=apache-spark)](https://spark.apache.org/)
[![Apache Airflow](https://img.shields.io/badge/Airflow-v2.9.1-red?logo=apache-airflow)](https://airflow.apache.org/)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-v26.9-yellow)](https://clickhouse.com/)
[![Apache Kafka](https://img.shields.io/badge/Kafka-KRaft-black?logo=apache-kafka)](https://kafka.apache.org/)

An enterprise-grade, end-to-end Big Data and Machine Learning pipeline designed to predict turbofan engine failures (**Remaining Useful Life - RUL**) and monitor real-time telemetry anomalies using NASA’s N-CMAPSS dataset.

---

## 🏗️ Architectural Overview & Data Flow

The platform implements a **Hybrid Lambda/Kappa Architecture** separating heavy historical batch ML training from low-latency real-time stream ingestion:

1. **Data Source:** NASA N-CMAPSS (`DS01-005` & `DS03-012` HDF5 files totaling **17.46M+ rows / 6.12 GB**).
2. **Storage & Data Lake:** Apache Hadoop HDFS (`v3.2.1`) storing Snappy-compressed Apache Parquet datasets and Spark MLlib PipelineModels.
3. **Compute Engine:** Apache Spark (`v4.1.3`) for distributed Batch ETL, Spark MLlib predictive regression, and Spark Structured Streaming.
4. **Real-Time Telemetry Ingestion:** Apache Kafka (`KRaft v4.0`) handling high-frequency sensor replay and Dead-Letter Queue (DLQ) quarantine.
5. **Workflow Orchestration:** Apache Airflow (`v2.9.1`) managing the automated 10-stage DAG (`aircraft_engine_batch_pipeline`).
6. **Analytical OLAP Serving & BI:** ClickHouse columnar database (`v26.9`) serving sub-second queries to Apache Superset (`v3.1.0`) dashboards.

---

## 📂 Project Repository Structure

```text
bigdata-aircraft/
├── airflow/
│   └── dags/
│       └── aircraft_engine_batch_pipeline.py   # Main 10-task Airflow orchestrator DAG
├── compose/
│   ├── docker-compose.yml                      # Modular active multi-container runtime
│   └── docker-stack-swarm.yml                  # 3-Node Docker Swarm + Tailscale mesh config
├── scripts/
│   ├── convert_hdf5_to_parquet.py            # Memory-safe chunked H5 to Parquet converter
│   ├── spark_batch_summary.py                # Distributed Spark batch aggregation
│   ├── ml_train.py                           # Leakage-free Spark MLlib model training (LR, RF, GBT)
│   ├── ml_predict.py                         # Large-scale test inference on unseen engine units
│   ├── load_to_clickhouse.py                 # Batch OLAP ingestion runner
│   ├── streaming_job.py                      # Spark Structured Streaming engine (30s windowed agg)
│   └── producer.py                           # Kafka historical telemetry replay producer with DLQ
├── jars/                                       # Spark-Kafka connector & client dependencies
├── pylibs/                                     # Local Python fallback libraries (h5py, pyarrow, numpy)
├── tests/
│   └── test_pipeline.py                        # Pytest suite for schema validation & anti-leakage splits
└── .github/
    └── workflows/
        └── ci.yml                              # GitHub Actions CI linting & testing workflow