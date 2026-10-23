from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder
        .appName("AircraftHdfsSmokeTest")
        .master("spark://spark-master:7077")
        .getOrCreate()
    )

    path = (
        "hdfs://namenode:9000/"
        "aircraft/processed/"
        "N-CMAPSS_DS01-005/dev.parquet"
    )

    df = spark.read.parquet(path)

    print("=== Spark HDFS Smoke Test ===")
    print(f"Rows: {df.count():,}")
    print(f"Columns: {len(df.columns)}")
    print("Column names:")
    print(df.columns)

    print("Sample rows:")
    df.select(
        "unit",
        "cycle",
        "sensor_T24",
        "sensor_T30",
        "rul",
        "dataset",
        "split",
    ).show(5, truncate=False)

    spark.stop()


if __name__ == "__main__":
    main()