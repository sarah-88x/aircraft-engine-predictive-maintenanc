-- ==============================================================================
-- Aircraft Predictive Maintenance: ClickHouse Analytics Schema
-- Fully compatible with Spark MLlib and Spark Streaming pipeline outputs
-- ==============================================================================

-- Create the dedicated analytics database if it does not already exist
CREATE DATABASE IF NOT EXISTS aircraft;

-- 1. Batch aggregation table (captures aggregated metrics from spark_batch_summary.py)
CREATE TABLE IF NOT EXISTS aircraft.engine_batch_summary (
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
) ENGINE = ReplacingMergeTree(created_at)
ORDER BY (dataset, unit);

-- 2. ML model inferences and Remaining Useful Life (RUL) predictions table (outputs from ml_predict.py)
CREATE TABLE IF NOT EXISTS aircraft.engine_rul_predictions (
    dataset LowCardinality(String),
    split LowCardinality(String),
    unit UInt32,
    cycle UInt32,
    actual_rul Nullable(Float32),
    predicted_rul Float32,
    absolute_error Nullable(Float32),
    predicted_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(predicted_at)
ORDER BY (dataset, unit, cycle, predicted_at);

-- 3. Critical maintenance alerts table for condition-based thresholds and diagnostics
CREATE TABLE IF NOT EXISTS aircraft.maintenance_alerts (
    alert_time DateTime DEFAULT now(),
    unit UInt32,
    severity LowCardinality(String),
    alert_type LowCardinality(String),
    max_sensor_T30 Float32,
    min_rul Float32,
    diagnosis String
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(alert_time)
ORDER BY (unit, alert_time);

-- 4. Real-time windowed streaming metrics table (outputs from streaming_job.py)
CREATE TABLE IF NOT EXISTS aircraft.engine_stream_metrics (
    window_start DateTime,
    window_end DateTime,
    unit UInt32,
    reading_count UInt32,
    avg_sensor_T24 Float32,
    avg_sensor_T30 Float32,
    max_sensor_T30 Float32,
    min_rul Float32,
    created_at DateTime DEFAULT now()
) ENGINE = MergeTree()
PARTITION BY toYYYYMM(window_start)
ORDER BY (unit, window_start);