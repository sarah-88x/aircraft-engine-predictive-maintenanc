from __future__ import annotations

import argparse
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaError = Exception
import pyarrow.parquet as pq


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("TelemetryProducer")


DEFAULT_LOCAL_PARQUET = (
    "data/processed/N-CMAPSS_DS01-005/test.parquet"
)


def create_kafka_producer(
    bootstrap_servers: str,
    retries: int = 5,
    delay: float = 3.0,
) -> KafkaProducer:
    """Establish connection to Kafka with retry logic upon startup."""
    if KafkaProducer is None:
        raise ImportError("kafka-python is required to run the Kafka producer. Install it via pip install kafka-python.")

    for attempt in range(1, retries + 1):
        try:
            logger.info(
                f"Connecting to Kafka on {bootstrap_servers} (Attempt {attempt}/{retries})..."
            )
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers.split(","),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: str(k).encode("utf-8") if k else None,
                acks=1,
                request_timeout_ms=10000,
            )
            logger.info("✔ Connected to Kafka successfully.")
            return producer
        except Exception as e:
            if attempt == retries:
                raise e
            logger.warning(f"Kafka not ready yet ({e}), retrying in {delay}s...")
            time.sleep(delay)
    raise RuntimeError("Failed to connect to Kafka brokers.")


def load_parquet_records(parquet_path: str, limit: int | None = None) -> list[dict]:
    """Read Parquet data and convert it into a list of dictionaries for streaming."""
    path = Path(parquet_path)
    if not path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    logger.info(f"Loading telemetry from {parquet_path}...")
    if limit:
        # Read only the first `limit` rows to avoid loading the entire Parquet file into memory
        pf = pq.ParquetFile(str(path))
        batches = next(pf.iter_batches(batch_size=limit))
        import pyarrow as pa

        table = pa.Table.from_batches([batches])
    else:
        table = pq.read_table(str(path))

    pydict = table.to_pydict()
    num_rows = table.num_rows

    records = []
    keys = list(pydict.keys())

    for i in range(num_rows):
        row = {k: pydict[k][i] for k in keys}
        records.append(row)

    logger.info(f"✔ Loaded {len(records):,} records ready for streaming.")
    return records


def format_telemetry_message(raw_record: dict, inject_anomaly: bool = False) -> dict:
    """
    Prepare a JSON-compatible telemetry message with an instantaneous timestamp added.
    """
    if inject_anomaly:
        # Intentionally corrupted/malformed message to test the Dead-Letter Queue (DLQ)
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "corrupted_payload": "ERR_MALFORMED_TELEMETRY",
            "anomaly_flag": True,
        }

    # Valid telemetry message with an instantaneous timestamp
    message = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "unit": float(raw_record.get("unit", 0.0)),
        "cycle": float(raw_record.get("cycle", 0.0)),
        "dataset": str(raw_record.get("dataset", "N-CMAPSS")),
    }

    # Append target column if available
    if "rul" in raw_record and raw_record["rul"] is not None:
        message["rul"] = float(raw_record["rul"])

    # Append all sensor readings and operational variables
    for k, v in raw_record.items():
        if k not in {"unit", "cycle", "rul", "dataset", "split", "source_row"}:
            if v is not None:
                message[k] = float(v)

    return message


def stream_telemetry(
    producer: KafkaProducer,
    records: list[dict],
    topic: str = "engine-telemetry",
    rate_per_sec: float = 20.0,
    batch_size: int = 5,
    loop: bool = True,
    max_messages: int | None = None,
    inject_anomaly_rate: float = 0.0,
) -> None:
    """Stream telemetry data at a steady, controlled rate with looping and batch support."""
    delay_between_batches = batch_size / rate_per_sec if rate_per_sec > 0 else 0.05
    total_sent = 0
    start_time = time.time()

    logger.info(f"[*] Starting telemetry stream to topic '{topic}'...")
    logger.info(f"    Rate: {rate_per_sec} msgs/sec | Batch size: {batch_size} | Loop: {loop}")

    try:
        while True:
            for idx in range(0, len(records), batch_size):
                batch_records = records[idx : idx + batch_size]

                for record in batch_records:
                    should_inject = (
                        inject_anomaly_rate > 0
                        and random.random() < inject_anomaly_rate
                    )
                    payload = format_telemetry_message(record, inject_anomaly=should_inject)
                    unit_key = str(payload.get("unit", "unknown"))

                    producer.send(
                        topic=topic,
                        key=unit_key,
                        value=payload,
                    )
                    total_sent += 1

                    if max_messages and total_sent >= max_messages:
                        producer.flush()
                        logger.info(f"✔ Reached max messages limit ({max_messages}). Stream finished.")
                        return

                producer.flush()
                time.sleep(delay_between_batches)

                if total_sent % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_rate = total_sent / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Sent {total_sent:,} msgs | Actual rate: {actual_rate:.1f} msg/s"
                    )

            if not loop:
                break
            logger.info("Reached end of Parquet dataset. Looping from start...")

    except KeyboardInterrupt:
        logger.info("Stream interrupted by user (Ctrl+C). Flushing messages...")
    finally:
        producer.flush()
        producer.close()
        elapsed = time.time() - start_time
        logger.info(f"Producer stopped. Total sent: {total_sent:,} messages in {elapsed:.1f}s.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream Aircraft Engine Telemetry to Kafka topic in real-time."
    )
    parser.add_argument(
        "--input",
        default=DEFAULT_LOCAL_PARQUET,
        help="Path to Parquet file to stream.",
    )
    parser.add_argument(
        "--bootstrap-servers",
        default="kafka:9092",
        help="Kafka bootstrap servers (e.g. kafka:9092 or localhost:9092).",
    )
    parser.add_argument(
        "--topic",
        default="engine-telemetry",
        help="Target Kafka topic.",
    )
    parser.add_argument(
        "--rate-per-sec",
        type=float,
        default=25.0,
        help="Messages to send per second.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of messages sent per batch.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        default=True,
        help="Loop continuously when the dataset reaches the end.",
    )
    parser.add_argument(
        "--no-loop",
        dest="loop",
        action="store_false",
        help="Do not loop when the dataset ends.",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop after sending N messages.",
    )
    parser.add_argument(
        "--inject-anomaly-rate",
        type=float,
        default=0.02,
        help="Fraction of corrupted messages to inject for DLQ testing (e.g. 0.02 for 2%).",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Load only the first N rows from the Parquet file (memory-safe replay).",
    )

    args = parser.parse_args()

    producer = create_kafka_producer(args.bootstrap_servers)
    records = load_parquet_records(args.input, limit=args.limit)

    stream_telemetry(
        producer=producer,
        records=records,
        topic=args.topic,
        rate_per_sec=args.rate_per_sec,
        batch_size=args.batch_size,
        loop=args.loop,
        max_messages=args.max_messages,
        inject_anomaly_rate=args.inject_anomaly_rate,
    )


if __name__ == "__main__":
    main()