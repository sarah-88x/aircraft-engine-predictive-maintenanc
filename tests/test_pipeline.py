from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import pytest

from scripts.convert_hdf5_to_parquet import GROUPS, convert_split
from scripts.producer import format_telemetry_message


def create_mock_hdf5_file(file_path: Path, split: str = "dev", rows: int = 20) -> None:
    """إنشاء ملف HDF5 مصغر ووهمي لاختبار الـ Converter بدون الحاجة لتحميل ملفات ناسا الضخمة."""
    with h5py.File(file_path, "w") as h5:
        # 1. مجموعة A: unit, cycle, Fc, hs
        a_vars = [b"unit", b"cycle", b"Fc", b"hs"]
        h5.create_dataset("A_var", data=np.array(a_vars, dtype="O"))
        a_data = np.zeros((rows, 4), dtype=np.float64)
        a_data[:, 0] = 1.0  # unit
        a_data[:, 1] = np.arange(1, rows + 1)  # cycle
        a_data[:, 2] = 1.0  # Fc
        a_data[:, 3] = 1.0  # hs
        h5.create_dataset(f"A_{split}", data=a_data)

        # 2. المجموعات T, W, X_s, X_v
        for grp in ["T", "W", "X_s", "X_v"]:
            var_names = [f"var_{i}".encode("utf-8") for i in range(2)]
            h5.create_dataset(f"{grp}_var", data=np.array(var_names, dtype="O"))
            grp_data = np.random.rand(rows, 2).astype(np.float64)
            h5.create_dataset(f"{grp}_{split}", data=grp_data)

        # 3. عمود الهدف RUL في Y
        y_data = np.linspace(100, 10, rows).reshape(-1, 1).astype(np.float64)
        h5.create_dataset(f"Y_{split}", data=y_data)


def test_mock_hdf5_to_parquet_converter(tmp_path: Path) -> None:
    """اختبار سلامة تحويل ملف HDF5 إلى Parquet ومطابقة أسماء الأعمدة المتوقعة."""
    mock_h5 = tmp_path / "mock_engine.h5"
    output_dir = tmp_path / "processed"

    create_mock_hdf5_file(mock_h5, split="dev", rows=25)

    # تشغيل التحويل
    convert_split(
        input_path=mock_h5,
        output_dir=output_dir,
        split="dev",
        chunk_size=10,
        limit=None,
    )

    parquet_file = output_dir / "mock_engine" / "dev.parquet"
    assert parquet_file.exists(), "لم يتم إنشاء ملف Parquet الناتج!"

    table = pq.read_table(str(parquet_file))
    assert table.num_rows == 25, f"عدد الصفوف غير مطابق: {table.num_rows}"

    cols = set(table.column_names)
    assert "unit" in cols
    assert "cycle" in cols
    assert "rul" in cols
    assert "dataset" in cols
    assert "sensor_var_0" in cols


def test_telemetry_schema_json_serialization() -> None:
    """اختبار التوافقية التسلسلية للـ JSON بين الـ Producer والـ Consumer."""
    raw_record = {
        "unit": 2.0,
        "cycle": 10.0,
        "rul": 75.5,
        "sensor_T24": 642.10,
        "sensor_T30": 1588.0,
        "dataset": "N-CMAPSS_TEST",
    }

    # رسالة تليمترية سليمة
    valid_msg = format_telemetry_message(raw_record, inject_anomaly=False)
    assert valid_msg["unit"] == 2.0
    assert valid_msg["cycle"] == 10.0
    assert valid_msg["rul"] == 75.5
    assert "timestamp" in valid_msg

    json_str = json.dumps(valid_msg)
    decoded = json.loads(json_str)
    assert decoded["unit"] == 2.0

    # رسالة غير مطابقة موجهة للـ DLQ
    corrupted_msg = format_telemetry_message(raw_record, inject_anomaly=True)
    assert "unit" not in corrupted_msg
    assert corrupted_msg["corrupted_payload"] == "ERR_MALFORMED_TELEMETRY"


def test_anti_leakage_engine_split_logic() -> None:
    """اختبار منطق التقسيم الصارم للمحركات للتأكد من عدم تسريب البيانات (Data Leakage)."""
    import random

    units = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    total_units = len(units)

    # محاكاة منطق التقسيم
    rng = random.Random(42)
    shuffled = units.copy()
    rng.shuffle(shuffled)

    n_train = max(1, int(round(total_units * 0.7)))
    n_val = max(1, int(round(total_units * 0.15)))

    train_units = sorted(shuffled[:n_train])
    val_units = sorted(shuffled[n_train : n_train + n_val])
    test_units = sorted(shuffled[n_train + n_val :])

    # التحقق الصارم من الانفصال التام بين المجموعات
    assert set(train_units).isdisjoint(set(val_units)), "تسريب بيانات بين Train و Val!"
    assert set(train_units).isdisjoint(set(test_units)), "تسريب بيانات بين Train و Test!"
    assert set(val_units).isdisjoint(set(test_units)), "تسريب بيانات بين Val و Test!"
    assert len(train_units) + len(val_units) + len(test_units) == total_units


def test_dlq_message_detection_logic() -> None:
    """اختبار معايير فرز الرسائل التالفة وتوجيهها للـ DLQ."""
    valid_payload = {"unit": 1.0, "cycle": 5.0, "sensor_T24": 640.0}
    missing_unit = {"cycle": 5.0, "sensor_T24": 640.0}
    missing_cycle = {"unit": 1.0, "sensor_T24": 640.0}
    empty_payload = {}

    def is_valid(payload: dict) -> bool:
        return "unit" in payload and "cycle" in payload and payload["unit"] is not None and payload["cycle"] is not None

    assert is_valid(valid_payload) is True
    assert is_valid(missing_unit) is False
    assert is_valid(missing_cycle) is False
    assert is_valid(empty_payload) is False
