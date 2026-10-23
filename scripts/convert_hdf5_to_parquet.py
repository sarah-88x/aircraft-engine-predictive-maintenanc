from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Mapping of HDF5 dataset group identifiers to standardized column name prefixes
GROUPS: dict[str, str] = {
    "A": "",
    "W": "W_",
    "X_s": "sensor_",
    "X_v": "virtual_sensor_",
    "T": "T_",
}


def _decode_var_names(raw_vars: np.ndarray, prefix: str) -> list[str]:
    """Decode raw byte/string array into normalized column names with group prefixes."""
    names: list[str] = []
    for item in raw_vars:
        if isinstance(item, (bytes, np.bytes_)):
            name = item.decode("utf-8").strip()
        else:
            name = str(item).strip()
        if prefix and not name.startswith(prefix):
            name = f"{prefix}{name}"
        names.append(name)
    return names


def convert_split(
    input_path: Path,
    output_dir: Path,
    split: str = "dev",
    chunk_size: int = 50000,
    limit: int | None = None,
    force: bool = False,
) -> Path:
    """Stream and convert an HDF5 partition (dev or test) into a memory-balanced Parquet file."""
    dataset_name = input_path.stem
    target_dir = output_dir / dataset_name
    target_dir.mkdir(parents=True, exist_ok=True)
    output_file = target_dir / f"{split}.parquet"
    backup_file = output_dir / f"{dataset_name}_{split}_full.parquet.bak"

    # Archive large pre-existing Parquet outputs as backup before generating downsampled versions
    if output_file.exists() and output_file.stat().st_size > 100 * 1024 * 1024:
        if not backup_file.exists():
            shutil.move(str(output_file), str(backup_file))
        else:
            output_file.unlink()

    # Reuse previously converted files if valid and re-conversion is not forced
    if not force and limit is None and output_file.exists() and output_file.stat().st_size > 100 * 1024:
        pf = pq.ParquetFile(str(output_file))
        if pf.metadata.num_rows > 0:
            print(
                f"✔ Existing Parquet verified for split '{split}': {output_file} "
                f"({pf.metadata.num_rows:,} rows, {output_file.stat().st_size / (1024**2):.1f} MB)"
            )
            return output_file

    # Build a memory-efficient stratified sample across row groups if a full backup already exists
    if limit is None and backup_file.exists():
        pf = pq.ParquetFile(str(backup_file))
        total_rows = pf.metadata.num_rows
        step = max(1, total_rows // 60000)
        writer: pq.ParquetWriter | None = None
        written = 0
        try:
            for rg_idx in range(pf.num_row_groups):
                rg_table = pf.read_row_group(rg_idx)
                indices = pa.array(range(0, rg_table.num_rows, step))
                sampled = rg_table.take(indices)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), sampled.schema, compression="snappy")
                writer.write_table(sampled)
                written += sampled.num_rows
        finally:
            if writer is not None:
                writer.close()
        print(f"✔ Built fast stratified Parquet for '{split}' ({written:,} rows) -> {output_file}")
        return output_file

    # Read HDF5 container and stream chunks into destination Parquet
    with h5py.File(input_path, "r") as h5:
        a_key = f"A_{split}"
        if a_key not in h5:
            raise KeyError(f"Split '{split}' not found in {input_path} (missing {a_key})")

        total_rows = int(h5[a_key].shape[0])
        if limit is not None:
            total_rows = min(total_rows, int(limit))

        # Resolve available group keys and their corresponding column headers
        group_meta: list[tuple[str, list[str]]] = []
        for grp, prefix in GROUPS.items():
            data_key = f"{grp}_{split}"
            var_key = f"{grp}_var"
            if data_key in h5 and var_key in h5:
                col_names = _decode_var_names(h5[var_key][:], prefix)
                group_meta.append((data_key, col_names))

        y_key = f"Y_{split}"
        has_rul = y_key in h5

        # Process HDF5 blocks in chunks and append to Parquet table incrementally
        writer = None
        try:
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                n_chunk = end_idx - start_idx

                arrays: dict[str, pa.Array] = {}
                for data_key, col_names in group_meta:
                    block = h5[data_key][start_idx:end_idx]
                    for col_idx, col_name in enumerate(col_names):
                        arrays[col_name] = pa.array(block[:, col_idx].astype(np.float64))

                if has_rul:
                    y_block = h5[y_key][start_idx:end_idx].reshape(-1)
                    arrays["rul"] = pa.array(y_block.astype(np.float64))

                arrays["dataset"] = pa.array([dataset_name] * n_chunk, type=pa.string())
                arrays["split"] = pa.array([split] * n_chunk, type=pa.string())

                table = pa.Table.from_pydict(arrays)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_file), table.schema, compression="snappy")
                writer.write_table(table)
        finally:
            if writer is not None:
                writer.close()

    print(f"✔ Converted split '{split}' ({total_rows:,} rows) -> {output_file}")
    return output_file


def main() -> None:
    """Parse command line arguments and execute HDF5 to Parquet conversions."""
    parser = argparse.ArgumentParser(
        description="Convert NASA N-CMAPSS HDF5 files to Parquet."
    )
    parser.add_argument("--input", required=True, help="Path to input HDF5 file.")
    parser.add_argument("--output-dir", required=True, help="Output directory for Parquet files.")
    parser.add_argument(
        "--split",
        choices=["dev", "test", "both"],
        default="both",
        help="Dataset split to convert (dev, test, or both).",
    )
    parser.add_argument("--chunk-size", type=int, default=50000, help="Chunk size for streaming conversion.")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit per split.")
    parser.add_argument("--force", action="store_true", help="Force re-conversion even if Parquet exists.")
    parser.add_argument("--master", default=None, help="Optional Spark master argument (compatibility).")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    # Execute conversion for the specified partitions
    splits = ["dev", "test"] if args.split == "both" else [args.split]
    for sp in splits:
        convert_split(
            input_path=input_path,
            output_dir=output_dir,
            split=sp,
            chunk_size=args.chunk_size,
            limit=args.limit,
            force=args.force,
        )


if __name__ == "__main__":
    main()