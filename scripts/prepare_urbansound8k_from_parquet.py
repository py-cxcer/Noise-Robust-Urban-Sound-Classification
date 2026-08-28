"""Reconstruct the UrbanSound8K WAV layout from Hugging Face Parquet shards."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import pyarrow.parquet as parquet
from tqdm import tqdm


EXPECTED_SHARDS = 16
EXPECTED_SAMPLES = 8_732
DEFAULT_SOURCE = Path("data/raw/UrbanSound8K_parquet")
DEFAULT_OUTPUT = Path("data/raw/UrbanSound8K")


def _metadata_inventory(metadata_path: Path) -> dict[str, int]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Metadata contains {len(rows):,} rows; expected {EXPECTED_SAMPLES:,}."
        )

    inventory: dict[str, int] = {}
    for row in rows:
        filename = row["slice_file_name"]
        fold = int(row["fold"])
        if filename in inventory:
            raise RuntimeError(f"Duplicate metadata filename: {filename}")
        inventory[filename] = fold
    return inventory


def _write_audio_file(destination: Path, payload: bytes) -> bool:
    if destination.exists():
        if destination.read_bytes() != payload:
            raise RuntimeError(f"Existing audio differs from source: {destination}")
        return False

    partial = destination.with_suffix(destination.suffix + ".partial")
    partial.write_bytes(payload)
    partial.replace(destination)
    return True


def prepare_dataset(source: Path, output: Path) -> tuple[int, int]:
    metadata_source = source / "UrbanSound8K.csv"
    shards = sorted((source / "data").glob("train-*-of-*.parquet"))
    if not metadata_source.is_file():
        raise FileNotFoundError(f"Missing metadata file: {metadata_source}")
    if len(shards) != EXPECTED_SHARDS:
        raise RuntimeError(
            f"Found {len(shards)} Parquet shards; expected {EXPECTED_SHARDS}. "
            "Resume the snapshot download before preparing the dataset."
        )

    metadata = _metadata_inventory(metadata_source)
    audio_root = output / "audio"
    metadata_root = output / "metadata"
    for fold in range(1, 11):
        (audio_root / f"fold{fold}").mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    extracted: dict[str, int] = {}
    written = 0
    with tqdm(total=EXPECTED_SAMPLES, unit="file", desc="Reconstructing WAV files") as progress:
        for shard in shards:
            parquet_file = parquet.ParquetFile(shard)
            for batch in parquet_file.iter_batches(
                batch_size=32,
                columns=["audio", "slice_file_name", "fold"],
            ):
                for row in batch.to_pylist():
                    filename = row["slice_file_name"]
                    fold = int(row["fold"])
                    audio = row["audio"]
                    payload = audio["bytes"]

                    if Path(filename).name != filename or Path(filename).suffix.lower() != ".wav":
                        raise RuntimeError(f"Unsafe or invalid audio filename: {filename!r}")
                    if audio["path"] != filename:
                        raise RuntimeError(f"Audio path does not match metadata: {filename}")
                    if not (1 <= fold <= 10):
                        raise RuntimeError(f"Invalid fold {fold} for {filename}")
                    if filename in extracted:
                        raise RuntimeError(f"Duplicate audio filename across shards: {filename}")
                    if metadata.get(filename) != fold:
                        raise RuntimeError(f"Parquet and CSV fold disagree for {filename}")
                    if not payload.startswith(b"RIFF") or payload[8:12] != b"WAVE":
                        raise RuntimeError(f"Embedded payload is not a RIFF/WAVE file: {filename}")

                    destination = audio_root / f"fold{fold}" / filename
                    written += int(_write_audio_file(destination, payload))
                    extracted[filename] = fold
                    progress.update(1)

    if len(extracted) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Extracted {len(extracted):,} unique files; expected {EXPECTED_SAMPLES:,}."
        )
    if extracted != metadata:
        missing = sorted(set(metadata) - set(extracted))
        extra = sorted(set(extracted) - set(metadata))
        raise RuntimeError(
            f"Audio/metadata inventory mismatch; missing={missing[:5]}, extra={extra[:5]}."
        )

    shutil.copy2(metadata_source, metadata_root / "UrbanSound8K.csv")
    readme_source = source / "README.md"
    if readme_source.is_file():
        shutil.copy2(readme_source, output / "MIRROR_README.md")

    provenance = output / "SOURCE.txt"
    provenance.write_text(
        "UrbanSound8K audio mirror: https://huggingface.co/datasets/danavery/urbansound8K\n"
        "Official release and attribution: https://zenodo.org/records/1203745\n"
        "The embedded WAV bytes and official fold metadata were reconstructed without "
        "audio decoding or resampling.\n",
        encoding="utf-8",
    )
    return len(extracted), written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    total, created = prepare_dataset(arguments.source, arguments.output)
    print(f"Prepared {total:,} verified audio files ({created:,} newly written).")
