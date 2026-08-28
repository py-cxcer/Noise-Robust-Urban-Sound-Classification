from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "prepare_urbansound8k_from_parquet.py"
SPEC = importlib.util.spec_from_file_location("prepare_urbansound8k", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


def _wav_payload(marker: bytes) -> bytes:
    return b"RIFF" + marker.ljust(4, b"0")[:4] + b"WAVE" + b"data"


def _write_fixture(source: Path) -> None:
    (source / "data").mkdir(parents=True)
    rows = [
        ("100-0-0-0.wav", 1, _wav_payload(b"one")),
        ("200-1-0-0.wav", 2, _wav_payload(b"two")),
    ]
    with (source / "UrbanSound8K.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice_file_name", "fold"])
        writer.writeheader()
        for filename, fold, _ in rows:
            writer.writerow({"slice_file_name": filename, "fold": fold})

    for index, (filename, fold, payload) in enumerate(rows):
        table = pa.table(
            {
                "audio": pa.array(
                    [{"bytes": payload, "path": filename}],
                    type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
                ),
                "slice_file_name": [filename],
                "fold": [fold],
            }
        )
        pq.write_table(table, source / "data" / f"train-{index:05d}-of-00002.parquet")


def test_prepare_dataset_reconstructs_and_resumes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_fixture(source)
    monkeypatch.setattr(prepare, "EXPECTED_SHARDS", 2)
    monkeypatch.setattr(prepare, "EXPECTED_SAMPLES", 2)

    assert prepare.prepare_dataset(source, output) == (2, 2)
    assert prepare.prepare_dataset(source, output) == (2, 0)
    assert (output / "audio/fold1/100-0-0-0.wav").read_bytes() == _wav_payload(b"one")
    assert (output / "audio/fold2/200-1-0-0.wav").read_bytes() == _wav_payload(b"two")
    assert (output / "metadata/UrbanSound8K.csv").is_file()
    assert (output / "SOURCE.txt").is_file()
