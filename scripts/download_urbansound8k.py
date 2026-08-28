"""Download and validate the public UrbanSound8K Kaggle archive.

This downloader uses independent HTTP byte ranges so an interrupted multi-gigabyte
transfer can resume without discarding completed work.  It deliberately keeps
download parts until the assembled ZIP has passed integrity validation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import shutil
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import zipfile

from kagglehub.clients import build_kaggle_client
from kagglehub.handle import parse_dataset_handle
from kagglehub.http_resolver import (
    _build_dataset_download_request,
    _get_current_version,
)
from tqdm import tqdm


DATASET_HANDLE = "chrisfilo/urbansound8k"
EXPECTED_ARCHIVE_SIZE = 6_026_232_524
DEFAULT_PARTS_DIRECTORY = Path("data/raw/UrbanSound8K_parts")
DEFAULT_ARCHIVE_PATH = Path("data/raw/UrbanSound8K_kaggle.zip")
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ByteRange:
    index: int
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def _signed_archive_url() -> str:
    handle = parse_dataset_handle(DATASET_HANDLE)
    with build_kaggle_client() as client:
        if not handle.is_versioned():
            handle = handle.with_version(_get_current_version(client, handle))
        request = _build_dataset_download_request(handle, None)
        response = client.datasets.dataset_api_client.download_dataset(request)
    return response.url


def _make_ranges(total_size: int, workers: int, first_part_size: int) -> list[ByteRange]:
    if first_part_size < 0 or first_part_size > total_size:
        raise ValueError("The existing prefix is larger than the expected archive.")

    ranges: list[ByteRange] = []
    if first_part_size:
        ranges.append(ByteRange(index=0, start=0, end=first_part_size - 1))

    remaining_start = first_part_size
    remaining = total_size - remaining_start
    if not remaining:
        return ranges

    range_count = min(workers, remaining)
    base_size, extra = divmod(remaining, range_count)
    cursor = remaining_start
    index_offset = 1 if first_part_size else 0
    for offset in range(range_count):
        size = base_size + (1 if offset < extra else 0)
        ranges.append(
            ByteRange(
                index=index_offset + offset,
                start=cursor,
                end=cursor + size - 1,
            )
        )
        cursor += size
    return ranges


def _part_path(parts_directory: Path, byte_range: ByteRange) -> Path:
    return parts_directory / f"part_{byte_range.index:05d}.bin"


def _download_range(
    url: str,
    byte_range: ByteRange,
    destination: Path,
    progress: tqdm,
    progress_lock: threading.Lock,
    retries: int,
) -> None:
    existing_size = destination.stat().st_size if destination.exists() else 0
    if existing_size > byte_range.size:
        raise RuntimeError(f"{destination} is larger than its assigned byte range.")
    if existing_size == byte_range.size:
        return

    request_start = byte_range.start + existing_size
    for attempt in range(1, retries + 1):
        try:
            request = Request(
                url,
                headers={
                    "Range": f"bytes={request_start}-{byte_range.end}",
                    "User-Agent": "UrbanSound8K-research-downloader/1.0",
                },
            )
            with urlopen(request, timeout=120) as response:
                if response.status != 206:
                    raise RuntimeError(
                        f"Expected HTTP 206 for range download, received {response.status}."
                    )
                expected_content_range = (
                    f"bytes {request_start}-{byte_range.end}/{EXPECTED_ARCHIVE_SIZE}"
                )
                if response.headers.get("Content-Range") != expected_content_range:
                    raise RuntimeError(
                        "Unexpected Content-Range: "
                        f"{response.headers.get('Content-Range')!r}; "
                        f"expected {expected_content_range!r}."
                    )

                with destination.open("ab") as output:
                    while chunk := response.read(CHUNK_SIZE):
                        output.write(chunk)
                        with progress_lock:
                            progress.update(len(chunk))
            break
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            if attempt == retries:
                raise RuntimeError(
                    f"Range {byte_range.index} failed after {retries} attempts: {error}"
                ) from error
            time.sleep(min(2**attempt, 15))
            existing_size = destination.stat().st_size if destination.exists() else 0
            request_start = byte_range.start + existing_size

    actual_size = destination.stat().st_size
    if actual_size != byte_range.size:
        raise RuntimeError(
            f"{destination} has {actual_size} bytes; expected {byte_range.size}."
        )


def _assemble_archive(ranges: list[ByteRange], parts_directory: Path, archive_path: Path) -> None:
    partial_archive = archive_path.with_suffix(archive_path.suffix + ".partial")
    with partial_archive.open("wb") as output:
        for byte_range in ranges:
            part = _part_path(parts_directory, byte_range)
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)

    if partial_archive.stat().st_size != EXPECTED_ARCHIVE_SIZE:
        raise RuntimeError("Assembled archive has an unexpected size.")

    print("Validating ZIP structure and CRC checks (this can take several minutes)...")
    with zipfile.ZipFile(partial_archive) as archive:
        bad_member = archive.testzip()
        if bad_member is not None:
            raise RuntimeError(f"ZIP CRC validation failed for {bad_member!r}.")
        if not archive.namelist():
            raise RuntimeError("The downloaded ZIP is empty.")

    partial_archive.replace(archive_path)


def download(workers: int, parts_directory: Path, archive_path: Path) -> Path:
    parts_directory.mkdir(parents=True, exist_ok=True)
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    prefix = parts_directory / "part_00000.bin"
    prefix_size = prefix.stat().st_size if prefix.exists() else 0
    ranges = _make_ranges(EXPECTED_ARCHIVE_SIZE, workers, prefix_size)
    completed_bytes = sum(
        min(_part_path(parts_directory, item).stat().st_size, item.size)
        if _part_path(parts_directory, item).exists()
        else 0
        for item in ranges
    )

    print(f"Resuming at {completed_bytes:,} of {EXPECTED_ARCHIVE_SIZE:,} bytes.")
    url = _signed_archive_url()
    progress_lock = threading.Lock()
    with tqdm(
        total=EXPECTED_ARCHIVE_SIZE,
        initial=completed_bytes,
        unit="B",
        unit_scale=True,
        desc="UrbanSound8K",
    ) as progress:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _download_range,
                    url,
                    item,
                    _part_path(parts_directory, item),
                    progress,
                    progress_lock,
                    5,
                )
                for item in ranges
                if not _part_path(parts_directory, item).exists()
                or _part_path(parts_directory, item).stat().st_size != item.size
            ]
            for future in as_completed(futures):
                future.result()

    _assemble_archive(ranges, parts_directory, archive_path)
    print(f"Validated archive: {archive_path.resolve()}")
    return archive_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parts-directory", type=Path, default=DEFAULT_PARTS_DIRECTORY)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    download(arguments.workers, arguments.parts_directory, arguments.archive)
