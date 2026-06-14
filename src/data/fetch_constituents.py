"""Fetch point-in-time S&P 500 constituent membership and pin a dated snapshot.

Source: https://github.com/fja05680/sp500
File:   "S&P 500 Historical Components & Changes (Updated).csv"

The source file provides, for each date, the full set of tickers in the
S&P 500 on that date. Tickers that have left the index carry a "-YYYYMM"
suffix encoding the month they departed (e.g. "AAMRQ-201312"). This raw
download is kept BYTE-FAITHFUL to the source -- no cleaning is done here.
Suffix stripping and parsing happen in a separate processing step so the
raw -> processed boundary stays clean and auditable.

Reproducibility note: we PIN a dated snapshot rather than re-fetching the
latest each run, because the index membership tail changes every few
months. Pinning guarantees the numbers in the paper stay reproducible.

We download via the codeload tarball endpoint rather than raw.githubusercontent
because the latter intermittently 404s on filenames containing spaces,
ampersands, and parentheses.
"""

from __future__ import annotations

import datetime as dt
import io
import tarfile
import urllib.request
from pathlib import Path

# --- Configuration -----------------------------------------------------------

REPO_TARBALL_URL = "https://codeload.github.com/fja05680/sp500/tar.gz/refs/heads/master"

# Path of the target file *inside* the tarball (top-level dir is "sp500-master").
MEMBER_PATH = "sp500-master/S&P 500 Historical Components & Changes (Updated).csv"

# Resolve project root as two levels up from this file: src/data/<this>.py -> root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def fetch_tarball(url: str) -> bytes:
    """Download the repo tarball and return its raw bytes."""
    print(f"Downloading repo tarball:\n  {url}")
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = resp.read()
    print(f"  received {len(data):,} bytes")
    return data


def extract_member(tarball_bytes: bytes, member_path: str) -> bytes:
    """Extract a single file's bytes from the in-memory tarball."""
    with tarfile.open(fileobj=io.BytesIO(tarball_bytes), mode="r:gz") as tar:
        try:
            member = tar.getmember(member_path)
        except KeyError:
            names = [m.name for m in tar.getmembers() if m.name.endswith(".csv")]
            raise SystemExit(
                f"Expected file not found in tarball:\n  {member_path}\n"
                f"Available CSVs:\n  " + "\n  ".join(names)
            )
        extracted = tar.extractfile(member)
        if extracted is None:
            raise SystemExit(f"Could not read file object for: {member_path}")
        return extracted.read()


def summarize(csv_bytes: bytes) -> tuple[str, str, int]:
    """Return (first_date, last_date, n_rows) without a pandas dependency."""
    text = csv_bytes.decode("utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    data_lines = lines[1:]  # drop header
    first_date = data_lines[0].split(",", 1)[0]
    last_date = data_lines[-1].split(",", 1)[0]
    return first_date, last_date, len(data_lines)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    tarball = fetch_tarball(REPO_TARBALL_URL)
    csv_bytes = extract_member(tarball, MEMBER_PATH)

    first_date, last_date, n_rows = summarize(csv_bytes)

    today = dt.date.today().isoformat()
    out_path = RAW_DIR / f"sp500_constituents_raw_{today}.csv"
    out_path.write_bytes(csv_bytes)

    print("\nPinned snapshot written:")
    print(f"  path        : {out_path.relative_to(PROJECT_ROOT)}")
    print(f"  size        : {len(csv_bytes):,} bytes")
    print(f"  date range  : {first_date}  ->  {last_date}")
    print(f"  rows (dates): {n_rows:,}")
    print(f"  fetched on  : {today}")
    print(
        "\nNext: force-add this file to git so the pinned snapshot is in the repo:\n"
        f"  git add -f {out_path.relative_to(PROJECT_ROOT)}"
    )


if __name__ == "__main__":
    main()