"""Build a clean, long-format S&P 500 membership table from the pinned raw snapshot.

Input  : data/raw/sp500_constituents_raw_<DATE>.csv   (pinned, byte-faithful source)
Output : data/processed/sp500_membership_long.csv      (date, ticker) one row per pair

Cleaning performed here (and ONLY here -- raw stays untouched):

  1. Yahoo ticker normalization: the source encodes share classes with a dot
     (e.g. "BRK.B"), but Yahoo Finance / yfinance expects a dash ("BRK-B").
     We convert "." -> "-" so downstream price fetching actually finds them.
     Without this, names like Berkshire Hathaway silently vanish from the
     price panel with no error -- a classic source of corrupted backtests.

  2. Parsing the comma-delimited ticker string into tidy long format, which
     is the natural primitive for filter/join operations against price data.

Note on suffixes: the original fja05680 file tagged departed tickers with a
"-YYYYMM" suffix. The maintained "(Updated)" file we pin has ALREADY stripped
these, so no suffix handling is needed. A ticker leaving the index is captured
simply by its absence from later dates -- which is all GGR's methodology needs.

The long table answers the core backtest question -- "who was in the index on
date X?" -- via a simple filter, and keeps survivorship bias auditable (you can
count membership per date, plot index size over time, and see exactly which
delisted names are retained).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def normalize_ticker(raw: str) -> str:
    """Convert a source ticker to Yahoo Finance convention.

    Share classes: source uses '.', Yahoo uses '-'  (BRK.B -> BRK-B).
    """
    return raw.strip().replace(".", "-")


def build_long(raw_path: Path) -> list[tuple[str, str]]:
    """Parse the raw snapshot into a sorted list of (date, ticker) pairs."""
    with raw_path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["date", "tickers"]:
            raise SystemExit(f"Unexpected header in {raw_path.name}: {header!r}")

        long_rows: list[tuple[str, str]] = []
        for date, tickers in reader:
            for tok in tickers.split(","):
                tok = tok.strip()
                if tok:
                    long_rows.append((date, normalize_ticker(tok)))
    return long_rows


def validate(long_rows: list[tuple[str, str]]) -> None:
    """Print sanity statistics that double as a survivorship-bias audit."""
    per_date: dict[str, int] = defaultdict(int)
    for d, _ in long_rows:
        per_date[d] += 1
    dates = sorted(per_date)
    tickers_ever = {t for _, t in long_rows}

    print("Validation:")
    print(f"  date range        : {dates[0]}  ->  {dates[-1]}")
    print(f"  distinct dates    : {len(dates):,}")
    print(f"  (date,ticker) rows: {len(long_rows):,}")
    print(f"  membership min/max: {min(per_date.values())} / {max(per_date.values())}")
    print(f"  unique tickers ever in index: {len(tickers_ever):,}")
    print(
        "  (that last number minus ~500 current names is the survivorship gap "
        "a naive study would miss)"
    )


def find_default_raw() -> Path:
    """Pick the pinned raw snapshot. Errors if zero or many are present."""
    candidates = sorted(RAW_DIR.glob("sp500_constituents_raw_*.csv"))
    if not candidates:
        raise SystemExit(
            f"No pinned snapshot found in {RAW_DIR}. "
            "Run src/data/fetch_constituents.py first."
        )
    if len(candidates) > 1:
        names = "\n  ".join(c.name for c in candidates)
        raise SystemExit(
            "Multiple pinned snapshots found -- pass one explicitly with --raw "
            f"to stay reproducible:\n  {names}"
        )
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        type=Path,
        default=None,
        help="Path to the pinned raw snapshot (defaults to the single file in data/raw/).",
    )
    args = parser.parse_args()

    raw_path = args.raw if args.raw is not None else find_default_raw()
    print(f"Reading raw snapshot: {raw_path.relative_to(PROJECT_ROOT)}")

    long_rows = build_long(raw_path)
    long_rows.sort()  # deterministic output: by date, then ticker

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "sp500_membership_long.csv"
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "ticker"])
        writer.writerows(long_rows)

    print(f"Wrote: {out_path.relative_to(PROJECT_ROOT)}\n")
    validate(long_rows)


if __name__ == "__main__":
    main()