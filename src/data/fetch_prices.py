"""Fetch daily adjusted-close prices for every ticker that was ever in the index.

Input  : data/processed/sp500_membership_long.csv   (date, ticker)
Output : data/raw/prices/<TICKER>.csv                (one file per ticker, cached)
         data/processed/price_coverage_report.csv    (per-ticker fetch outcome)

Why per-ticker (not bulk download):
  ~700 of the 1,202 tickers left the index over 1996-2026. Many are delisted
  and yfinance serves them partially or not at all. Fetching one ticker at a
  time gives clean per-ticker success/failure classification and isolates
  failures so one bad symbol can't corrupt a batch.

Why resumable:
  A full run hits Yahoo ~1,200 times and takes a while. Each ticker's data is
  cached to its own file; rerunning skips tickers already fetched. You pay the
  slow fetch once. Delete a ticker's file (or the whole prices/ dir) to refetch.

Coverage classification (the survivorship-bias audit):
  full    - data covers most of the ticker's index-membership span
  partial - some data, but a meaningful chunk of the membership span is missing
            (the classic signature of a delisted name yfinance only half-keeps)
  empty   - Yahoo returned nothing
  error   - the request raised an exception

We use auto_adjust (the yfinance default), giving a split- and dividend-adjusted
close -- the total-return series pairs trading needs, since positions are held
for weeks and dividends materially move the spread.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEMBERSHIP = PROJECT_ROOT / "data" / "processed" / "sp500_membership_long.csv"
PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
REPORT = PROJECT_ROOT / "data" / "processed" / "price_coverage_report.csv"

# Fetch window: index data starts 1996-01-02; pad the end slightly past the snapshot.
START = "1996-01-01"
END = "2026-06-03"

# Be polite to Yahoo; avoids throttling on a long run.
SLEEP_BETWEEN = 0.4  # seconds

# A ticker is "full" if fetched data covers at least this fraction of the
# trading days between its first and last index-membership date.
FULL_COVERAGE_THRESHOLD = 0.90


def load_membership() -> pd.DataFrame:
    df = pd.read_csv(MEMBERSHIP, parse_dates=["date"])
    return df


def membership_span(df: pd.DataFrame) -> pd.DataFrame:
    """First and last index-membership date for each ticker."""
    span = df.groupby("ticker")["date"].agg(["min", "max", "count"])
    span.columns = ["mem_start", "mem_end", "mem_obs"]
    return span


def fetch_one(ticker: str) -> pd.DataFrame | None:
    """Download one ticker's adjusted daily close. Returns a DataFrame or None."""
    df = yf.download(
        ticker,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        return None
    # Single-ticker downloads come back with a multi-level column index in this
    # version; flatten to just the price fields.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Keep only the adjusted close, named clearly.
    if "Close" not in df.columns:
        return None
    out = df[["Close"]].rename(columns={"Close": "adj_close"})
    out.index.name = "date"
    return out


def classify(price_df: pd.DataFrame | None, mem_start, mem_end) -> tuple[str, dict]:
    """Return (status, stats) for the coverage report."""
    if price_df is None or price_df.empty:
        return "empty", {"price_start": None, "price_end": None, "price_obs": 0,
                         "coverage_frac": 0.0}

    p_start = price_df.index.min()
    p_end = price_df.index.max()
    p_obs = len(price_df)

    # Expected trading days over the membership span (approx via business days).
    expected = pd.bdate_range(mem_start, mem_end)
    expected_n = max(len(expected), 1)
    # Count fetched observations that fall within the membership span.
    within = price_df.loc[
        (price_df.index >= mem_start) & (price_df.index <= mem_end)
    ]
    coverage = len(within) / expected_n

    stats = {
        "price_start": p_start.date(),
        "price_end": p_end.date(),
        "price_obs": p_obs,
        "coverage_frac": round(coverage, 4),
    }
    status = "full" if coverage >= FULL_COVERAGE_THRESHOLD else "partial"
    return status, stats


def main() -> None:
    PRICE_DIR.mkdir(parents=True, exist_ok=True)

    membership = load_membership()
    span = membership_span(membership)
    tickers = sorted(span.index)
    print(f"Tickers to fetch: {len(tickers):,}")
    print(f"Cache dir       : {PRICE_DIR.relative_to(PROJECT_ROOT)}")
    print(f"Window          : {START} -> {END}\n")

    report_rows = []
    for i, ticker in enumerate(tickers, 1):
        cache_path = PRICE_DIR / f"{ticker}.csv"
        mem_start = span.loc[ticker, "mem_start"]
        mem_end = span.loc[ticker, "mem_end"]

        # Resume: if already cached, just reclassify from the cached file.
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
            except Exception:
                cached = None
            status, stats = classify(cached, mem_start, mem_end)
            report_rows.append({"ticker": ticker, "status": status,
                                "mem_start": mem_start.date(), "mem_end": mem_end.date(),
                                **stats, "cached": True})
            if i % 100 == 0:
                print(f"[{i:4d}/{len(tickers)}] {ticker:8s} cached ({status})")
            continue

        # Fetch fresh.
        try:
            price_df = fetch_one(ticker)
            status, stats = classify(price_df, mem_start, mem_end)
            if price_df is not None and not price_df.empty:
                price_df.to_csv(cache_path)
        except Exception as e:  # noqa: BLE001  -- we want to log, not crash the run
            status = "error"
            stats = {"price_start": None, "price_end": None, "price_obs": 0,
                     "coverage_frac": 0.0}
            print(f"[{i:4d}/{len(tickers)}] {ticker:8s} ERROR: {type(e).__name__}: {str(e)[:60]}")

        report_rows.append({"ticker": ticker, "status": status,
                            "mem_start": mem_start.date(), "mem_end": mem_end.date(),
                            **stats, "cached": False})

        if i % 25 == 0:
            done = sum(1 for r in report_rows if r["status"] in ("full", "partial"))
            print(f"[{i:4d}/{len(tickers)}] {ticker:8s} {status:8s}  "
                  f"(usable so far: {done})")

        time.sleep(SLEEP_BETWEEN)

    report = pd.DataFrame(report_rows)
    report.to_csv(REPORT, index=False)

    # Summary -- this is the survivorship-bias audit headline.
    print("\n" + "=" * 60)
    print("COVERAGE SUMMARY")
    print("=" * 60)
    counts = report["status"].value_counts()
    for status in ("full", "partial", "empty", "error"):
        print(f"  {status:8s}: {counts.get(status, 0):5d}")
    print(f"  {'TOTAL':8s}: {len(report):5d}")
    usable = counts.get("full", 0) + counts.get("partial", 0)
    print(f"\nUsable (full+partial): {usable} / {len(report)} "
          f"({usable / len(report):.1%})")
    print(f"\nReport written: {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()