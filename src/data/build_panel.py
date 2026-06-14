"""Assemble cached per-ticker price files into one aligned price panel.

Input  : data/raw/prices/<TICKER>.csv      (date, adj_close) -- 834 usable files
         data/processed/price_coverage_report.csv
Output : data/processed/price_panel.parquet  (index=date, columns=tickers, values=adj_close)
         data/processed/price_panel.csv       (same, human-inspectable)

Design (kept deliberately minimal -- normalization is NOT done here):
  GGR forms normalized cumulative total-return indices PER formation window
  (each stock reset to 1.0 at the window start). That normalization is a
  backtest-time operation, not a panel operation, so the panel stores raw
  adjusted close. adj_close is already split/dividend-adjusted (yfinance
  auto_adjust), i.e. it IS a total-return series -- the right GGR input.

Partial-coverage tickers are included as-is: missing dates stay NaN. The
backtest's per-window liquidity screen (drop any stock with missing data in
that formation window) handles them exactly as GGR screens illiquid names.
Keeping them in the panel means a stock that delisted mid-sample can still
participate in the windows where it DID trade -- which preserves data and
keeps the analysis survivorship-honest.

We only assemble tickers classified full/partial in the coverage report;
empty/error tickers have no file to read.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRICE_DIR = PROJECT_ROOT / "data" / "raw" / "prices"
REPORT = PROJECT_ROOT / "data" / "processed" / "price_coverage_report.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "price_panel.parquet"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "price_panel.csv"


def usable_tickers() -> list[str]:
    report = pd.read_csv(REPORT)
    usable = report.loc[report["status"].isin(["full", "partial"]), "ticker"]
    return sorted(usable.astype(str))


def load_series(ticker: str) -> pd.Series | None:
    path = PRICE_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["date"], index_col="date")
    if "adj_close" not in df.columns or df.empty:
        return None
    s = df["adj_close"].copy()
    s.name = ticker
    # Guard against accidental duplicate dates in a cached file.
    s = s[~s.index.duplicated(keep="last")]
    return s


def main() -> None:
    tickers = usable_tickers()
    print(f"Usable tickers in report: {len(tickers):,}")

    series_list: list[pd.Series] = []
    missing_files = []
    for t in tickers:
        s = load_series(t)
        if s is None:
            missing_files.append(t)
        else:
            series_list.append(s)

    if missing_files:
        print(f"  WARNING: {len(missing_files)} usable tickers had no readable file: "
              f"{missing_files[:10]}{' ...' if len(missing_files) > 10 else ''}")

    print(f"Series loaded: {len(series_list):,}")

    # Outer-join all series on date -> aligned panel, gaps as NaN.
    panel = pd.concat(series_list, axis=1, sort=False)
    panel = panel.sort_index()
    panel.index.name = "date"

    # Validation stats.
    n_dates, n_tickers = panel.shape
    fill = panel.notna().mean().mean()  # avg fraction of non-NaN cells
    print("\nPanel assembled:")
    print(f"  shape            : {n_dates:,} dates x {n_tickers:,} tickers")
    print(f"  date range       : {panel.index.min().date()} -> {panel.index.max().date()}")
    print(f"  overall fill rate: {fill:.1%}  (non-NaN cells)")

    # Per-era fill to show the time-concentration of the gap.
    for era_start, era_end in [("1996", "2001"), ("2002", "2009"),
                               ("2010", "2017"), ("2018", "2026")]:
        sub = panel.loc[era_start:era_end]
        if len(sub):
            era_fill = sub.notna().mean().mean()
            avg_names = sub.notna().sum(axis=1).mean()
            print(f"  {era_start}-{era_end}: fill {era_fill:5.1%}  "
                  f"avg tradable names/day {avg_names:6.0f}")

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(OUT_PARQUET)
    panel.to_csv(OUT_CSV)
    print(f"\nWrote:\n  {OUT_PARQUET.relative_to(PROJECT_ROOT)}")
    print(f"  {OUT_CSV.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()