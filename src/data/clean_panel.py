"""Clean the price panel: remove bad ticks, drop pervasively-corrupt series.

Input  : data/processed/price_panel.parquet      (raw assembled panel)
Output : data/processed/price_panel_clean.parquet (cleaned)
         data/processed/price_panel_clean.csv
         data/processed/cleaning_report.csv        (what was dropped / nulled)

Free price data (yfinance) contains corrupt observations -- bad ticks and
entirely broken series for some delisted names. Left untreated, a single bad
price detonates the buy-and-hold return engine (a normalized price that jumps
100x produces an absurd payoff).

The hard part: a simple magnitude threshold CANNOT distinguish a real crash from
a data error. AAPL fell ~50% on 2000-09-29 (real profit warning); AIG collapsed
in Sept 2008 (real); Avis doubled on 2021-11-02 (real short squeeze). These look
identical to a threshold and must be KEPT.

The distinguishing feature is REVERSAL, not magnitude:
  - A bad tick spikes and round-trips in one day (e.g. 16.77 -> 36.89 -> 16.77).
  - A real move persists (AAPL: 0.80 -> 0.385 -> 0.363 -> 0.334; keeps going).

Rule: a price at day t is a bad tick iff the move INTO it (t-1 -> t) is
implausible (|log return| > ln 2, i.e. >2x or <0.5x) AND the move OUT of it
(t -> t+1) reverses it -- opposite sign, returning to within RESTORE_TOL (log)
of the pre-spike level. We iterate to convergence (clustered ticks peel away in
layers). A ticker whose bad-tick count exceeds DROP_BADTICK_COUNT is deemed
fundamentally broken and dropped entirely (e.g. a series jumping between 19,325
and 2.19).

Log returns are used because they are symmetric (a 2x jump and a halving have
equal magnitude ln 2). Everything removed is logged for the methodology section.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
IN_PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel.parquet"
OUT_PARQUET = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.csv"
REPORT = PROJECT_ROOT / "data" / "processed" / "cleaning_report.csv"

np.seterr(divide="ignore", invalid="ignore")  # NaN/zero gaps handled by isfinite checks
LOG2 = np.log(2.0)            # implausible single-day move threshold
RESTORE_TOL = 0.25           # t+1 within this log-distance of t-1 => round-trip
DROP_BADTICK_COUNT = 10      # more bad ticks than this -> drop the whole ticker
MAX_PASSES = 20


def bad_tick_mask(s: pd.Series) -> pd.Series:
    """Boolean mask: implausible move IN that reverses OUT within one day."""
    p = s.to_numpy(dtype=float)
    n = len(p)
    mask = np.zeros(n, dtype=bool)
    for t in range(1, n - 1):
        if not (np.isfinite(p[t]) and np.isfinite(p[t - 1]) and np.isfinite(p[t + 1])):
            continue
        r_in = np.log(p[t] / p[t - 1])
        if abs(r_in) <= LOG2:
            continue
        r_out = np.log(p[t + 1] / p[t])
        opposite = np.sign(r_out) == -np.sign(r_in)
        back_near = abs(np.log(p[t + 1] / p[t - 1])) < RESTORE_TOL
        if opposite and back_near:
            mask[t] = True
    return pd.Series(mask, index=s.index)


def clean_series(s: pd.Series) -> tuple[pd.Series, int]:
    """Iteratively null bad ticks until none remain. Returns (clean, n_nulled)."""
    work = s.copy()
    total = 0
    for _ in range(MAX_PASSES):
        m = bad_tick_mask(work)
        k = int(m.sum())
        if k == 0:
            break
        work[m.to_numpy()] = np.nan
        total += k
    return work, total


def clean(panel: pd.DataFrame):
    """Return (cleaned_panel, report_df, dropped_list, n_nulled)."""
    cleaned_cols = {}
    dropped = []
    nulled_by = {}
    for col in panel.columns:
        c, n = clean_series(panel[col])
        if n > DROP_BADTICK_COUNT:
            dropped.append(col)
        else:
            cleaned_cols[col] = c
            if n > 0:
                nulled_by[col] = n

    cleaned = pd.DataFrame(cleaned_cols).sort_index()
    cleaned.index.name = "date"

    rows = []
    for t in sorted(dropped):
        rows.append({"ticker": t, "action": "dropped",
                     "reason": "pervasively corrupt (too many bad ticks)"})
    for t, n in sorted(nulled_by.items()):
        rows.append({"ticker": t, "action": "despiked",
                     "reason": f"{n} bad tick(s) nulled"})
    report = pd.DataFrame(rows, columns=["ticker", "action", "reason"])

    return cleaned, report, sorted(dropped), int(sum(nulled_by.values()))


def main() -> None:
    panel = pd.read_parquet(IN_PANEL)
    print(f"Input panel: {panel.shape[0]:,} dates x {panel.shape[1]:,} tickers")

    cleaned, report, dropped, n_nulled = clean(panel)

    # Verify convergence: no bad ticks remain in any survivor.
    remaining = sum(int(bad_tick_mask(cleaned[c]).sum()) for c in cleaned.columns)

    print(f"\nDropped {len(dropped)} pervasively-corrupt tickers:")
    print(f"  {dropped}")
    print(f"Nulled {n_nulled} isolated bad ticks across survivors")
    print(f"\nCleaned panel: {cleaned.shape[0]:,} dates x {cleaned.shape[1]:,} tickers")
    print(f"Bad ticks remaining after cleaning: {remaining}  (want 0)")
    print("\nNote: real crashes (AAPL 2000, AIG 2008, APA 2020, CAR 2021) are")
    print("preserved -- only one-day round-trip ticks are removed.")

    cleaned.to_parquet(OUT_PARQUET)
    cleaned.to_csv(OUT_CSV)
    report.to_csv(REPORT, index=False)
    print(f"\nWrote:\n  {OUT_PARQUET.relative_to(PROJECT_ROOT)}")
    print(f"  {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()