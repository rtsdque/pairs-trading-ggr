"""GGR walk-forward backtest engine with the 6-strategy monthly stagger.

Orchestrates formation -> trading -> returns across the whole sample, producing
monthly portfolio returns from ~1997 to 2026 under both capital conventions.

Walk-forward structure (strict out-of-sample):
  - 12-month FORMATION window selects pairs (uses only past data).
  - The following 6-month TRADING window trades them.
  - Formation rolls forward and repeats.

6-strategy monthly stagger (GGR Section 2):
  Forming pairs in only one month per year makes results depend on that
  arbitrary start month. GGR instead runs SIX overlapping cohorts, each offset
  by one month. Each cohort trades for 6 months, so at any calendar month all
  six are active, each in a different month of its own cycle. The headline
  monthly return is the AVERAGE across the active cohorts. This removes
  start-month dependence and keeps capital continuously deployed.

Normalization continuity:
  Each cohort normalizes the trading window on the SAME base as its formation
  window (formation day-1 prices), so the trading indices continue the
  formation indices rather than re-basing -- a GGR detail.

Pair sets captured (cheap once the machinery runs):
  top5, top20 (headline), and the 101-120 control set (GGR's check that
  profits aren't a pure top-pairs / utility artifact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.formation import run_formation, screen
from backtest.returns import portfolio_daily_returns, to_monthly
from backtest.trading import trade_pair

FORMATION_MONTHS = 12
TRADING_MONTHS = 6
N_STAGGER = 6

# Sanity guard: max allowed normalized trading-window price vs the formation
# base. Legit pairs stay under ~2.3x (99.9th pct); broken data reaches 8-7000x.
# 5x sits cleanly between, with >2x margin over any real pair.
MAX_EXCURSION = 5.0

PAIR_SETS = {
    "top5": (1, 5),
    "top20": (1, 20),
    "control_101_120": (101, 120),
}


@dataclass
class CohortResult:
    formation_start: pd.Timestamp
    trading_start: pd.Timestamp
    trading_end: pd.Timestamp
    n_screened_in: int
    daily: dict[str, pd.DataFrame]   # pair_set -> daily returns DataFrame


def run_cohort(
    panel: pd.DataFrame,
    formation_start: pd.Timestamp,
    k: float = 2.0,
) -> CohortResult | None:
    """Run one formation+trading cohort starting at formation_start."""
    f_start = formation_start
    f_end = f_start + relativedelta(months=FORMATION_MONTHS) - relativedelta(days=1)
    t_start = f_start + relativedelta(months=FORMATION_MONTHS)
    t_end = t_start + relativedelta(months=TRADING_MONTHS) - relativedelta(days=1)

    form_raw = panel.loc[f_start:f_end]
    trade_raw = panel.loc[t_start:t_end]
    if form_raw.empty or trade_raw.empty:
        return None

    try:
        res = run_formation(form_raw)
    except ValueError:
        return None

    form_kept, _ = screen(form_raw)
    base = form_kept.iloc[0]
    norm_form = form_kept / base
    # Trading window on the SAME base; restrict to screened survivors.
    norm_trade = trade_raw[form_kept.columns] / base

    daily_by_set: dict[str, pd.DataFrame] = {}
    for set_name, (lo, hi) in PAIR_SETS.items():
        pairs = res.slice_rank(lo, hi)
        n_committed = hi - lo + 1
        pair_trades = {}
        for p in pairs:
            if p.a not in norm_trade.columns or p.b not in norm_trade.columns:
                continue
            # Sanity guard (defense-in-depth): a legitimate pair's normalized
            # trading-window price stays near its formation base (real pairs:
            # 99.9th pct excursion ~2.3x). A price exceeding MAX_EXCURSION x the
            # base is broken data (gap/corruption the upstream cleaner missed),
            # not a real relative-value position -- refuse to trade it.
            legs = norm_trade[[p.a, p.b]].to_numpy()
            if not np.isfinite(legs).all():
                legs = legs[np.isfinite(legs)]
            if legs.size == 0 or np.abs(legs).max() > MAX_EXCURSION:
                continue
            pair_trades[(p.a, p.b)] = trade_pair(p.a, p.b, norm_form, norm_trade,
                                                 k=k, delay=1)
        daily_by_set[set_name] = portfolio_daily_returns(
            pair_trades, norm_trade, n_committed=n_committed
        )

    return CohortResult(
        formation_start=f_start, trading_start=t_start, trading_end=t_end,
        n_screened_in=res.n_screened_in, daily=daily_by_set,
    )


def run_backtest(
    panel: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
    k: float = 2.0,
) -> dict[str, pd.DataFrame]:
    """Full staggered walk-forward backtest.

    Returns {pair_set: monthly DataFrame[fully_invested, committed]}, where each
    month is the average across active staggered cohorts.
    """
    panel = panel.sort_index()
    first = pd.Timestamp(start) if start else panel.index.min().normalize()
    last = pd.Timestamp(end) if end else panel.index.max().normalize()

    # Cohort formation starts: every month, offset by 1 (the stagger emerges
    # naturally because each cohort trades 6 months -> 6 overlap at any time).
    formation_starts = pd.date_range(
        first, last - relativedelta(months=FORMATION_MONTHS + TRADING_MONTHS),
        freq="MS",  # month start
    )

    # Collect each cohort's daily returns, then average across cohorts per month.
    per_set_daily: dict[str, list[pd.DataFrame]] = {s: [] for s in PAIR_SETS}
    n_cohorts = 0
    for f_start in formation_starts:
        cohort = run_cohort(panel, f_start, k=k)
        if cohort is None:
            continue
        n_cohorts += 1
        for set_name, daily in cohort.daily.items():
            per_set_daily[set_name].append(daily)

    print(f"Cohorts run: {n_cohorts}")

    # For each pair set: concatenate all cohorts' daily returns and average by
    # date (multiple overlapping cohorts contribute to the same calendar day),
    # then compound to monthly.
    results: dict[str, pd.DataFrame] = {}
    for set_name, daily_list in per_set_daily.items():
        if not daily_list:
            continue
        fi = pd.concat([d["fully_invested"] for d in daily_list], axis=1)
        cm = pd.concat([d["committed"] for d in daily_list], axis=1)
        combined = pd.DataFrame({
            "fully_invested": fi.mean(axis=1, skipna=True),
            "committed": cm.mean(axis=1, skipna=True),
        }).sort_index()
        combined = combined.fillna(0.0)
        results[set_name] = to_monthly(combined)

    return results


if __name__ == "__main__":
    # Synthetic 4-year panel to test stagger plumbing, date alignment, averaging.
    dates = pd.bdate_range("2000-01-03", "2004-12-31")
    rng = np.random.default_rng(7)
    n_stocks = 60
    common = np.cumprod(1 + rng.normal(0, 0.008, len(dates)))
    cols = {}
    for i in range(n_stocks):
        idio = np.cumprod(1 + rng.normal(0, 0.006, len(dates)))
        cols[f"S{i:02d}"] = common * idio * (10 + i)
    panel = pd.DataFrame(cols, index=dates)

    out = run_backtest(panel, k=2.0)
    for set_name, monthly in out.items():
        cum_fi = (1 + monthly["fully_invested"]).prod() - 1
        cum_cm = (1 + monthly["committed"]).prod() - 1
        print(f"{set_name:16s} months={len(monthly):3d}  "
              f"cum_fully={cum_fi:+.4f}  cum_committed={cum_cm:+.4f}")
    print("OK")