"""Cached backtest engine: run formation+trades ONCE, re-apply costs cheaply.

Motivation: formation (the ~80k-pair SSD step) is the expensive part and is run
347 times in a full walk-forward. It depends only on the price panel -- NOT on
transaction costs. The cost sweep re-ran the entire backtest (formation included)
seven times, which is wasteful. This module separates the dependency chain:

  build_cache(panel, k)   -- runs formation + trade generation once (expensive)
  returns_from_cache(cache, cost_bps_per_leg)  -- cheap: applies costs only

A cost sweep then calls build_cache once and returns_from_cache N times, turning
slow sweeps into seconds after the one-time build.

This module ADDS a cached path; it does not modify the validated engine. The
equivalence test verifies returns_from_cache(cache, c) is identical to the
original run_backtest(panel, cost_bps_per_leg=c) for any c.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.engine import (
    FORMATION_MONTHS,
    MAX_EXCURSION,
    PAIR_SETS,
    TRADING_MONTHS,
)
from backtest.formation import run_formation, screen
from backtest.returns import portfolio_daily_returns, to_monthly
from backtest.trading import trade_pair


@dataclass
class CohortCache:
    """Everything needed to recompute one cohort's returns at any cost."""
    norm_trade: pd.DataFrame
    sets: dict[str, tuple[dict, int]] = field(default_factory=dict)


@dataclass
class BacktestCache:
    cohorts: list[CohortCache]


def build_cache(panel: pd.DataFrame, k: float = 2.0,
                start: str | None = None, end: str | None = None) -> BacktestCache:
    """Run formation + trade generation once across the full walk-forward."""
    panel = panel.sort_index()
    first = pd.Timestamp(start) if start else panel.index.min().normalize()
    last = pd.Timestamp(end) if end else panel.index.max().normalize()
    formation_starts = pd.date_range(
        first, last - relativedelta(months=FORMATION_MONTHS + TRADING_MONTHS),
        freq="MS",
    )

    cohorts: list[CohortCache] = []
    for f_start in formation_starts:
        f_end = f_start + relativedelta(months=FORMATION_MONTHS) - relativedelta(days=1)
        t_start = f_start + relativedelta(months=FORMATION_MONTHS)
        t_end = t_start + relativedelta(months=TRADING_MONTHS) - relativedelta(days=1)

        form_raw = panel.loc[f_start:f_end]
        trade_raw = panel.loc[t_start:t_end]
        if form_raw.empty or trade_raw.empty:
            continue
        try:
            res = run_formation(form_raw)
        except ValueError:
            continue

        form_kept, _ = screen(form_raw)
        base = form_kept.iloc[0]
        norm_form = form_kept / base
        norm_trade = trade_raw[form_kept.columns] / base

        cc = CohortCache(norm_trade=norm_trade)
        for set_name, (lo, hi) in PAIR_SETS.items():
            pairs = res.slice_rank(lo, hi)
            n_committed = hi - lo + 1
            pair_trades = {}
            for p in pairs:
                if p.a not in norm_trade.columns or p.b not in norm_trade.columns:
                    continue
                legs = norm_trade[[p.a, p.b]].to_numpy()
                if not np.isfinite(legs).all():
                    legs = legs[np.isfinite(legs)]
                if legs.size == 0 or np.abs(legs).max() > MAX_EXCURSION:
                    continue
                pair_trades[(p.a, p.b)] = trade_pair(p.a, p.b, norm_form,
                                                     norm_trade, k=k, delay=1)
            cc.sets[set_name] = (pair_trades, n_committed)
        cohorts.append(cc)

    return BacktestCache(cohorts=cohorts)


def returns_from_cache(cache: BacktestCache,
                       cost_bps_per_leg: float = 0.0) -> dict[str, pd.DataFrame]:
    """Cheap: apply costs to cached trades and produce monthly returns."""
    per_set_daily: dict[str, list[pd.DataFrame]] = {s: [] for s in PAIR_SETS}
    for cc in cache.cohorts:
        for set_name, (pair_trades, n_committed) in cc.sets.items():
            daily = portfolio_daily_returns(
                pair_trades, cc.norm_trade, n_committed=n_committed,
                cost_bps_per_leg=cost_bps_per_leg,
            )
            per_set_daily[set_name].append(daily)

    results: dict[str, pd.DataFrame] = {}
    for set_name, daily_list in per_set_daily.items():
        if not daily_list:
            continue
        fi = pd.concat([d["fully_invested"] for d in daily_list], axis=1)
        cm = pd.concat([d["committed"] for d in daily_list], axis=1)
        combined = pd.DataFrame({
            "fully_invested": fi.mean(axis=1, skipna=True),
            "committed": cm.mean(axis=1, skipna=True),
        }).sort_index().fillna(0.0)
        results[set_name] = to_monthly(combined)
    return results