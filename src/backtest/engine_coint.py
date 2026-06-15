"""Cointegration-selection cached engine (Phase 3 extension).

Mirrors engine_cached.py exactly, but the SELECTION step uses Engle-Granger
cointegration (as a refinement of the distance candidate pool, Option B) instead
of distance ranking. Everything downstream -- trading, returns, costs, stagger,
monthly aggregation -- is identical, so results are directly comparable to the
distance strategy.

Per cohort:
  1. Distance formation -> ranked candidate pool (as before).
  2. select_cointegrated on the top CANDIDATE_POOL distance pairs -> cointegrated
     pairs ranked by ADF statistic.
  3. Pair sets: top5, top20 (by cointegration strength), and a coint_21_40
     control (weaker-but-cointegrated tier, parallel to distance's 101-120).

This build is slower than the distance cache (~150 ADF tests x 347 cohorts), but
the cache means you pay it once, then re-apply costs cheaply.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

from backtest.cointegration import select_cointegrated, CANDIDATE_POOL, P_VALUE_MAX
from backtest.engine import FORMATION_MONTHS, MAX_EXCURSION, TRADING_MONTHS
from backtest.formation import run_formation, screen
from backtest.returns import portfolio_daily_returns, to_monthly
from backtest.trading import trade_pair

# Pair sets for the cointegration strategy (1-indexed inclusive rank ranges).
COINT_PAIR_SETS = {
    "coint_top5": (1, 5),
    "coint_top20": (1, 20),
    "coint_21_40": (21, 40),
}


@dataclass
class CohortCache:
    norm_trade: pd.DataFrame
    sets: dict[str, tuple[dict, int]] = field(default_factory=dict)


@dataclass
class BacktestCache:
    cohorts: list[CohortCache]


def build_cache_coint(panel: pd.DataFrame, k: float = 2.0,
                      start: str | None = None, end: str | None = None,
                      candidate_pool: int = CANDIDATE_POOL,
                      p_value_max: float = P_VALUE_MAX) -> BacktestCache:
    """Run distance formation + cointegration selection + trades once."""
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

        # Cointegration selection on the distance candidate pool (raw prices).
        coint = select_cointegrated(form_kept, res.ranked,
                                    candidate_pool=candidate_pool,
                                    p_value_max=p_value_max)

        cc = CohortCache(norm_trade=norm_trade)
        for set_name, (lo, hi) in COINT_PAIR_SETS.items():
            selected = coint[lo - 1:hi]  # 1-indexed inclusive
            n_committed = hi - lo + 1
            pair_trades = {}
            for c in selected:
                if c.a not in norm_trade.columns or c.b not in norm_trade.columns:
                    continue
                legs = norm_trade[[c.a, c.b]].to_numpy()
                if not np.isfinite(legs).all():
                    legs = legs[np.isfinite(legs)]
                if legs.size == 0 or np.abs(legs).max() > MAX_EXCURSION:
                    continue
                pair_trades[(c.a, c.b)] = trade_pair(c.a, c.b, norm_form,
                                                     norm_trade, k=k, delay=1)
            cc.sets[set_name] = (pair_trades, n_committed)
        cohorts.append(cc)

    return BacktestCache(cohorts=cohorts)


def returns_from_cache(cache: BacktestCache,
                       cost_bps_per_leg: float = 0.0) -> dict[str, pd.DataFrame]:
    """Cheap: apply costs to cached cointegration-selected trades."""
    set_names = COINT_PAIR_SETS.keys()
    per_set_daily: dict[str, list[pd.DataFrame]] = {s: [] for s in set_names}
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