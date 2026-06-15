"""Cointegration-based pair selection (Engle-Granger), as a refinement of distance.

This is the Phase 3 methodological EXTENSION beyond GGR's distance metric.

GGR selects pairs by minimum distance (sum of squared deviations of normalized
prices) -- purely co-movement. Distance has a known weakness: two series can
have tracked closely yet not be mean-reverting, so a divergence may never
converge (the forced-close losers). Cointegration tests the property pairs
trading actually relies on: that a linear combination of the two prices is
stationary (mean-reverting).

Option B framing (chosen): cointegration is a REFINEMENT of distance, not a
from-scratch search. We take the top-N distance-ranked candidate pairs, test
each for cointegration, and select the final pairs from those that pass. This
keeps the comparison apples-to-apples (same portfolio size as the distance
strategy) and is fast (N ~ 150 ADF tests per window, not ~100k).

Engle-Granger two-step (on formation-window PRICE LEVELS, not normalized):
  1. OLS regress P_a on P_b -> hedge ratio beta, residual spread = P_a - (a+bP_b).
  2. ADF-test the residuals. If the unit-root null is rejected at P_VALUE_MAX,
     the spread is stationary -> the pair is cointegrated and eligible.
  Rank eligible pairs by ADF statistic (most negative = most strongly
  mean-reverting) and take the top k.

Only SELECTION changes. Trading, returns, costs, the walk-forward loop, and the
cache are unchanged -- this produces a ranked pair list just like distance does,
and everything downstream consumes it identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from backtest.formation import Pair, run_formation

# Candidate pool: how many top distance pairs to test for cointegration.
CANDIDATE_POOL = 150
# Significance: reject unit-root null below this p-value to call a pair cointegrated.
P_VALUE_MAX = 0.05


@dataclass
class CointPair:
    a: str
    b: str
    adf_stat: float
    p_value: float
    beta: float


def engle_granger(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float, float]:
    """Return (adf_stat, p_value, beta) for the EG residual stationarity test."""
    X = sm.add_constant(pb)
    ols = sm.OLS(pa, X).fit()
    beta = float(ols.params[1])
    resid = ols.resid
    # maxlag=1, fixed (no autolag) for speed and determinism across ~150*347 tests.
    adf_stat, p_value = adfuller(resid, maxlag=1, autolag=None)[:2]
    return float(adf_stat), float(p_value), beta


def select_cointegrated(
    form_prices: pd.DataFrame,
    distance_pairs: list[Pair],
    candidate_pool: int = CANDIDATE_POOL,
    p_value_max: float = P_VALUE_MAX,
) -> list[CointPair]:
    """Test the top-`candidate_pool` distance pairs for cointegration.

    form_prices : RAW (un-normalized) formation-window prices (screened universe)
    distance_pairs : the distance-ranked pairs from run_formation(...).ranked
    Returns cointegrated pairs ranked by ADF statistic (most negative first).
    """
    out: list[CointPair] = []
    for p in distance_pairs[:candidate_pool]:
        if p.a not in form_prices.columns or p.b not in form_prices.columns:
            continue
        pa = form_prices[p.a].to_numpy(dtype=float)
        pb = form_prices[p.b].to_numpy(dtype=float)
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        try:
            adf_stat, p_value, beta = engle_granger(pa, pb)
        except Exception:
            continue
        if p_value <= p_value_max:
            out.append(CointPair(p.a, p.b, adf_stat, p_value, beta))

    # Rank by ADF statistic ascending (most negative = strongest mean reversion).
    out.sort(key=lambda c: c.adf_stat)
    return out


if __name__ == "__main__":
    # Validate on synthetic data: build a panel with a few genuinely cointegrated
    # pairs hidden among independent random walks; confirm EG finds them.
    rng = np.random.default_rng(1)
    T = 252
    idx = pd.bdate_range("2010-01-04", periods=T)

    cols = {}
    # 3 independent random walks
    for nm in ["IND1", "IND2", "IND3"]:
        cols[nm] = np.cumsum(rng.normal(0, 1, T)) + 100
    # A cointegrated trio: COA, COB tied to COA via stationary spread
    base = np.cumsum(rng.normal(0, 1, T)) + 100
    cols["COA"] = base + 50
    sp = np.zeros(T)
    for t in range(1, T):
        sp[t] = 0.6 * sp[t - 1] + rng.normal(0, 1)
    cols["COB"] = 1.5 * base + sp + 20

    panel = pd.DataFrame(cols, index=idx)

    # Distance ranking over the same window (normalized), then EG on raw prices.
    res = run_formation(panel)
    coint = select_cointegrated(panel, res.ranked, candidate_pool=50)

    print("Cointegrated pairs found (ranked by ADF):")
    for c in coint:
        print(f"  {c.a}-{c.b}  adf={c.adf_stat:.2f}  p={c.p_value:.4f}  beta={c.beta:.2f}")
    # The COA-COB pair must be detected.
    found = any({c.a, c.b} == {"COA", "COB"} for c in coint)
    print(f"\nCOA-COB detected as cointegrated: {found}  (expect True)")
    assert found
    print("OK")