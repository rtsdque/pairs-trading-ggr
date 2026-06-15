"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The most stringent test in the project. A normal Sharpe assumes ONE strategy was
tested. But both selection methods search many candidate pairs and keep winners
(distance: top-20 of ~80k pairs/window; cointegration: ~150 ADF-tested
candidates/window). Selecting the best of many noisy trials inflates the winner's
Sharpe even with zero true edge -- the multiple-testing problem, the central
critique of empirical finance. Non-normal returns (skew, fat tails) inflate naive
Sharpe further.

The DSR corrects both: it computes the probability the TRUE Sharpe is positive
after (a) subtracting the Sharpe expected from the best of N noise trials and
(b) adjusting for the return distribution's skewness and kurtosis. DSR > 0.95 is
the usual bar for "survives". Because the effective number of independent trials
N is genuinely uncertain, we report DSR across a range of N rather than one value
-- robustness (or fragility) across N is itself the finding.

Expected outcome given the Newey-West results (post-2006 already insignificant):
the DSR will likely show that even the significant full-sample edge is
substantially attributable to multiple testing -- the rigorous, honest endpoint.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.engine_cached import build_cache as build_dist
from backtest.engine_cached import returns_from_cache as dist_returns
from backtest.engine_coint import build_cache_coint
from backtest.engine_coint import returns_from_cache as coint_returns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "deflated_sharpe.csv"

NET_COST_BPS = 5.0
CONV = "committed"
EULER = 0.5772156649
TRIAL_COUNTS = [1, 20, 50, 150, 500]


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    if n_trials < 2:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return np.sqrt(var_sr) * ((1 - EULER) * z1 + EULER * z2)


def deflated_sharpe(returns: pd.Series, n_trials: int) -> dict:
    r = np.asarray(returns.dropna(), dtype=float)
    T = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    sr = mu / sd if sd > 0 else 0.0
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))
    var_sr_est = (1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2) / (T - 1)
    sr0 = expected_max_sharpe(n_trials, var_sr=1.0 / T)
    dsr = float(stats.norm.cdf((sr - sr0) / np.sqrt(var_sr_est))) if var_sr_est > 0 else np.nan
    return dict(sr_ann=sr * np.sqrt(12), benchmark=sr0, dsr=dsr)


def main() -> None:
    panel = pd.read_parquet(PANEL)
    print("Building caches...", flush=True)
    dc = build_dist(panel, k=2.0)
    cc = build_cache_coint(panel, k=2.0)

    series = {
        "distance top20": dist_returns(dc, cost_bps_per_leg=NET_COST_BPS)["top20"][CONV],
        "coint top20": coint_returns(cc, cost_bps_per_leg=NET_COST_BPS)["coint_top20"][CONV],
    }

    rows = []
    print("\n" + "=" * 70)
    print(f"DEFLATED SHARPE RATIO  ({CONV}, net {NET_COST_BPS:.0f} bps/leg, full sample)")
    print("DSR = P(true Sharpe > 0) after multiple-testing + non-normality haircut")
    print("DSR > 0.95 => survives.  Reported across trial counts N.")
    print("=" * 70)

    for name, ret in series.items():
        base = deflated_sharpe(ret, 1)
        print(f"\n{name}:  annualized Sharpe = {base['sr_ann']:.2f}")
        for N in TRIAL_COUNTS:
            d = deflated_sharpe(ret, N)
            verdict = "survives" if d["dsr"] > 0.95 else ("marginal" if d["dsr"] > 0.5 else "fails")
            print(f"  N={N:4d}: benchmark_SR={d['benchmark']:.3f}  DSR={d['dsr']:.4f}  ({verdict})")
            rows.append({"series": name, "n_trials": N, "sr_ann": d["sr_ann"],
                         "benchmark_sr": d["benchmark"], "dsr": d["dsr"]})

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()