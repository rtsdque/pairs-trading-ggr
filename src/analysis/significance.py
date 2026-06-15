"""Newey-West (HAC) significance tests for mean monthly returns.

A naive t-test assumes independent months, but pairs-trading returns are
autocorrelated: pairs held across month boundaries and the overlapping staggered
cohorts both induce serial correlation. That makes the naive standard error too
small and inflates significance. The Newey-West HAC estimator gives a standard
error robust to autocorrelation and heteroskedasticity -- honest t-stats.

We regress each return series on a constant (so the coefficient is the mean
return) and report the HAC t-stat and p-value. Lag length uses the Newey-West
rule of thumb floor(4*(n/100)^(2/9)); we also report lag sensitivity.

Run on both strategies, full sample and per sub-period, to establish WHICH of
the sub-period findings (distance decay, cointegration's surviving modern edge)
are statistically significant rather than noise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.engine_cached import build_cache as build_dist
from backtest.engine_cached import returns_from_cache as dist_returns
from backtest.engine_coint import build_cache_coint
from backtest.engine_coint import returns_from_cache as coint_returns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "significance.csv"

NET_COST_BPS = 5.0
CONV = "committed"


def nw_lag(n: int) -> int:
    return int(np.floor(4 * (n / 100) ** (2 / 9)))


def hac_test(ret: pd.Series, lags: int | None = None) -> dict:
    """Newey-West HAC test of mean(ret) != 0."""
    ret = ret.dropna()
    n = len(ret)
    if n < 10:
        return dict(n=n, mean=np.nan, t=np.nan, p=np.nan, lags=0)
    L = lags if lags is not None else nw_lag(n)
    X = np.ones(n)
    res = sm.OLS(ret.values, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    return dict(n=n, mean=float(res.params[0]), t=float(res.tvalues[0]),
                p=float(res.pvalues[0]), lags=L)


def stars(p: float) -> str:
    if np.isnan(p): return ""
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


def report(name: str, ret: pd.Series, rows: list, split: str, period: str) -> None:
    r = hac_test(ret)
    print(f"  {name:18s} mean={r['mean']:+.5f}  t={r['t']:+.2f}  "
          f"p={r['p']:.4f}{stars(r['p'])}  (n={r['n']}, lag={r['lags']})")
    rows.append({"split": split, "period": period, "series": name, **r})


def main() -> None:
    panel = pd.read_parquet(PANEL)
    print("Building caches...", flush=True)
    dc = build_dist(panel, k=2.0)
    cc = build_cache_coint(panel, k=2.0)

    dist20 = dist_returns(dc, cost_bps_per_leg=NET_COST_BPS)["top20"][CONV]
    coint20 = coint_returns(cc, cost_bps_per_leg=NET_COST_BPS)["coint_top20"][CONV]
    idx = dist20.index
    rows: list = []

    print("\n" + "=" * 70)
    print(f"NEWEY-WEST SIGNIFICANCE  ({CONV}, net {NET_COST_BPS:.0f} bps/leg)")
    print("(*** p<.01  ** p<.05  * p<.10)")
    print("=" * 70)

    print("\n--- Full sample ---")
    report("distance top20", dist20, rows, "full", "1997-2026")
    report("coint top20", coint20, rows, "full", "1997-2026")

    print("\n--- Even quartiles ---")
    for i, c in enumerate(np.array_split(np.arange(len(idx)), 4)):
        label = f"Q{i+1} ({idx[c[0]].year}-{idx[c[-1]].year})"
        print(f"\n{label}")
        report("distance top20", dist20.iloc[c], rows, "quartile", label)
        report("coint top20", coint20.iloc[c], rows, "quartile", label)

    print("\n--- Pre/post-2006 ---")
    for label, sl in [("pre-2006", slice(None, "2006-12-31")),
                      ("post-2006", slice("2007-01-01", None))]:
        print(f"\n{label}")
        report("distance top20", dist20.loc[sl], rows, "pre_post_2006", label)
        report("coint top20", coint20.loc[sl], rows, "pre_post_2006", label)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()