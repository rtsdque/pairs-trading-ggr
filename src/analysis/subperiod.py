"""Sub-period analysis: did the pairs-trading edge persist, shrink, or vanish?

The central research question. Full-sample averages (distance top20 ~0.13%/mo,
cointegration ~0.22%/mo) cannot distinguish "always weak" from "strong early,
decayed late". This splits both strategies' monthly returns into eras and reports
the trajectory of returns and Sharpe -- and crucially compares distance vs
cointegration per era, to see whether the better selection metric PERSISTS where
the classic one decays.

Boundary choice (anti data-snooping): boundaries are fixed on EXTERNAL grounds,
not tuned to the data. Two splits are reported:
  1. Even quartiles -- neutral.
  2. Pre/post-2006 (GGR publication) -- tests the "edge erodes once public" claim.

Returns are shown NET of a realistic modern cost (5 bps/leg, below both
break-evens so the trajectory shows through), with gross alongside for reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.engine_cached import build_cache as build_dist
from backtest.engine_cached import returns_from_cache as dist_returns
from backtest.engine_coint import build_cache_coint
from backtest.engine_coint import returns_from_cache as coint_returns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "subperiod_analysis.csv"

NET_COST_BPS = 5.0
CONV = "committed"


def stats(x: pd.Series) -> dict:
    n = len(x)
    if n == 0:
        return dict(n=0, mean_monthly=np.nan, ann_return=np.nan,
                    ann_vol=np.nan, sharpe=np.nan)
    mean_m = x.mean()
    ann_ret = (1 + x).prod() ** (12 / n) - 1
    ann_vol = x.std() * np.sqrt(12)
    sharpe = (mean_m * 12) / ann_vol if ann_vol > 0 else np.nan
    return dict(n=n, mean_monthly=mean_m, ann_return=ann_ret,
                ann_vol=ann_vol, sharpe=sharpe)


def quartiles(index: pd.DatetimeIndex, k: int = 4):
    return [(f"Q{i+1} ({index[c[0]].year}-{index[c[-1]].year})", c)
            for i, c in enumerate(np.array_split(np.arange(len(index)), k))]


def line(name: str, s: dict) -> str:
    return (f"  {name:18s} n={s['n']:3d}  mean={s['mean_monthly']:+.4f}  "
            f"ann={s['ann_return']:+.3f}  vol={s['ann_vol']:.3f}  "
            f"sharpe={s['sharpe']:+.2f}")


def main() -> None:
    panel = pd.read_parquet(PANEL)
    print("Building caches...", flush=True)
    dc = build_dist(panel, k=2.0)
    cc = build_cache_coint(panel, k=2.0)

    # Net series at 5 bps for the headline strategy of each approach.
    dist20 = dist_returns(dc, cost_bps_per_leg=NET_COST_BPS)["top20"][CONV]
    coint20 = coint_returns(cc, cost_bps_per_leg=NET_COST_BPS)["coint_top20"][CONV]
    # Gross for reference.
    dist20_g = dist_returns(dc, cost_bps_per_leg=0.0)["top20"][CONV]
    coint20_g = coint_returns(cc, cost_bps_per_leg=0.0)["coint_top20"][CONV]

    idx = dist20.index
    rows = []

    print("\n" + "=" * 70)
    print(f"SUB-PERIOD ANALYSIS  ({CONV}, net of {NET_COST_BPS:.0f} bps/leg)")
    print("=" * 70)

    print("\n--- Even quartiles (neutral split) ---")
    for label, c in quartiles(idx):
        print(f"\n{label}")
        ds = stats(dist20.iloc[c]); cs = stats(coint20.iloc[c])
        print(line("distance top20", ds))
        print(line("coint top20", cs))
        rows.append({"split": "quartile", "period": label, "strategy": "distance", **ds})
        rows.append({"split": "quartile", "period": label, "strategy": "coint", **cs})

    print("\n\n--- Pre/post-2006 (GGR publication) ---")
    for label, sl in [("pre-2006", slice(None, "2006-12-31")),
                      ("post-2006", slice("2007-01-01", None))]:
        print(f"\n{label}")
        ds = stats(dist20.loc[sl]); cs = stats(coint20.loc[sl])
        print(line("distance top20", ds))
        print(line("coint top20", cs))
        rows.append({"split": "pre_post_2006", "period": label, "strategy": "distance", **ds})
        rows.append({"split": "pre_post_2006", "period": label, "strategy": "coint", **cs})

    print("\n\n--- Full sample (gross vs net 5bps) ---")
    print(line("distance gross", stats(dist20_g)))
    print(line("distance net5", stats(dist20)))
    print(line("coint gross", stats(coint20_g)))
    print(line("coint net5", stats(coint20)))

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()