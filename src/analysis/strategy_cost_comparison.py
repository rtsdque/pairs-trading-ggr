"""Compare distance vs cointegration selection across transaction-cost levels.

The cointegration extension earns more GROSS than distance (~0.22% vs ~0.13%
monthly, top20 committed). The question that matters: does that larger gross edge
survive realistic costs BETTER? A strategy is only a contribution if its edge
persists net of friction.

This builds BOTH caches once (distance + cointegration), then sweeps the same
per-leg cost levels on each, reporting top20 net mean monthly return side by side
and each strategy's break-even. The comparison is the deliverable for the
cointegration half of Phase 3.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.engine_cached import build_cache as build_dist
from backtest.engine_cached import returns_from_cache as dist_returns
from backtest.engine_coint import build_cache_coint
from backtest.engine_coint import returns_from_cache as coint_returns

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "strategy_cost_comparison.csv"

COST_LEVELS = [0, 1, 2, 5, 10, 20, 50]


def mean_monthly(out: dict, key: str) -> float:
    return out[key]["committed"].mean()


def break_even(levels, means) -> str:
    """Largest cost level with positive mean; interpolate a rough crossing."""
    pos = [(c, m) for c, m in zip(levels, means) if m > 0]
    if not pos:
        return "<0 at 0bps"
    last_pos = pos[-1][0]
    for c, m in zip(levels, means):
        if m <= 0 and c > last_pos:
            return f"~{last_pos}-{c} bps/leg"
    return f">{last_pos} bps/leg"


def main() -> None:
    panel = pd.read_parquet(PANEL)

    print("Building distance cache...", flush=True)
    dist_cache = build_dist(panel, k=2.0)
    print("Building cointegration cache (slower)...", flush=True)
    coint_cache = build_cache_coint(panel, k=2.0)
    print("Both caches built. Sweeping costs...\n", flush=True)

    rows = []
    dist_means, coint_means = [], []
    print(f"{'bps/leg':>8s}  {'distance_top20':>15s}  {'coint_top20':>13s}")
    for bps in COST_LEVELS:
        d = mean_monthly(dist_returns(dist_cache, cost_bps_per_leg=bps), "top20")
        c = mean_monthly(coint_returns(coint_cache, cost_bps_per_leg=bps), "coint_top20")
        dist_means.append(d)
        coint_means.append(c)
        rows.append({"bps_per_leg": bps, "distance_top20": d, "coint_top20": c})
        print(f"{bps:>8d}  {d:>+15.5f}  {c:>+13.5f}", flush=True)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nSaved: {OUT.relative_to(PROJECT_ROOT)}")
    print(f"\nDistance break-even     : {break_even(COST_LEVELS, dist_means)}")
    print(f"Cointegration break-even: {break_even(COST_LEVELS, coint_means)}")


if __name__ == "__main__":
    main()