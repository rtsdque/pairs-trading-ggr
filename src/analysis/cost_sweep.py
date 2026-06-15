"""Transaction-cost sensitivity sweep: find the break-even cost level.

GGR's edge was real but cost-sensitive. This sweep re-runs the full backtest at
several flat per-leg cost levels and reports net mean monthly return for each
pair set, locating the BREAK-EVEN -- the cost at which the edge hits zero.

Each round-trip pair trade touches 4 legs, so a cost of c bps/leg removes
4c/10000 in return per trade. With gross top20 ~0.13%/month, even modest costs
are expected to erase the edge -- and the break-even, compared to realistic
modern bid-ask spreads (~1-5 bps/leg for liquid large-caps), is the answer to
'does the strategy survive costs today?'.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtest.engine import run_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PANEL = PROJECT_ROOT / "data" / "processed" / "price_panel_clean.parquet"
OUT = PROJECT_ROOT / "data" / "processed" / "cost_sweep.csv"

COST_LEVELS = [0, 1, 2, 5, 10, 20, 50]  # bps per leg
PAIR_SETS = ["top5", "top20", "control_101_120"]


def main() -> None:
    panel = pd.read_parquet(PANEL)
    rows = []
    for bps in COST_LEVELS:
        out = run_backtest(panel, k=2.0, cost_bps_per_leg=bps)
        row = {"bps_per_leg": bps, "cost_per_roundtrip_pct": 4 * bps / 100}
        for s in PAIR_SETS:
            m = out[s]["committed"]
            row[f"{s}_mean_monthly"] = m.mean()
            row[f"{s}_ann_return"] = (1 + m).prod() ** (12 / len(m)) - 1
        rows.append(row)
        print(f"bps/leg={bps:3d}  "
              + "  ".join(f"{s.split('_')[0]}={row[f'{s}_mean_monthly']:+.5f}"
                          for s in PAIR_SETS))

    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nSaved: {OUT.relative_to(PROJECT_ROOT)}")

    # Locate break-even for top20 (committed): highest bps with positive mean.
    t20 = df[["bps_per_leg", "top20_mean_monthly"]]
    pos = t20[t20["top20_mean_monthly"] > 0]
    if len(pos):
        be = pos["bps_per_leg"].max()
        print(f"\ntop20 stays positive up to ~{be} bps/leg "
              f"({4*be/100:.2f}% per round-trip).")
    else:
        print("\ntop20 is non-positive even at 0 bps/leg (no gross edge).")


if __name__ == "__main__":
    main()