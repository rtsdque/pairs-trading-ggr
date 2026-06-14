"""GGR trading stage: trade one selected pair over the 6-month trading window.

Consumes a Pair (from formation.py) plus the normalized price series, and
produces the list of round-trip trades for that pair over the trading window.

GGR trading rule (Section 1):
  - The "spread" is the difference between the two normalized total-return
    indices: spread_t = P_a,t - P_b,t.
  - sigma is the standard deviation of the spread DURING THE FORMATION period.
  - Open when |spread| exceeds k*sigma (k=2 in GGR's main spec):
      spread > +k*sigma  -> A relatively expensive: SHORT A, LONG B  (dir = -1)
      spread < -k*sigma  -> A relatively cheap:     LONG A, SHORT B  (dir = +1)
  - Close when the spread reverts through zero (the normalized series cross).
  - One-day-delay execution (GGR's preferred spec): a signal generated from
    day t's spread is EXECUTED at day t+1's prices. This avoids using the same
    closing price to both generate and execute a trade (a subtle lookahead).
  - If a position is still open at the end of the trading window, force-close
    it on the last day.

IMPORTANT: the trading window continues the formation-period normalization --
prices are NOT re-based to 1.0 at the trading-window start. The normalized
panel passed in must already be on the formation-period base.

This stage records trades (entry/exit dates, direction, entry/exit spread,
and the normalized prices at entry/exit for each leg). Converting trades into
returns (committed-capital vs fully-invested, monthly stagger) is a SEPARATE
downstream stage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Trade:
    pair_a: str
    pair_b: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    direction: int            # -1: short A / long B ; +1: long A / short B
    entry_spread: float
    exit_spread: float
    # normalized prices of each leg at entry/exit (for return calc downstream)
    entry_a: float
    entry_b: float
    exit_a: float
    exit_b: float
    forced_close: bool        # True if closed by end-of-window, not convergence


def formation_sigma(norm_formation: pd.DataFrame, a: str, b: str) -> float:
    """Std dev of the (A - B) spread over the formation window."""
    spread = norm_formation[a] - norm_formation[b]
    return float(spread.std())


def trade_pair(
    a: str,
    b: str,
    norm_formation: pd.DataFrame,
    norm_trading: pd.DataFrame,
    k: float = 2.0,
    delay: int = 1,
) -> list[Trade]:
    """Generate round-trip trades for one pair over the trading window.

    norm_formation / norm_trading: normalized price panels (columns include a, b)
    for the formation and trading windows respectively, on a COMMON base.
    """
    sigma = formation_sigma(norm_formation, a, b)
    thresh = k * sigma
    if not np.isfinite(thresh) or thresh == 0:
        return []  # degenerate pair (no spread variation) -> no trades

    pa = norm_trading[a].to_numpy()
    pb = norm_trading[b].to_numpy()
    spread = pa - pb
    dates = norm_trading.index
    n = len(spread)

    trades: list[Trade] = []
    position = 0
    entry_i: int | None = None
    pending: tuple | None = None

    for t in range(n):
        # Execute any pending (delayed) action at TODAY's prices.
        if pending is not None:
            kind = pending[0]
            if kind == "open":
                position = pending[1]
                entry_i = t
            elif kind == "close":
                trades.append(
                    Trade(
                        pair_a=a, pair_b=b,
                        entry_date=dates[entry_i], exit_date=dates[t],
                        direction=position,
                        entry_spread=float(spread[entry_i]),
                        exit_spread=float(spread[t]),
                        entry_a=float(pa[entry_i]), entry_b=float(pb[entry_i]),
                        exit_a=float(pa[t]), exit_b=float(pb[t]),
                        forced_close=False,
                    )
                )
                position = 0
                entry_i = None
            pending = None

        # Generate a signal on TODAY's spread for execution NEXT day (delay).
        if position == 0 and pending is None:
            if spread[t] > thresh:
                pending = ("open", -1)
            elif spread[t] < -thresh:
                pending = ("open", +1)
        elif position != 0 and pending is None:
            crossed = (position == -1 and spread[t] <= 0) or (
                position == +1 and spread[t] >= 0
            )
            if crossed:
                pending = ("close",)

        # NOTE on delay at the boundary: a signal on the last day cannot execute
        # (no t+1 within the window). Such positions are handled by the forced
        # close below if a position is open; un-executed open-signals simply
        # never become positions, matching "no trade without a next-day fill".

    # Force-close any open position on the last day of the window.
    if position != 0 and entry_i is not None:
        last = n - 1
        trades.append(
            Trade(
                pair_a=a, pair_b=b,
                entry_date=dates[entry_i], exit_date=dates[last],
                direction=position,
                entry_spread=float(spread[entry_i]),
                exit_spread=float(spread[last]),
                entry_a=float(pa[entry_i]), entry_b=float(pb[entry_i]),
                exit_a=float(pa[last]), exit_b=float(pb[last]),
                forced_close=True,
            )
        )

    return trades


if __name__ == "__main__":
    # Deterministic test: known divergence + convergence, verify delayed fills.
    form_dates = pd.bdate_range("2010-01-04", periods=20)
    trade_dates = pd.bdate_range("2010-02-01", periods=20)

    form_spread = np.array([0.01, -0.01] * 10)            # sigma ~ 0.01
    trade_spread = np.array(
        [0.000, 0.002, -0.001, 0.050, 0.048, 0.045, 0.040, 0.030, 0.020, 0.010,
         -0.001, 0.000, 0.001, 0.000, -0.002, 0.001, 0.000, 0.002, -0.001, 0.000]
    )

    nf = pd.DataFrame({"A": 1.0 + form_spread, "B": np.ones(20)}, index=form_dates)
    nt = pd.DataFrame({"A": 1.0 + trade_spread, "B": np.ones(20)}, index=trade_dates)

    out = trade_pair("A", "B", nf, nt, k=2.0, delay=1)
    assert len(out) == 1, f"expected 1 trade, got {len(out)}"
    tr = out[0]
    print(f"direction   : {tr.direction}  (expect -1: spread went positive)")
    print(f"entry_date  : {tr.entry_date.date()}")
    print(f"exit_date   : {tr.exit_date.date()}")
    print(f"entry_spread: {tr.entry_spread:.3f}  (expect 0.048: day AFTER 0.050 signal)")
    print(f"exit_spread : {tr.exit_spread:.3f}")
    print(f"forced_close: {tr.forced_close}")
    assert tr.direction == -1
    assert abs(tr.entry_spread - 0.048) < 1e-9  # delayed fill, not 0.050
    assert tr.forced_close is False
    print("OK")