"""GGR return calculation: buy-and-hold pair payoffs -> portfolio monthly returns.

Consumes Trades (from trading.py) plus the normalized trading-window prices, and
produces monthly portfolio returns under BOTH of GGR's capital conventions, with
optional transaction costs.

Buy-and-hold accounting (GGR-faithful):
  At entry, commit $1 long + $1 short. Legs then DRIFT (no daily rebalancing).
  Cumulative payoff per $1-long/$1-short at day t (relative to entry day 0):
      payoff_t = (P_long_t / P_long_0 - 1) + (1 - P_short_t / P_short_0)
  Daily payoff change is the day's P&L. Which leg is long depends on direction:
      dir = +1  -> long A, short B ;  dir = -1 -> long B, short A

Transaction costs (flat, configurable):
  Charge cost_bps_per_leg basis points per leg per transaction. A round-trip
  pair trade touches 4 legs (open A, open B, close A, close B), so it pays
  4 * bps/10000 in total, in return terms. We book 2 legs' cost on the entry
  day and 2 legs' cost on the exit day (faithful to when the trades occur).
  Default 0 -> gross returns, identical to the no-cost engine.

Two capital conventions (GGR Section 2):
  FULLY-INVESTED  : divide by capital in pairs that actually OPENED.
  COMMITTED-CAPITAL: divide by capital committed to ALL selected pairs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.trading import Trade


def pair_daily_payoff(
    trades: list[Trade],
    norm_trading: pd.DataFrame,
    a: str,
    b: str,
    cost_bps_per_leg: float = 0.0,
) -> tuple[pd.Series, pd.Series]:
    """Daily payoff series for one pair over the trading window (buy-and-hold).

    Returns (payoff_change, is_open):
      payoff_change : daily P&L per $1-long/$1-short, net of costs, 0 on flat days
      is_open       : boolean, True on days the pair holds an open position
    """
    idx = norm_trading.index
    pa = norm_trading[a].to_numpy()
    pb = norm_trading[b].to_numpy()
    pos = {t: i for i, t in enumerate(idx)}

    payoff_change = np.zeros(len(idx))
    is_open = np.zeros(len(idx), dtype=bool)

    leg_cost = cost_bps_per_leg / 10000.0  # cost per single leg, in return terms

    for tr in trades:
        ei = pos[tr.entry_date]
        xi = pos[tr.exit_date]
        if tr.direction == +1:
            p_long, p_short = pa, pb
        else:
            p_long, p_short = pb, pa

        seg_long = p_long[ei:xi + 1] / p_long[ei]
        seg_short = p_short[ei:xi + 1] / p_short[ei]
        payoff = (seg_long - 1.0) + (1.0 - seg_short)
        dchg = np.diff(payoff, prepend=payoff[0])
        dchg[0] = 0.0

        payoff_change[ei:xi + 1] += dchg
        # Transaction costs: 2 legs charged at entry, 2 legs at exit.
        payoff_change[ei] -= 2 * leg_cost
        payoff_change[xi] -= 2 * leg_cost

        is_open[ei + 1:xi + 1] = True

    return (
        pd.Series(payoff_change, index=idx, name=f"{a}-{b}"),
        pd.Series(is_open, index=idx, name=f"{a}-{b}"),
    )


def portfolio_daily_returns(
    pair_trades: dict[tuple[str, str], list[Trade]],
    norm_trading: pd.DataFrame,
    n_committed: int,
    cost_bps_per_leg: float = 0.0,
) -> pd.DataFrame:
    """Daily portfolio returns under both capital conventions, net of costs."""
    idx = norm_trading.index
    payoff_cols = []
    open_cols = []
    for (a, b), trades in pair_trades.items():
        pc, op = pair_daily_payoff(trades, norm_trading, a, b, cost_bps_per_leg)
        payoff_cols.append(pc)
        open_cols.append(op)

    if not payoff_cols:
        return pd.DataFrame(
            {"fully_invested": np.zeros(len(idx)), "committed": np.zeros(len(idx))},
            index=idx,
        )

    payoffs = pd.concat(payoff_cols, axis=1)
    opens = pd.concat(open_cols, axis=1)

    committed = payoffs.sum(axis=1) / n_committed

    n_open = opens.sum(axis=1)
    fully = payoffs.where(opens).sum(axis=1) / n_open.replace(0, np.nan)
    fully = fully.fillna(0.0)

    return pd.DataFrame({"fully_invested": fully, "committed": committed}, index=idx)


def to_monthly(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns within each calendar month."""
    return (1.0 + daily_returns).resample("ME").prod() - 1.0


if __name__ == "__main__":
    # Test: same trade, with and without costs. Cost must reduce the payoff by
    # exactly 4 * leg_cost over the round trip.
    dates = pd.bdate_range("2010-01-04", periods=40)
    P = np.linspace(1.00, 1.06, 40)
    Q = np.full(40, 1.00)
    Q[:11] = np.linspace(1.00, 1.03, 11)
    norm = pd.DataFrame({"P": P, "Q": Q}, index=dates)
    t1 = Trade("P", "Q", dates[1], dates[10], +1, 0, 0,
               float(P[1]), float(Q[1]), float(P[10]), float(Q[10]), False)

    gross = pair_daily_payoff([t1], norm, "P", "Q", cost_bps_per_leg=0.0)[0].sum()
    net = pair_daily_payoff([t1], norm, "P", "Q", cost_bps_per_leg=10.0)[0].sum()
    expected_cost = 4 * (10 / 10000.0)
    print(f"gross payoff: {gross:+.5f}")
    print(f"net payoff  : {net:+.5f}")
    print(f"cost applied: {gross - net:.5f}  (expect {expected_cost:.5f})")
    assert abs((gross - net) - expected_cost) < 1e-9
    print("OK")