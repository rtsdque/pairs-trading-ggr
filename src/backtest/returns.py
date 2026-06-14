"""GGR return calculation: buy-and-hold pair payoffs -> portfolio monthly returns.

Consumes Trades (from trading.py) plus the normalized trading-window prices, and
produces monthly portfolio returns under BOTH of GGR's capital conventions.

Buy-and-hold accounting (GGR-faithful):
  At entry, commit $1 long + $1 short. Legs then DRIFT (no daily rebalancing).
  Cumulative payoff per $1-long/$1-short at day t (relative to entry day 0):
      payoff_t = (P_long_t / P_long_0 - 1) + (1 - P_short_t / P_short_0)
  The first term is the long leg's gain; the second is the short leg's gain
  (profit when the shorted leg falls). Daily payoff change is the day's P&L.

  Which leg is long depends on trade direction:
      dir = +1  -> long A, short B
      dir = -1  -> long B, short A

Two capital conventions (GGR Section 2 -- a stated project non-negotiable):
  FULLY-INVESTED  : divide total payoff by the capital in pairs that ACTUALLY
                    OPENED (return on employed capital). Flatters the strategy.
  COMMITTED-CAPITAL: divide by capital committed to ALL selected pairs for the
                    whole period, traded or not (return on committed capital).
                    More conservative and more realistic -- idle pairs still tie
                    up capital because they can trigger any day. GGR's preferred
                    honest number.

We build a daily payoff matrix (days x pairs), then:
  - fully-invested daily return = mean payoff across pairs that are OPEN that day
  - committed daily return      = sum of payoffs / N_committed_pairs
Daily returns are compounded within each calendar month to monthly returns.
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
) -> tuple[pd.Series, pd.Series]:
    """Daily payoff series for one pair over the trading window (buy-and-hold).

    Returns (payoff_change, is_open):
      payoff_change : daily P&L per $1-long/$1-short, 0 on flat days
      is_open       : boolean, True on days the pair holds an open position
    """
    idx = norm_trading.index
    pa = norm_trading[a].to_numpy()
    pb = norm_trading[b].to_numpy()
    pos = {t: i for i, t in enumerate(idx)}

    payoff_change = np.zeros(len(idx))
    is_open = np.zeros(len(idx), dtype=bool)

    for tr in trades:
        ei = pos[tr.entry_date]
        xi = pos[tr.exit_date]
        if tr.direction == +1:
            p_long, p_short = pa, pb
        else:
            p_long, p_short = pb, pa

        # cumulative payoff over the holding span, relative to entry day
        seg_long = p_long[ei:xi + 1] / p_long[ei]
        seg_short = p_short[ei:xi + 1] / p_short[ei]
        payoff = (seg_long - 1.0) + (1.0 - seg_short)
        dchg = np.diff(payoff, prepend=payoff[0])
        dchg[0] = 0.0  # entry day: position just opened, no P&L yet

        payoff_change[ei:xi + 1] += dchg
        # position is "open" from the day after entry through exit
        is_open[ei + 1:xi + 1] = True

    return (
        pd.Series(payoff_change, index=idx, name=f"{a}-{b}"),
        pd.Series(is_open, index=idx, name=f"{a}-{b}"),
    )


def portfolio_daily_returns(
    pair_trades: dict[tuple[str, str], list[Trade]],
    norm_trading: pd.DataFrame,
    n_committed: int,
) -> pd.DataFrame:
    """Daily portfolio returns under both capital conventions.

    pair_trades : {(a, b): [Trade, ...]} for the selected pairs
    n_committed : number of pairs committed (e.g. 20), used as the
                  committed-capital denominator
    Returns a DataFrame indexed by date with columns
    ['fully_invested', 'committed'].
    """
    idx = norm_trading.index
    payoff_cols = []
    open_cols = []
    for (a, b), trades in pair_trades.items():
        pc, op = pair_daily_payoff(trades, norm_trading, a, b)
        payoff_cols.append(pc)
        open_cols.append(op)

    if not payoff_cols:
        return pd.DataFrame(
            {"fully_invested": np.zeros(len(idx)), "committed": np.zeros(len(idx))},
            index=idx,
        )

    payoffs = pd.concat(payoff_cols, axis=1)  # days x pairs
    opens = pd.concat(open_cols, axis=1)

    # Committed: total payoff spread over ALL committed pairs every day.
    committed = payoffs.sum(axis=1) / n_committed

    # Fully-invested: average payoff over pairs OPEN that day (0 if none open).
    n_open = opens.sum(axis=1)
    fully = payoffs.where(opens).sum(axis=1) / n_open.replace(0, np.nan)
    fully = fully.fillna(0.0)

    return pd.DataFrame({"fully_invested": fully, "committed": committed}, index=idx)


def to_monthly(daily_returns: pd.DataFrame) -> pd.DataFrame:
    """Compound daily returns within each calendar month."""
    return (1.0 + daily_returns).resample("ME").prod() - 1.0


if __name__ == "__main__":
    # Construct 2 pairs over ~2 months to test both denominators deterministically.
    dates = pd.bdate_range("2010-01-04", periods=40)

    # Pair 1 (P-Q): a winning convergence trade open days 1..10.
    P = np.linspace(1.00, 1.06, 40)             # drifts up
    Q = np.full(40, 1.00)
    Q[:11] = np.linspace(1.00, 1.03, 11)        # rises then flat -> short loses early
    # Pair 2 (R-S): never trades (stays flat) -> tests committed vs fully-invested.
    R = np.full(40, 1.00)
    S = np.full(40, 1.00)

    norm = pd.DataFrame({"P": P, "Q": Q, "R": R, "S": S}, index=dates)

    t1 = Trade("P", "Q", dates[1], dates[10], +1, 0, 0,
               float(P[1]), float(Q[1]), float(P[10]), float(Q[10]), False)

    pair_trades = {("P", "Q"): [t1], ("R", "S"): []}  # R-S committed but idle

    daily = portfolio_daily_returns(pair_trades, norm, n_committed=2)
    monthly = to_monthly(daily)

    print("Monthly returns (2 pairs, 1 trades, 1 idle):")
    print(monthly.round(5).to_string())

    # Committed return should be ~half the fully-invested return in the trading
    # month, because the idle R-S pair doubles the committed denominator.
    jan = monthly.loc["2010-01-31"]
    print(f"\nJan fully-invested: {jan['fully_invested']:+.5f}")
    print(f"Jan committed     : {jan['committed']:+.5f}")
    ratio = jan["committed"] / jan["fully_invested"]
    print(f"committed/fully ratio: {ratio:.3f}  (expect ~0.5: 1 of 2 pairs active)")
    assert 0.45 < ratio < 0.55, "committed-capital denominator not behaving"
    print("OK")