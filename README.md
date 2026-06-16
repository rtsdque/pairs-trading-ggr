# Pairs Trading: A Modern Replication and Cointegration Extension

This project asks a simple question: does pairs trading, one of the most famous
systematic trading strategies ever published, still work today?

It's a replication and extension of Gatev, Goetzmann & Rouwenhorst (2006),
*"Pairs Trading: Performance of a Relative-Value Arbitrage Rule"*, tested on S&P 500
stocks from 1996 to 2026. The original paper found a real, profitable edge. This
project rebuilds their method faithfully, carries it forward through two more
decades of data, and then asks the harder question their era couldn't: once you
account for transaction costs and the statistics of having searched through
thousands of pairs to find the good ones, is there anything left?

## What I found

The honest answer is no, and that turns out to be the interesting part.

The classic **distance-based** strategy earned real, statistically significant
returns before 2006 (Newey-West *t* ≈ 2.8). After 2006, it became
indistinguishable from zero (*t* ≈ 0.06). That timing isn't a coincidence, because when a
strategy becomes widely known and capital floods in, the edge dissappears.
The data shows that decay happening almost exactly when you'd expect it to.

I also built an extension the original paper didn't use: **cointegration-based**
selection, which tests whether a pair is *genuinely* mean-reverting rather than
just historically similar. It's the better method on every measure, because it reveals higher
returns, a more sensible ranking of pairs, and it survives higher trading costs
(a break-even around 18 basis points per leg versus 12 for the classic version).
But "better" isn't the same as "profitable." Once the returns are corrected for
multiple testing using the deflated Sharpe ratio, even cointegration's modern edge
can't be distinguished from luck.

So neither version clears the bar. The easy, public edge has decayed, and the more
rigorous method, while genuinely stronger, doesn't bring it back. That's not the
result I was looking for, but it's the truthful one, and it lines up with what a
lot of research says about how trading edges fade once they're known.

## How it works

- **Universe:** I used point-in-time S&P 500 membership, so every test only
  ever sees the stocks that were *actually* in the index at that moment, including ones that later went bankrupt or got acquired. This avoids
  survivorship bias, the trap of only studying the winners that are still around.
- **Replication:** 12-month formation windows pick pairs by distance; the next
  6 months trade them on a 2-sigma rule with one-day-delayed execution and
  committed-capital accounting, averaged across six staggered start months, which is faithful to the original paper.
- **Extension:** Engle-Granger cointegration, applied as a refinement on top
  of the distance-selected candidates.
- **Honesty Checks:** Transaction-cost sweeps with break-even analysis,
  Newey-West standard errors (which correct for the fact that monthly returns
  aren't independent), and the deflated Sharpe ratio (which corrects for having
  tested many pairs).

## A note on the data

The price data comes from a free source, which has an honest limitation: it's
missing history for about 30% of the historical constituents — almost all of them
delisted names like bankruptcies and acquisitions. Rather than hide that gap, I
measured it and documented exactly where it falls (it's worst in the 1990s and
best in recent years). Corrupt price ticks are cleaned with a filter that removes
data errors while carefully *keeping* real market crashes like Apple in 2000 or
AIG in 2008 — the difference being whether the move snaps back the next day or
sticks. The dashboard's "Data & Survivorship" section lays all of this out.

## Reproducing it

```bash
git clone https://github.com/rtsdque/pairs-trading-ggr.git
cd pairs-trading-ggr
uv sync

# Data pipeline
uv run python src/data/fetch_constituents.py
uv run python src/data/build_membership.py
uv run python src/data/fetch_prices.py      # slow — fetches ~1,200 tickers
uv run python src/data/build_panel.py
uv run python src/data/clean_panel.py

# Analysis
uv run python src/analysis/cost_sweep.py
uv run python src/analysis/strategy_cost_comparison.py
uv run python src/analysis/subperiod.py
uv run python src/analysis/significance.py
uv run python src/analysis/deflated_sharpe.py

# Dashboard
uv run streamlit run src/dash/app.py
```

## References

- Gatev, Goetzmann & Rouwenhorst (2006), *Review of Financial Studies*
- McLean & Pontiff (2016), *Journal of Finance*
- Bailey & López de Prado (2014), *Journal of Portfolio Management*

---

*Built with Python (pandas, statsmodels, scipy), uv, and Streamlit. This is an
educational research project, not investment advice.*