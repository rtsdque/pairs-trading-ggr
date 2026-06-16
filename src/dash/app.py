"""Pairs-trading research dashboard.

Presents the project's findings interactively. This is a PRESENTATION layer: it
reads saved result CSVs from data/processed/ and visualizes them -- it does NOT
re-run the backtest (that would be slow and fragile under Streamlit's rerun
model). Compute lives in src/analysis and src/backtest; this only displays.

Run:  uv run streamlit run src/dashboard/app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROC = Path(__file__).resolve().parents[2] / "data" / "processed"

st.set_page_config(page_title="Pairs Trading: A Modern Replication",
                   layout="wide", initial_sidebar_state="expanded")


# --- helpers ---------------------------------------------------------------

def load_csv(name: str, **kw) -> pd.DataFrame | None:
    path = PROC / name
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except Exception:
        return None


def returns_series(name: str) -> pd.Series | None:
    df = load_csv(name, index_col=0, parse_dates=True)
    if df is None or "committed" not in df.columns:
        return None
    return df["committed"]


# --- sidebar ---------------------------------------------------------------

st.sidebar.title("Pairs Trading")
st.sidebar.caption("Replication & extension of Gatev, Goetzmann & "
                   "Rouwenhorst (2006), 1996–2026")
section = st.sidebar.radio(
    "Section",
    ["Overview", "Data & Survivorship", "Returns & Decay",
     "Transaction Costs", "Statistical Significance"],
)
st.sidebar.markdown("---")
st.sidebar.caption("Returns shown on committed capital. Net figures use "
                   "5 bps/leg unless noted.")


# --- Overview --------------------------------------------------------------

if section == "Overview":
    st.title("Does the pairs-trading edge survive in the modern market?")
    st.markdown(
        "A faithful replication of the classic **distance-based** pairs-trading "
        "strategy, extended with a **cointegration-based** selection method, "
        "tested out-of-sample on S&P 500 stocks from 1996 to 2026 with "
        "survivorship-bias controls, realistic transaction costs, and "
        "selection-bias-corrected statistics."
    )

    st.markdown("### The finding")
    st.info(
        "**The modern pairs-trading edge, net of costs and corrected for "
        "selection, is statistically indistinguishable from zero.**\n\n"
        "The distance strategy earned significant returns before 2006 "
        "(t ≈ 2.8) but became indistinguishable from zero afterward "
        "(t ≈ 0.06) — consistent with post-publication alpha decay. "
        "Cointegration outperforms distance on every metric and is the "
        "stronger method, but once multiple testing is accounted for, "
        "neither strategy's edge survives the deflated Sharpe test at "
        "realistic trial counts."
    )

    c1, c2, c3 = st.columns(3)
    sig = load_csv("significance.csv")
    if sig is not None:
        pre = sig[(sig.period == "pre-2006") & (sig.series == "distance top20")]
        post = sig[(sig.period == "post-2006") & (sig.series == "distance top20")]
        if len(pre):
            c1.metric("Distance t-stat, pre-2006", f"{pre.iloc[0]['t']:+.2f}")
            c1.caption("significant (p < 0.01)")
        if len(post):
            c2.metric("Distance t-stat, post-2006", f"{post.iloc[0]['t']:+.2f}")
            c2.caption("not significant (p ≈ 0.95)")
    cmp = load_csv("strategy_cost_comparison.csv")
    if cmp is not None:
        c3.metric("Coint. break-even", "~18 bps/leg")
        c3.caption("vs ~12 for distance")

    st.markdown("### Why this is the right answer")
    st.markdown(
        "- The result replicates the **alpha-decay** phenomenon (McLean & "
        "Pontiff 2016) on a new strategy: a published edge erodes once known.\n"
        "- It resists the standard critique of backtests, multiple testing, "
        "by reporting the **deflated Sharpe ratio**, not just raw performance.\n"
        "- Cointegration is shown to be the better method *without* "
        "overclaiming that it restores a profitable edge."
    )


# --- Data & Survivorship ---------------------------------------------------

elif section == "Data & Survivorship":
    st.title("Data & survivorship controls")
    st.markdown(
        "Survivorship bias is the central data hazard in this kind of study. "
        "We use **point-in-time** S&P 500 membership (so each formation window "
        "sees only the stocks actually in the index then, including names later "
        "delisted) and quantify exactly how much price history the free data "
        "source recovers."
    )

    cov = load_csv("price_coverage_report.csv")
    if cov is not None:
        total = len(cov)
        usable = int(cov["status"].isin(["full", "partial"]).sum())
        empty = int((cov["status"] == "empty").sum())
        c1, c2, c3 = st.columns(3)
        c1.metric("Historical constituents", f"{total:,}")
        c2.metric("Usable price history", f"{usable:,}")
        c3.metric("No recoverable history", f"{empty:,}")
        st.caption(f"{usable/total:.0%} of constituents have usable history; "
                   f"the {empty/total:.0%} with none are overwhelmingly delisted "
                   "names (bankruptcies, acquisitions) — the survivorship gap, "
                   "quantified rather than hidden.")
        st.dataframe(cov["status"].value_counts().rename("count"),
                     use_container_width=True)
    else:
        st.warning("Coverage report not found.")

    clean = load_csv("cleaning_report.csv")
    if clean is not None:
        st.markdown("### Data cleaning")
        st.markdown(
            "A reversal-based filter removes corrupt ticks (one-day round-trip "
            "spikes) while **preserving genuine crashes** (e.g. AAPL 2000, AIG "
            "2008) — distinguished by whether the move reverses or persists."
        )
        st.dataframe(clean, use_container_width=True, height=240)


# --- Returns & Decay -------------------------------------------------------

elif section == "Returns & Decay":
    st.title("Returns and the decay of the edge")

    dist = returns_series("returns_top20.csv")
    coint = returns_series("returns_coint_top20.csv")

    if dist is not None and coint is not None:
        st.markdown("### Cumulative committed-capital return (gross)")
        cum = pd.DataFrame({
            "Distance (top 20)": (1 + dist).cumprod() - 1,
            "Cointegration (top 20)": (1 + coint).cumprod() - 1,
        })
        st.line_chart(cum, use_container_width=True)
        st.caption("Both strategies accumulate most of their gain early; the "
                   "curves flatten markedly after the mid-2000s.")

    sub = load_csv("subperiod_analysis.csv")
    if sub is not None:
        st.markdown("### Sharpe ratio by sub-period (net of 5 bps/leg)")
        q = sub[sub["split"] == "quartile"].copy()
        if len(q):
            pivot = q.pivot_table(index="period", columns="strategy",
                                  values="sharpe")
            fig = go.Figure()
            for col in pivot.columns:
                fig.add_trace(go.Bar(name=col, x=pivot.index, y=pivot[col]))
            fig.add_hline(y=0, line_color="black", line_width=1)
            fig.update_layout(barmode="group",
                              yaxis_title="Sharpe ratio",
                              legend_title_text="Strategy")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("The decay is visible: both strategies' Sharpe falls "
                       "across successive quartiles. Distance reaches zero / "
                       "negative; cointegration holds a higher but still "
                       "modest level.")
        st.dataframe(sub, use_container_width=True, height=300)


# --- Transaction Costs -----------------------------------------------------

elif section == "Transaction Costs":
    st.title("Transaction cost sensitivity")
    st.markdown(
        "Pairs-trading returns are famously cost-sensitive. Each round-trip "
        "trade touches four legs, so a cost of *c* bps/leg removes *4c/10000* "
        "in return per trade. The **break-even** — the cost at which the edge "
        "hits zero — is the practical test of viability."
    )

    cmp = load_csv("strategy_cost_comparison.csv")
    if cmp is not None:
        chart = cmp.set_index("bps_per_leg")[["distance_top20", "coint_top20"]]
        st.line_chart(chart, use_container_width=True)
        st.caption("Cointegration is higher at every cost level and crosses "
                   "zero later (~18 bps/leg) than distance (~12 bps/leg) — "
                   "roughly a 50% improvement in cost resilience.")
        st.dataframe(cmp, use_container_width=True)
        st.markdown(
            "**Context:** modern bid-ask spreads for liquid large-caps run "
            "~1–5 bps/leg, so both strategies clear costs *gross* today — but "
            "1990s-era spreads (20–50 bps) would have made both unprofitable, "
            "and the edge is thin enough that the statistical tests, not the "
            "break-even, are the binding constraint."
        )


# --- Statistical Significance ----------------------------------------------

elif section == "Statistical Significance":
    st.title("Statistical significance — the honest test")
    st.markdown(
        "Two corrections separate a real edge from a backtest artifact: "
        "**Newey-West** standard errors (for autocorrelation) and the "
        "**deflated Sharpe ratio** (for having searched many pairs to find "
        "the winners)."
    )

    sig = load_csv("significance.csv")
    if sig is not None:
        st.markdown("### Newey-West t-statistics (net 5 bps/leg)")
        show = sig[["split", "period", "series", "mean", "t", "p"]].copy()
        show["mean"] = show["mean"].map(lambda v: f"{v:+.4f}")
        show["t"] = show["t"].map(lambda v: f"{v:+.2f}")
        show["p"] = show["p"].map(lambda v: f"{v:.4f}")
        st.dataframe(show, use_container_width=True, height=320)
        st.caption("Pre-2006 returns are significant for both methods; "
                   "post-2006 neither is distinguishable from zero.")

    dsr = load_csv("deflated_sharpe.csv")
    if dsr is not None:
        st.markdown("### Deflated Sharpe ratio vs number of trials")
        pivot = dsr.pivot_table(index="n_trials", columns="series", values="dsr")
        st.line_chart(pivot, use_container_width=True)
        st.caption("DSR is the probability the true Sharpe is positive after "
                   "the multiple-testing haircut. The 0.95 line is the bar to "
                   "'survive'. At realistic trial counts (≥150), both "
                   "strategies fall well below it — the apparent edge is "
                   "largely a product of selection.")
        st.dataframe(dsr, use_container_width=True)

    st.markdown("---")
    st.markdown(
        "**Conclusion.** Net of costs and corrected for selection, neither "
        "strategy's modern edge survives. Cointegration is consistently the "
        "stronger method, but 'stronger' here means 'fails less decisively' — "
        "not 'profitable'. The honest reading: the easy, public pairs-trading "
        "edge has decayed, and what remains is statistically indistinguishable "
        "from zero."
    )