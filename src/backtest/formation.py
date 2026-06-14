"""GGR formation-window machinery: screen -> normalize -> pairwise SSD -> select.

This is the first stage of the GGR two-stage engine. Given a 12-month formation
slice of the price panel, it produces a ranked list of candidate pairs that the
trading stage will then trade over the following 6 months.

Steps (GGR, Section 1):
  1. SCREEN  -- drop any stock with a missing price anywhere in the window
               (GGR's liquidity screen). Partial-coverage tickers that don't
               span the whole window fall out here automatically.
  2. NORMALIZE -- each survivor -> cumulative total-return index starting at 1.0.
               (adj_close is already dividend/split adjusted, so dividing by the
               first value yields the normalized total-return path GGR uses.)
  3. SSD     -- sum of squared deviations between every pair of normalized
               series, computed vectorized via the Gram matrix:
                 SSD_ij = sum_t (P_it - P_jt)^2
                        = ||P_i||^2 + ||P_j||^2 - 2 <P_i, P_j>
               so the full distance matrix is one X^T X away.
  4. SELECT  -- rank pairs ascending and return top 5, top 20, and the 101-120
               control set (GGR uses 101-120 to test whether profits are just a
               top-pairs / utility artifact).

The trading stage will also need, per selected pair, the in-formation spread
standard deviation (the 2-sigma entry threshold is set from it), so we return
the normalized window alongside the pairs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Pair:
    a: str
    b: str
    ssd: float


@dataclass
class FormationResult:
    normalized: pd.DataFrame          # screened + normalized window
    ranked: list[Pair]                # all unique pairs, ascending SSD
    n_screened_in: int
    n_screened_out: int

    def top(self, k: int) -> list[Pair]:
        return self.ranked[:k]

    def slice_rank(self, start: int, end: int) -> list[Pair]:
        """1-indexed inclusive rank slice, e.g. slice_rank(101, 120)."""
        return self.ranked[start - 1:end]


def screen(window: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop columns with ANY missing value in the window. Returns (kept, n_dropped)."""
    n_before = window.shape[1]
    kept = window.dropna(axis=1, how="any")
    return kept, n_before - kept.shape[1]


def normalize_to_index(window: pd.DataFrame) -> pd.DataFrame:
    """Normalize each column to start at 1.0 (cumulative total-return index)."""
    return window / window.iloc[0]


def all_pairwise_ssd(norm: pd.DataFrame) -> np.ndarray:
    """Full (n x n) matrix of pairwise SSD distances, vectorized via Gram matrix."""
    X = norm.to_numpy()
    gram = X.T @ X
    sq = np.diag(gram)
    ssd = sq[:, None] + sq[None, :] - 2.0 * gram
    np.fill_diagonal(ssd, np.inf)       # no self-pairs
    ssd[ssd < 0] = 0.0                  # floor tiny float-error negatives
    return ssd


def rank_pairs(ssd: np.ndarray, tickers: list[str]) -> list[Pair]:
    """Unique pairs (upper triangle) sorted by ascending SSD."""
    iu, ju = np.triu_indices(len(tickers), k=1)
    dists = ssd[iu, ju]
    order = np.argsort(dists, kind="stable")
    return [Pair(tickers[iu[k]], tickers[ju[k]], float(dists[k])) for k in order]


def run_formation(window: pd.DataFrame) -> FormationResult:
    """Full formation stage for one 12-month window."""
    kept, n_out = screen(window)
    if kept.shape[1] < 2:
        raise ValueError(
            f"Formation window has <2 stocks after screening "
            f"(kept {kept.shape[1]} of {window.shape[1]})."
        )
    norm = normalize_to_index(kept)
    ssd = all_pairwise_ssd(norm)
    ranked = rank_pairs(ssd, list(kept.columns))
    return FormationResult(
        normalized=norm,
        ranked=ranked,
        n_screened_in=kept.shape[1],
        n_screened_out=n_out,
    )


if __name__ == "__main__":
    # Smoke test on synthetic data with a known nearest pair.
    rng = np.random.default_rng(42)
    T = 252
    idx = pd.bdate_range("2010-01-04", periods=T)
    base = np.cumprod(1 + rng.normal(0, 0.01, T))
    df = pd.DataFrame(
        {
            "AAA": base * 50,
            "BBB": base * 50 * (1 + rng.normal(0, 0.0005, T)),  # ~identical to AAA
            "CCC": np.cumprod(1 + rng.normal(0, 0.01, T)) * 30,
            "DDD": np.cumprod(1 + rng.normal(0, 0.01, T)) * 80,
        },
        index=idx,
    )
    df.loc[df.index[10], "DDD"] = np.nan  # DDD should be screened out

    res = run_formation(df)
    print(f"screened in : {res.n_screened_in}")
    print(f"screened out: {res.n_screened_out}  (expect 1: DDD)")
    print("top pair    :", res.top(1)[0])
    assert {res.top(1)[0].a, res.top(1)[0].b} == {"AAA", "BBB"}
    assert "DDD" not in res.normalized.columns
    print("OK")