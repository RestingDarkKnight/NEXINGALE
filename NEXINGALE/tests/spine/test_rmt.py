"""Tests for the RMT cleaner (default: Ledoit-Wolf linear shrinkage).

The contract we assert is the one linear shrinkage actually guarantees, not a
stronger one it does not:

* UNCONDITIONAL  -- valid output (symmetric, unit diagonal, PSD); shrinkage in
  [0,1]; and the cleaned matrix is at least as well-conditioned as the raw
  sample matrix. Better conditioning is the property every downstream
  inverse-covariance consumer (VOLTA's correlation penalty, TDA's metric)
  relies on, and it holds on every sample.
* ON AVERAGE     -- the cleaned matrix is closer to the population matrix in
  expected Frobenius loss. This is LW's actual theorem: it is an expectation
  over noise, not a per-sample certainty, so we test the mean over many seeds.
* MONOTONE       -- noisier inputs (shorter series) shrink at least as hard.

A market-factor generator is used because real equity correlations have a
strong positive common component; that is the regime LW is built for and the
one quantflow runs in.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantflow.spine.rmt.filter import clean, clean_rie, estimate_n_factors


def _market_planted(n, t, k, seed, mkt=0.8):
    """Returns with one positive market factor + k sector factors + noise."""
    rng = np.random.default_rng(seed)
    mkt_load = np.abs(rng.normal(mkt, 0.2, (n, 1)))   # all-positive market loadings
    sec = rng.standard_normal((n, k)) * 0.6
    b = np.hstack([mkt_load, sec])
    f = rng.standard_normal((k + 1, t))
    returns = b @ f + rng.standard_normal((n, t))
    cov = b @ b.T + np.eye(n)
    d = np.sqrt(np.diag(cov))
    return returns, cov / np.outer(d, d)


def _cond(m):
    ev = np.clip(np.linalg.eigvalsh(m), 1e-12, None)
    return ev[-1] / ev[0]


def _raw_corr(returns):
    r = (returns - returns.mean(1, keepdims=True)) / returns.std(1, keepdims=True)
    return (r @ r.T) / returns.shape[1]


# ---- unconditional guarantees ----

def test_output_is_valid_correlation_matrix():
    for seed in range(5):
        returns, _ = _market_planted(60, 200, 3, seed)
        c = clean(returns).cleaned
        assert np.allclose(np.diag(c), 1.0)
        assert np.allclose(c, c.T)
        assert np.all(np.linalg.eigvalsh(c) > -1e-8)


def test_shrinkage_in_unit_interval():
    for seed in range(10):
        returns, _ = _market_planted(100, 250, 5, seed)
        assert 0.0 <= clean(returns).shrinkage <= 1.0


def test_improves_conditioning_every_sample():
    """The rock-solid guarantee: cleaned is never worse-conditioned than raw."""
    configs = [(100, 400, 4), (100, 250, 5), (100, 150, 4), (80, 240, 2),
               (120, 60, 4), (150, 500, 6)]
    for (n, t, k) in configs:
        for seed in range(8):
            returns, _ = _market_planted(n, t, k, seed)
            assert _cond(clean(returns).cleaned) <= _cond(_raw_corr(returns))


def test_handles_q_greater_than_one():
    """T < N makes the raw matrix singular; cleaned must be well-conditioned."""
    returns, _ = _market_planted(120, 60, 4, 0)  # q = 2
    res = clean(returns)
    assert np.all(np.linalg.eigvalsh(res.cleaned) > -1e-8)
    assert _cond(res.cleaned) < 1e6  # raw is ~1e13 here


# ---- average (expected-loss) guarantee ----

def test_beats_raw_against_truth_on_average():
    """LW's theorem is about expected loss; assert the mean over seeds, not each."""
    for (n, t, k) in [(100, 250, 5), (100, 150, 4)]:
        imps = []
        for seed in range(40):
            returns, true = _market_planted(n, t, k, seed)
            e = np.linalg.norm(clean(returns).cleaned - true, "fro")
            r = np.linalg.norm(_raw_corr(returns) - true, "fro")
            imps.append((r - e) / r)
        assert np.mean(imps) > 0.0, (n, t, k, np.mean(imps))


# ---- monotone shrinkage ----

def test_more_noise_more_shrinkage():
    long_returns, _ = _market_planted(80, 1000, 3, 1)
    short_returns, _ = _market_planted(80, 120, 3, 1)
    assert clean(short_returns).shrinkage >= clean(long_returns).shrinkage


# ---- Onatski diagnostic ----

def test_onatski_single_market_mode():
    ev = np.full(100, 0.8); ev[0] = 25.0
    assert estimate_n_factors(ev)[0] == 1


def test_onatski_multi_factor_cliff():
    ev = np.full(100, 0.3); ev[:4] = [22.9, 20.3, 14.3, 12.8]
    assert estimate_n_factors(ev)[0] == 4


def test_onatski_pure_noise():
    rng = np.random.default_rng(1)
    assert clean(rng.standard_normal((80, 240))).n_signal <= 2


# ---- RIE opt-in: validity only (not guaranteed optimal at fixed eta) ----

def test_rie_produces_valid_matrix():
    returns, _ = _market_planted(100, 600, 4, 0)
    c = clean_rie(returns).cleaned
    assert np.allclose(np.diag(c), 1.0)
    assert np.all(np.linalg.eigvalsh(c) > -1e-8)


def test_rie_rejects_q_ge_one():
    returns, _ = _market_planted(100, 60, 3, 0)
    with pytest.raises(ValueError):
        clean_rie(returns)


# ---- shape guards ----

def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        clean(np.zeros((10,)))
    with pytest.raises(ValueError):
        clean(np.zeros((10, 1)))
