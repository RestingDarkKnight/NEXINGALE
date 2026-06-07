"""Random Matrix Theory correlation cleaning (Stage 2 of the quantflow spine).

Sample correlation matrices built from finite time series are dominated by
noise: when the number of observations T is not vastly larger than the number of
assets N, most of the eigenvalue spectrum is statistically indistinguishable
from a random matrix. Cleaning this noise is the cheapest reliable improvement
available to any downstream stage that consumes correlations (TDA's distance
metric, VOLTA's correlation penalty).

This module ships two estimators:

* ``clean`` (default) -- **Ledoit-Wolf linear shrinkage**. Shrinks the sample
  correlation matrix toward a structured target (the identity, i.e. the average
  correlation) by a closed-form, data-driven intensity. It is proven to beat the
  sample matrix in expected Frobenius loss with no tuning and **cannot degenerate**:
  there is no bandwidth, grid, or cross-validation to misfire. This is the
  estimator everything downstream uses unless told otherwise.

* ``clean_rie`` (opt-in) -- **Ledoit-Peche rotationally-invariant estimator**
  with Onatski factor counting. Mathematically the Frobenius-optimal
  rotationally-invariant cleaner in the high-dimensional limit, but its
  bandwidth must be tuned, and (as we found empirically) no single
  cross-validation objective generalises across uses. Kept for research and for
  the regime where N is large and the bulk is dense.

References
----------
Ledoit & Wolf (2004), "A well-conditioned estimator for large-dimensional
covariance matrices," J. Multivariate Analysis. Ledoit & Peche (2011);
Bun, Bouchaud & Potters (2017); Onatski (2010).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RMTResult:
    """Output of a cleaning run, with the diagnostics worth keeping."""

    cleaned: np.ndarray              # (N, N) cleaned correlation matrix, unit diagonal
    method: str                      # "ledoit_wolf" or "rie"
    shrinkage: float                 # LW intensity in [0,1]; nan for RIE
    eigenvalues: np.ndarray          # (N,) sample-correlation eigenvalues, descending
    cleaned_eigenvalues: np.ndarray  # (N,) eigenvalues of the cleaned matrix, descending
    n_signal: int                    # Onatski ED factor count (diagnostic)
    q: float                         # N / T


def _standardize(returns: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance each row (asset). Guards against zero variance."""
    mu = returns.mean(axis=1, keepdims=True)
    sd = returns.std(axis=1, keepdims=True)
    sd = np.where(sd == 0.0, 1.0, sd)
    return (returns - mu) / sd


def _sample_corr(returns_std: np.ndarray) -> np.ndarray:
    return (returns_std @ returns_std.T) / returns_std.shape[1]


def estimate_n_factors(
    eigenvalues: np.ndarray,
    rmax: int | None = None,
    max_iter: int = 100,
) -> tuple[int, float]:
    """Onatski (2010) Eigenvalue-Difference estimator of the number of factors.

    Locates the cliff in the spectrum: above the true factor count, consecutive
    eigenvalue differences collapse to the Tracy-Widom edge scale, while genuine
    factors create order-of-magnitude larger gaps. Robust to how much variance
    the top factors absorb. Returns (k, delta) where delta is the edge threshold.
    Kept as a diagnostic feature (useful later for TDA regime descriptors).
    """
    ev = np.sort(eigenvalues)[::-1]
    n = ev.size
    if rmax is None:
        rmax = max(1, min(n // 2, 40))
    rmax = max(1, min(rmax, n - 5))
    if rmax < 1:
        gaps = ev[:-1] - ev[1:]
        return (1 if gaps.size and gaps[0] > 3.0 * np.median(gaps) else 0), 0.0

    diffs = ev[:-1] - ev[1:]
    j = rmax + 1
    delta = 0.0
    for _ in range(max_iter):
        lo, hi = j - 1, j + 4
        if hi > n:
            hi = n
            lo = max(0, hi - 5)
        y = ev[lo:hi]
        ranks = np.arange(j - 1, j - 1 + y.size, dtype=float)
        x = ranks ** (2.0 / 3.0)
        slope = 0.0 if (np.ptp(x) == 0 or y.size < 2) else np.polyfit(x, y, 1)[0]
        delta = 2.0 * abs(slope)
        above = np.where(diffs[:rmax] >= delta)[0]
        k_hat = int(above[-1] + 1) if above.size else 0
        j_new = k_hat + 1
        if j_new == j:
            return k_hat, float(delta)
        j = j_new
    above = np.where(diffs[:rmax] >= delta)[0]
    return (int(above[-1] + 1) if above.size else 0), float(delta)


def ledoit_wolf_shrinkage(returns_std: np.ndarray) -> tuple[np.ndarray, float]:
    """Closed-form Ledoit-Wolf shrinkage of the sample correlation matrix.

    Shrinks the sample correlation S toward the target F (the equicorrelation /
    identity-like target derived from the mean off-diagonal correlation):

        C_hat = (1 - rho_star) * S + rho_star * F

    The optimal intensity rho_star = pi_hat / gamma_hat is estimated directly
    from the data (Ledoit-Wolf 2003 "Honey, I shrunk the sample covariance
    matrix" applied to correlations): pi_hat measures the variance of the sample
    correlations (how noisy S is), gamma_hat measures how far S is from the
    target (the misspecification of F). More noise -> shrink more; more
    structure -> shrink less.

    Parameters
    ----------
    returns_std : (N, T) standardized returns (unit-variance rows).

    Returns
    -------
    (C_hat, rho_star) with rho_star clipped to [0, 1].
    """
    n, t = returns_std.shape
    s = (returns_std @ returns_std.T) / t  # sample correlation

    # Target F: constant correlation model. r_bar = mean off-diagonal correlation.
    off = s[np.triu_indices(n, k=1)]
    r_bar = off.mean() if off.size else 0.0
    f = np.full((n, n), r_bar)
    np.fill_diagonal(f, 1.0)

    # pi_hat: total asymptotic variance of the (sqrt-T-scaled) sample entries.
    # pi_ij = mean_t[(y_it y_jt)^2] - s_ij^2 ; summed over all i, j.
    y = returns_std
    y2 = y ** 2
    mean_y2y2 = (y2 @ y2.T) / t
    pi_hat = (mean_y2y2 - s ** 2).sum()

    # gamma_hat: squared Frobenius distance between S and the target F.
    gamma_hat = np.sum((s - f) ** 2)

    # Ledoit-Wolf optimal intensity rho* = (pi / gamma) / T, clipped to [0,1].
    # pi/gamma is the ratio of total estimation variance to misspecification;
    # dividing by T converts it to the shrinkage weight. Clipping makes it
    # impossible to degenerate.
    if gamma_hat <= 0:
        rho_star = 0.0
    else:
        rho_star = (pi_hat / gamma_hat) / t
    rho_star = float(np.clip(rho_star, 0.0, 1.0))

    c_hat = (1.0 - rho_star) * s + rho_star * f
    # numerical hygiene: symmetric, unit diagonal
    c_hat = 0.5 * (c_hat + c_hat.T)
    np.fill_diagonal(c_hat, 1.0)
    return c_hat, rho_star


def clean(returns: np.ndarray, rmax: int | None = None) -> RMTResult:
    """Clean a return matrix via Ledoit-Wolf linear shrinkage (default).

    Parameters
    ----------
    returns : (N, T) array. N assets, T observations.
    rmax : upper bound passed to the Onatski factor-count diagnostic.

    Returns
    -------
    RMTResult with ``cleaned`` the denoised correlation matrix (unit diagonal),
    guaranteed symmetric and positive semi-definite.
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2-D (N, T)")
    n, t = returns.shape
    if t < 2:
        raise ValueError("need at least 2 observations")

    r = _standardize(returns)
    s = _sample_corr(r)
    vals = np.sort(np.linalg.eigvalsh(s))[::-1]

    c_hat, rho = ledoit_wolf_shrinkage(r)
    cleaned_vals = np.sort(np.linalg.eigvalsh(c_hat))[::-1]
    k, _ = estimate_n_factors(vals, rmax=rmax)

    return RMTResult(
        cleaned=c_hat,
        method="ledoit_wolf",
        shrinkage=rho,
        eigenvalues=vals,
        cleaned_eigenvalues=cleaned_vals,
        n_signal=int(k),
        q=n / t,
    )


def _rie_shrink(eigenvalues: np.ndarray, q: float, eta: float) -> np.ndarray:
    """Ledoit-Peche RIE shrinkage of eigenvalues (Bun-Bouchaud-Potters form)."""
    lam = eigenvalues.astype(np.complex128)
    z = lam - 1j * eta
    n = lam.size
    s = (1.0 / n) * (1.0 / (z[:, None] - eigenvalues[None, :])).sum(axis=1)
    denom = np.abs(1.0 - q + q * z * s) ** 2
    return (eigenvalues / denom).real


def clean_rie(
    returns: np.ndarray,
    eta: float | None = None,
    rmax: int | None = None,
) -> RMTResult:
    """Opt-in Ledoit-Peche rotationally-invariant cleaner (research path).

    Frobenius-optimal among rotationally-invariant estimators in the high-dim
    limit, but the bandwidth ``eta`` must be supplied or tuned and no single CV
    objective generalises across uses (see the README's "mathematical
    improvements" ledger). Requires T > N (q < 1). Defaults eta to N**-1/3,
    which beat the raw matrix in the majority of T >> N configurations we tested
    but is not guaranteed optimal.
    """
    if returns.ndim != 2:
        raise ValueError("returns must be 2-D (N, T)")
    n, t = returns.shape
    if t <= n:
        raise ValueError("clean_rie requires T > N (q < 1)")
    q = n / t
    if eta is None:
        eta = n ** -(1.0 / 3.0)

    r = _standardize(returns)
    s = _sample_corr(r)
    vals_asc, vecs_asc = np.linalg.eigh(s)
    vals = vals_asc[::-1]
    vecs = vecs_asc[:, ::-1]

    xi = np.clip(_rie_shrink(vals, q, eta), 0.0, None)
    if xi.sum() > 0:
        xi = xi * (vals.sum() / xi.sum())  # preserve trace

    c_clean = (vecs * xi) @ vecs.T
    d = np.sqrt(np.diag(c_clean))
    d = np.where(d == 0.0, 1.0, d)
    c_clean = c_clean / np.outer(d, d)
    np.fill_diagonal(c_clean, 1.0)

    k, _ = estimate_n_factors(vals, rmax=rmax)
    return RMTResult(
        cleaned=c_clean,
        method="rie",
        shrinkage=float("nan"),
        eigenvalues=vals,
        cleaned_eigenvalues=np.sort(xi)[::-1],
        n_signal=int(k),
        q=q,
    )
