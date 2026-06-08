"""
Nexingale-D : single-file derivatives pipeline (generated, runnable standalone)
================================================================================
GENERATED from the Nexingale package. Edit the package, then regenerate.
Pipeline: signal engines -> RMT clean -> TDA regime -> VOLTA select -> risk review
Dependencies: numpy, dimod, dwave-neal, ripser
Run: python Nexingale_d_standalone.py   (runs the demo at the bottom)
The agent layer is deterministic by design; a real LLM overlay slots into
risk.review's llm_review hook.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable
import numpy as np
import dimod
from neal import SimulatedAnnealingSampler
from ripser import ripser


# ==============================================================================
# RMT cleaning (Stage 2)
# ==============================================================================

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

# ==============================================================================
# TDA regime detection (Stage 3)
# ==============================================================================

"""Stage 3: topological regime detection.

Takes the cleaned correlation matrix from Stage 2, turns it into a distance
matrix, computes persistent homology with ripser, and maps the topological
features to a market regime. The intuition: different regimes (trending,
range-bound, rotating, stressed) leave characteristically different topological
signatures in the correlation structure -- and these often shift *before*
price/volatility-based detectors react.

This v1 uses a transparent rule-based classifier over interpretable persistence
features rather than a trained model, so you can read why a regime was assigned.
A trained classifier is the documented upgrade (see MATH_LEDGER).
"""





class Regime(str, Enum):
    TREND = "trend"                # strong common mode, few loops
    SECTOR_ROTATION = "sector_rotation"  # rich H1/H2 structure
    RANGE = "range"                # moderate, stable structure
    STRESS = "stress"              # correlations collapse toward 1 (everything moves together)


@dataclass
class RegimeResult:
    regime: Regime
    confidence: float
    features: dict
    note: str


def _persistence_features(corr: np.ndarray, max_dim: int = 2) -> dict:
    """Persistent homology features from a cleaned correlation matrix.

    Distance d_ij = sqrt(1 - corr_ij^2) is a proper metric for a valid
    correlation matrix. We compute H0/H1/H2 persistence and summarise each
    dimension by its total persistence (sum of death-birth), which measures how
    much topological structure of that order is present.
    """
    n = corr.shape[0]
    dist = np.sqrt(np.clip(1.0 - corr ** 2, 0.0, None))
    np.fill_diagonal(dist, 0.0)

    dgms = ripser(dist, maxdim=max_dim, distance_matrix=True)["dgms"]

    feats = {}
    for dim, dgm in enumerate(dgms):
        if len(dgm) == 0:
            feats[f"energy_H{dim}"] = 0.0
            feats[f"count_H{dim}"] = 0
            continue
        finite = dgm[np.isfinite(dgm[:, 1])]
        persist = (finite[:, 1] - finite[:, 0]) if len(finite) else np.array([0.0])
        feats[f"energy_H{dim}"] = float(persist.sum())
        feats[f"count_H{dim}"] = int((persist > 0.05).sum())  # non-trivial features

    # average correlation: a fast proxy for "everything moving together"
    off = corr[np.triu_indices(n, k=1)]
    feats["avg_corr"] = float(np.abs(off).mean())
    # share of variance in the top mode (market mode dominance)
    ev = np.sort(np.linalg.eigvalsh(corr))[::-1]
    feats["top_mode_share"] = float(ev[0] / ev.sum())
    return feats


def detect_regime(cleaned_corr: np.ndarray) -> RegimeResult:
    """Classify the market regime from the cleaned correlation matrix.

    Rule-based over interpretable features. The thresholds are sensible defaults;
    they are exactly the kind of thing a trained classifier would replace once we
    have labelled historical regimes.
    """
    f = _persistence_features(cleaned_corr)

    avg_corr = f["avg_corr"]
    top_share = f["top_mode_share"]
    h1 = f["energy_H1"]
    h2 = f["energy_H2"]

    # decision logic, most extreme first
    if avg_corr > 0.55 and top_share > 0.40:
        regime, conf = Regime.STRESS, min(0.95, avg_corr)
        note = "correlations collapsing toward 1; market moving as one block"
    elif h2 > 0.25 or (h1 > 0.6 and f["count_H1"] >= 2):
        regime, conf = Regime.SECTOR_ROTATION, 0.6 + min(0.3, h2)
        note = "rich higher-order topology; sectors decoupling and rotating"
    elif top_share > 0.30 and h1 < 0.4:
        regime, conf = Regime.TREND, 0.55 + min(0.35, top_share)
        note = "dominant common mode, little loop structure; directional regime"
    else:
        regime, conf = Regime.RANGE, 0.55
        note = "moderate, stable structure; range-bound"

    return RegimeResult(regime=regime, confidence=round(float(conf), 3),
                        features=f, note=note)

# ==============================================================================
# Candidate schema
# ==============================================================================

"""Common types for the derivatives branch.

A ``Candidate`` is the contract between the signal engines (which propose trades)
and VOLTA (which selects a subset). Every engine, no matter how different its
logic, emits the same shape: a structure with an estimated edge, the capital it
ties up, its Greek exposures, and tags VOLTA uses for diversification penalties.

This is the piece the v2 walkthrough hand-waved -- the C01..C12 candidates
"appeared". Here they are produced explicitly, with the fields VOLTA's QUBO needs.
"""




class Structure(str, Enum):
    """The trade structures the derivatives branch knows how to build."""

    LONG_CALL = "long_call"
    LONG_PUT = "long_put"
    SHORT_STRANGLE = "short_strangle"
    IRON_CONDOR = "iron_condor"
    CALENDAR = "calendar"
    LONG_FUTURE = "long_future"
    SHORT_FUTURE = "short_future"


class StrategyType(str, Enum):
    """High-level strategy family, used for type-diversity penalties in VOLTA."""

    VOL_CRUSH = "vol_crush"        # short vega into an event, harvest IV collapse
    DIRECTIONAL = "directional"    # futures with a directional thesis
    THETA_HARVEST = "theta_harvest"  # range-bound premium selling
    FLOW_FOLLOW = "flow_follow"    # ride FII/DII positioning
    HEDGE = "hedge"                # protective, negative-edge-on-purpose


@dataclass(frozen=True)
class OptionLeg:
    """One leg of an option structure."""

    underlying: str
    expiry: str            # e.g. "2026-06-26"
    strike: float
    is_call: bool
    is_long: bool          # True = bought, False = sold
    lots: int = 1


@dataclass
class Candidate:
    """A proposed trade, the unit VOLTA selects over.

    Greeks are net for the whole structure, per lot, signed in the usual
    convention (long option -> positive vega/gamma; short -> negative).
    """

    id: str
    underlying: str
    structure: Structure
    strategy_type: StrategyType
    sector: str

    edge: float            # expected edge as a fraction of capital, e.g. 0.023 = 2.3%
    capital: float         # capital tied up (margin or premium), in rupees
    vega: float            # net vega exposure
    delta: float           # net delta exposure
    theta: float           # net theta (per day); positive = collects time decay

    legs: list[OptionLeg] = field(default_factory=list)
    note: str = ""         # human-readable rationale, for the reasoning trace

    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError(f"candidate {self.id}: capital must be positive")
        if not (-1.0 <= self.edge <= 1.0):
            raise ValueError(f"candidate {self.id}: edge {self.edge} out of [-1,1]")

# ==============================================================================
# Signal engines (Stage 1)
# ==============================================================================

"""Signal engines for the derivatives branch.

Four engines, each a pure function from market inputs to a list of Candidates.
They are deliberately simple and rule-based: the point is that you can read the
logic, see exactly what each one trades, and tune the thresholds to your own
book. They are the derivatives analog of AlphaCrafter's Miner -- except they mine
option/future *structures*, not cross-sectional equity factors.

Each engine is independent; ``generate_all`` runs them and concatenates. VOLTA
then selects a subset. The engines do NOT size or risk-check -- that is VOLTA's
and the agent layer's job. They only propose.

Market input shape
------------------
Each engine takes a list of ``dict`` rows describing instruments. The fields used
are documented per engine. Missing optional fields fall back to neutral defaults.
This keeps the engines decoupled from any specific data vendor; the data adapter
layer (spine/data/adapters) is responsible for producing these dicts from Kite /
broker snapshots.
"""




# ---------------------------------------------------------------------------
# Engine 1: event volatility crush
# ---------------------------------------------------------------------------
def vol_crush(
    events: list[dict],
    iv_percentile_floor: float = 0.60,
    min_edge: float = 0.010,
) -> list[Candidate]:
    """Sell elevated implied vol into a scheduled event, harvest the post-event crush.

    Logic: when an underlying has a known event (earnings, policy) and its option
    IV sits in a high percentile of its own history, the market is over-paying for
    the event. A short strangle collects that premium; the crush after the event
    realises the edge. We only fire when IV is genuinely elevated.

    Each event dict needs:
        underlying, sector, expiry, spot,
        iv_percentile (0-1), call_strike, put_strike, est_premium, est_margin,
        vega_per_lot (negative for the short structure), days_to_event
    """
    out: list[Candidate] = []
    for i, e in enumerate(events):
        ivp = e.get("iv_percentile", 0.0)
        if ivp < iv_percentile_floor:
            continue  # not enough premium to be worth the tail risk
        # crude edge model: the further IV is above the floor, the larger the
        # expected crush, scaled by premium-to-margin.
        premium, margin = e["est_premium"], e["est_margin"]
        edge = (ivp - iv_percentile_floor) * (premium / margin)
        if edge < min_edge:
            continue
        legs = [
            OptionLeg(e["underlying"], e["expiry"], e["call_strike"], True, False),
            OptionLeg(e["underlying"], e["expiry"], e["put_strike"], False, False),
        ]
        out.append(Candidate(
            id=f"VC{i:02d}",
            underlying=e["underlying"],
            structure=Structure.SHORT_STRANGLE,
            strategy_type=StrategyType.VOL_CRUSH,
            sector=e.get("sector", "unknown"),
            edge=round(edge, 4),
            capital=margin,
            vega=e.get("vega_per_lot", -300.0),
            delta=e.get("delta_per_lot", 0.0),
            theta=e.get("theta_per_lot", abs(premium) * 0.03),
            legs=legs,
            note=f"IV at {ivp:.0%} pctile, {e.get('days_to_event','?')}d to event",
        ))
    return out


# ---------------------------------------------------------------------------
# Engine 2: directional futures
# ---------------------------------------------------------------------------
def directional_futures(
    instruments: list[dict],
    min_score: float = 0.5,
    min_edge: float = 0.010,
) -> list[Candidate]:
    """Take a futures position when fundamental + technical signals agree.

    Logic: combine a fundamental score (e.g. from SOVRENN/pattern memory) and a
    technical score (trend/momentum) into a conviction. Long if both positive,
    short if both negative; skip when they disagree (no edge in a coin flip).

    Each instrument dict needs:
        underlying, sector, expiry,
        fundamental_score (-1..1), technical_score (-1..1),
        est_margin, est_move (expected % move), delta_per_lot
    """
    out: list[Candidate] = []
    for i, ins in enumerate(instruments):
        fs, ts = ins.get("fundamental_score", 0.0), ins.get("technical_score", 0.0)
        if fs * ts <= 0:
            continue  # signals disagree or one is flat -> no conviction
        score = (abs(fs) + abs(ts)) / 2.0
        if score < min_score:
            continue
        long = fs > 0
        edge = score * abs(ins.get("est_move", 0.02))
        if edge < min_edge:
            continue
        out.append(Candidate(
            id=f"DF{i:02d}",
            underlying=ins["underlying"],
            structure=Structure.LONG_FUTURE if long else Structure.SHORT_FUTURE,
            strategy_type=StrategyType.DIRECTIONAL,
            sector=ins.get("sector", "unknown"),
            edge=round(edge, 4),
            capital=ins["est_margin"],
            vega=0.0,
            delta=ins.get("delta_per_lot", 75.0) * (1 if long else -1),
            theta=0.0,
            note=f"fund={fs:+.2f} tech={ts:+.2f} -> {'LONG' if long else 'SHORT'}",
        ))
    return out


# ---------------------------------------------------------------------------
# Engine 3: theta harvest (index iron condors in calm regimes)
# ---------------------------------------------------------------------------
def theta_harvest(
    indices: list[dict],
    vix_ceiling: float = 18.0,
    min_edge: float = 0.008,
) -> list[Candidate]:
    """Sell index iron condors when volatility is low and range-bound.

    Logic: in a calm, range-bound regime, an iron condor collects theta with
    defined risk. We only fire when VIX is below a ceiling (high VIX means the
    range can break and the condor loses). Wings are placed at +/- a multiple of
    the expected move so the short strikes sit outside one standard deviation.

    Each index dict needs:
        underlying, expiry, spot, vix, expected_move (points),
        wing_width (points), est_credit, est_margin
    """
    out: list[Candidate] = []
    for i, ix in enumerate(indices):
        vix = ix.get("vix", 99.0)
        if vix > vix_ceiling:
            continue  # too volatile for a range trade
        spot, em, ww = ix["spot"], ix["expected_move"], ix["wing_width"]
        credit, margin = ix["est_credit"], ix["est_margin"]
        edge = credit / margin
        if edge < min_edge:
            continue
        # short strikes outside 1 sigma, long wings beyond
        put_short, call_short = spot - em, spot + em
        put_long, call_long = put_short - ww, call_short + ww
        legs = [
            OptionLeg(ix["underlying"], ix["expiry"], put_long, False, True),
            OptionLeg(ix["underlying"], ix["expiry"], put_short, False, False),
            OptionLeg(ix["underlying"], ix["expiry"], call_short, True, False),
            OptionLeg(ix["underlying"], ix["expiry"], call_long, True, True),
        ]
        out.append(Candidate(
            id=f"TH{i:02d}",
            underlying=ix["underlying"],
            structure=Structure.IRON_CONDOR,
            strategy_type=StrategyType.THETA_HARVEST,
            sector="Index",
            edge=round(edge, 4),
            capital=margin,
            vega=ix.get("vega_per_lot", -500.0),
            delta=ix.get("delta_per_lot", 0.0),
            theta=ix.get("theta_per_lot", credit * 0.04),
            legs=legs,
            note=f"VIX {vix:.1f}, condor {put_long:.0f}/{put_short:.0f}/"
                 f"{call_short:.0f}/{call_long:.0f}",
        ))
    return out


# ---------------------------------------------------------------------------
# Engine 4: FII / DII flow follow
# ---------------------------------------------------------------------------
def flow_follow(
    flow_signals: list[dict],
    min_abs_flow: float = 1500.0,   # rupees crore
    min_edge: float = 0.008,
) -> list[Candidate]:
    """Follow persistent institutional flow on the index futures.

    Logic: sustained FII buying (or selling) tends to push the index in the same
    direction over days. When cumulative flow over the window exceeds a threshold
    and is consistent in sign, take a directional index future with it.

    Each flow dict needs:
        underlying, expiry, cum_fii (crore, signed), cum_dii (crore, signed),
        consistency (0-1, fraction of days same sign), est_margin, est_move,
        delta_per_lot
    """
    out: list[Candidate] = []
    for i, f in enumerate(flow_signals):
        net = f.get("cum_fii", 0.0)
        if abs(net) < min_abs_flow:
            continue
        consistency = f.get("consistency", 0.0)
        edge = consistency * abs(f.get("est_move", 0.015))
        if edge < min_edge:
            continue
        long = net > 0
        out.append(Candidate(
            id=f"FF{i:02d}",
            underlying=f["underlying"],
            structure=Structure.LONG_FUTURE if long else Structure.SHORT_FUTURE,
            strategy_type=StrategyType.FLOW_FOLLOW,
            sector="Index",
            edge=round(edge, 4),
            capital=f["est_margin"],
            vega=0.0,
            delta=f.get("delta_per_lot", 75.0) * (1 if long else -1),
            theta=0.0,
            note=f"FII {net:+.0f}cr, {consistency:.0%} consistent -> "
                 f"{'LONG' if long else 'SHORT'}",
        ))
    return out


# ---------------------------------------------------------------------------
def generate_all(market: dict) -> list[Candidate]:
    """Run every engine on a market snapshot and concatenate the candidates.

    ``market`` is a dict with keys ``events``, ``instruments``, ``indices``,
    ``flows`` -- each a list of the row-dicts the corresponding engine expects.
    Missing keys are treated as empty.
    """
    return (
        vol_crush(market.get("events", []))
        + directional_futures(market.get("instruments", []))
        + theta_harvest(market.get("indices", []))
        + flow_follow(market.get("flows", []))
    )

# ==============================================================================
# VOLTA optimizer (Stage 4)
# ==============================================================================

"""Stage 4: VOLTA -- Ising/QUBO portfolio selection over candidates.

Given a list of candidates, select the subset that maximises total edge subject
to capital, vega, sector-concentration, strategy-type, and correlation penalties.
The problem is combinatorial (2^N subsets); we formulate it as a QUBO and solve
with simulated annealing (D-Wave neal). Edges are conditioned on the regime from
Stage 3 -- in a range regime theta-harvest is favoured, in a trend regime
directional trades are, and so on.

The QUBO objective over binary x_i in {0,1}:

    H(x) = - sum_i e_i(regime) x_i
           + lambda_cap  * (sum_i cap_i x_i - C_max)^2        [capital]
           + lambda_vega * (sum_i vega_i x_i)^2               [net vega -> 0]
           + lambda_sec  * sum_s (count in sector s - K_s)^2  [concentration]
           + lambda_type * sum_t (count of type t - K_t)^2    [type diversity]
           + lambda_corr * sum_{i<j} rho_ij^2 x_i x_j         [correlation]
"""






# regime -> per-strategy edge multipliers. >1 favours, <1 penalises.
REGIME_TILT: dict[str, dict[StrategyType, float]] = {
    "trend": {
        StrategyType.DIRECTIONAL: 1.4, StrategyType.FLOW_FOLLOW: 1.3,
        StrategyType.THETA_HARVEST: 0.7, StrategyType.VOL_CRUSH: 0.9,
    },
    "range": {
        StrategyType.THETA_HARVEST: 1.4, StrategyType.VOL_CRUSH: 1.2,
        StrategyType.DIRECTIONAL: 0.8, StrategyType.FLOW_FOLLOW: 0.8,
    },
    "sector_rotation": {
        StrategyType.THETA_HARVEST: 1.2, StrategyType.DIRECTIONAL: 1.1,
        StrategyType.VOL_CRUSH: 1.0, StrategyType.FLOW_FOLLOW: 1.0,
    },
    "stress": {  # in stress, only hedges and high-conviction directional survive
        StrategyType.DIRECTIONAL: 0.9, StrategyType.HEDGE: 1.5,
        StrategyType.THETA_HARVEST: 0.4, StrategyType.VOL_CRUSH: 0.3,
        StrategyType.FLOW_FOLLOW: 0.7,
    },
}


@dataclass
class VoltaConfig:
    capital_max: float = 1_500_000.0   # rupees available
    vega_max: float = 500.0            # |net vega| target ceiling
    sector_cap: int = 2                # max trades per sector
    type_cap: int = 3                  # max trades per strategy type
    lambda_cap: float = 1.0
    lambda_vega: float = 1.0
    lambda_sec: float = 1.0
    lambda_type: float = 0.5
    lambda_corr: float = 1.0
    num_reads: int = 2000
    seed: int = 0


@dataclass
class VoltaResult:
    selected: list[Candidate]
    energy: float
    total_capital: float
    net_vega: float
    net_delta: float
    total_edge: float
    note: str
    all_candidates: list[Candidate] = field(default_factory=list)


def _regime_edge(c: Candidate, regime: str) -> float:
    tilt = REGIME_TILT.get(regime, {}).get(c.strategy_type, 1.0)
    return c.edge * tilt


def _normalise_scales(candidates: list[Candidate], cfg: VoltaConfig):
    """Penalty terms must be commensurate with edge or one term dominates.

    We rescale capital and vega to units of their caps so the squared penalties
    are O(1) per violation, matching the O(edge) linear rewards. Returns scaled
    capital and vega arrays plus the scaled cap targets.
    """
    cap = np.array([c.capital for c in candidates]) / cfg.capital_max
    vega = np.array([c.vega for c in candidates]) / max(cfg.vega_max, 1.0)
    return cap, vega


def _auto_penalty_scale(candidates: list[Candidate], regime: str) -> float:
    """Scale for penalty weights so constraints dominate the edge reward.

    The linear edge reward has magnitude ~ sum_i |e_i|. If penalty weights are
    comparable to a single edge, the optimiser will happily breach a constraint
    to collect a few more edges. We scale penalties to the *total* edge on the
    table so a normalised (O(1)) constraint violation always costs more than any
    edge it could buy.
    """
    total_edge = sum(abs(_regime_edge(c, regime)) for c in candidates)
    return max(1.0, total_edge * 2.0)


def build_qubo(
    candidates: list[Candidate],
    regime: str,
    corr: np.ndarray | None,
    cfg: VoltaConfig,
) -> dimod.BinaryQuadraticModel:
    """Construct the QUBO BinaryQuadraticModel for the selection problem."""
    n = len(candidates)
    bqm = dimod.BinaryQuadraticModel("BINARY")
    cap, vega = _normalise_scales(candidates, cfg)
    scale = _auto_penalty_scale(candidates, regime)
    lam_cap = cfg.lambda_cap * scale
    lam_vega = cfg.lambda_vega * scale
    lam_sec = cfg.lambda_sec * scale
    lam_type = cfg.lambda_type * scale
    lam_corr = cfg.lambda_corr * scale

    # linear reward: -edge (we minimise H, so negative edge maximises edge)
    for i, c in enumerate(candidates):
        bqm.add_variable(i, -_regime_edge(c, regime))

    # capital penalty: (sum cap_i x_i - 1)^2 with cap in units of C_max
    for i in range(n):
        bqm.add_linear(i, lam_cap * (cap[i] ** 2 - 2 * cap[i]))
        for j in range(i + 1, n):
            bqm.add_quadratic(i, j, 2 * lam_cap * cap[i] * cap[j])

    # vega penalty: (sum vega_i x_i)^2  (drive net vega toward 0)
    for i in range(n):
        bqm.add_linear(i, lam_vega * vega[i] ** 2)
        for j in range(i + 1, n):
            bqm.add_quadratic(i, j, 2 * lam_vega * vega[i] * vega[j])

    # sector concentration
    sectors = {}
    for i, c in enumerate(candidates):
        sectors.setdefault(c.sector, []).append(i)
    for members in sectors.values():
        for a in range(len(members)):
            i = members[a]
            bqm.add_linear(i, lam_sec * (1 - 2 * cfg.sector_cap))
            for b in range(a + 1, len(members)):
                bqm.add_quadratic(i, members[b], 2 * lam_sec)

    # strategy-type diversity
    types = {}
    for i, c in enumerate(candidates):
        types.setdefault(c.strategy_type, []).append(i)
    for members in types.values():
        for a in range(len(members)):
            i = members[a]
            bqm.add_linear(i, lam_type * (1 - 2 * cfg.type_cap))
            for b in range(a + 1, len(members)):
                bqm.add_quadratic(i, members[b], 2 * lam_type)

    # correlation penalty
    if corr is not None:
        for i in range(n):
            for j in range(i + 1, n):
                rho2 = float(corr[i, j]) ** 2
                if rho2 > 0:
                    bqm.add_quadratic(i, j, lam_corr * rho2)

    return bqm


def solve(
    candidates: list[Candidate],
    regime: str = "range",
    corr: np.ndarray | None = None,
    cfg: VoltaConfig | None = None,
) -> VoltaResult:
    """Select the optimal subset of candidates via simulated annealing."""
    cfg = cfg or VoltaConfig()
    if not candidates:
        return VoltaResult([], 0.0, 0.0, 0.0, 0.0, 0.0, "no candidates")

    bqm = build_qubo(candidates, regime, corr, cfg)
    sampler = SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=cfg.num_reads, seed=cfg.seed)
    best = sampleset.first.sample

    chosen = [candidates[i] for i in range(len(candidates)) if best.get(i, 0) == 1]

    # Hard feasibility repair. The QUBO capital term is a *soft* penalty and
    # cannot strictly enforce an inequality; a capital cap, however, is a hard
    # limit. So if the annealed set breaches capital, greedily drop the lowest
    # edge-per-capital trades until it fits. This guarantees the output never
    # violates the cap, regardless of penalty calibration.
    if cfg.capital_max > 0:
        while chosen and sum(c.capital for c in chosen) > cfg.capital_max:
            drop = min(chosen, key=lambda c: _regime_edge(c, regime) / max(c.capital, 1.0))
            chosen.remove(drop)

    # if the optimiser selected nothing (penalties too tight), fall back to the
    # single best-edge candidate that fits capital -- never return empty silently
    if not chosen:
        affordable = [c for c in candidates if c.capital <= cfg.capital_max]
        if affordable:
            chosen = [max(affordable, key=lambda c: _regime_edge(c, regime))]

    total_cap = sum(c.capital for c in chosen)
    net_vega = sum(c.vega for c in chosen)
    net_delta = sum(c.delta for c in chosen)
    total_edge = sum(_regime_edge(c, regime) * c.capital for c in chosen)

    return VoltaResult(
        selected=chosen,
        energy=float(sampleset.first.energy),
        total_capital=total_cap,
        net_vega=net_vega,
        net_delta=net_delta,
        total_edge=total_edge,
        note=f"regime={regime}, selected {len(chosen)}/{len(candidates)}",
        all_candidates=candidates,
    )

# ==============================================================================
# Risk agent (Stage 5)
# ==============================================================================

"""Stage 5: lean rationality-driven agent layer (derivatives branch).

The derivatives branch deliberately does NOT use a bull/bear debate committee.
AlphaCrafter's evidence is that role-playing agents add behavioural noise and
latency; for fast, Greek-exposed trading we want disciplined, auditable rules.
This layer is therefore deterministic: it applies the same checks a competent
risk manager would, produces a structured decision, and can veto or size down.

A hook (``llm_review``) is provided where a real LLM call would slot in to add
qualitative judgement (news context, narrative sanity) on top of the hard rules.
It is a no-op by default because this environment makes no external calls; wire
it to the Claude API when deploying.
"""





@dataclass
class RiskLimits:
    capital_max: float = 1_500_000.0
    vega_abs_max: float = 500.0
    delta_abs_max: float = 150.0
    max_single_capital_frac: float = 0.30   # no trade > 30% of capital
    max_per_sector: int = 2


@dataclass
class RiskDecision:
    approved: list[Candidate]
    rejected: list[tuple[Candidate, str]]   # (candidate, reason)
    flags: list[str]
    passed: bool
    note: str
    trace: dict = field(default_factory=dict)


def _no_llm(_plan: VoltaResult) -> list[str]:
    """Default LLM hook: no external call available. Returns no extra flags."""
    return []


def review(
    volta_result: VoltaResult,
    limits: RiskLimits | None = None,
    llm_review: Callable[[VoltaResult], list[str]] = _no_llm,
) -> RiskDecision:
    """Apply hard risk rules to VOLTA's selection; veto or size-down as needed.

    Returns an auditable decision: which trades passed, which were rejected and
    why, and any soft flags for monitoring. This is the final gate before a plan
    would go to (dry-run) execution.
    """
    limits = limits or RiskLimits()
    selected = list(volta_result.selected)
    approved: list[Candidate] = []
    rejected: list[tuple[Candidate, str]] = []
    flags: list[str] = []

    # 1. per-trade capital concentration
    for c in selected:
        if c.capital > limits.max_single_capital_frac * limits.capital_max:
            rejected.append((c, f"single-trade capital {c.capital:.0f} exceeds "
                                f"{limits.max_single_capital_frac:.0%} of book"))
        else:
            approved.append(c)

    # 2. portfolio-level Greek and capital checks on the approved set
    total_cap = sum(c.capital for c in approved)
    net_vega = sum(c.vega for c in approved)
    net_delta = sum(c.delta for c in approved)

    # size-down loop: while a portfolio limit is breached, drop the trade that
    # best relieves the *binding* constraint per unit of edge given up. De-risking
    # must target the constraint actually breached -- dropping a low-vega trade
    # does nothing for a vega breach.
    def state(cands):
        return (sum(c.capital for c in cands),
                sum(c.vega for c in cands),
                sum(c.delta for c in cands))

    total_cap, net_vega, net_delta = state(approved)

    def breach_kind():
        if total_cap > limits.capital_max:
            return "capital"
        if abs(net_vega) > limits.vega_abs_max:
            return "vega"
        if abs(net_delta) > limits.delta_abs_max:
            return "delta"
        return None

    while approved:
        kind = breach_kind()
        if kind is None:
            break
        # relief = how much this trade reduces the binding quantity;
        # cost = edge given up. Drop the highest relief-per-edge trade.
        def relief(c):
            if kind == "capital":
                return c.capital
            if kind == "vega":
                # only trades on the same side as the breach relieve it
                return c.vega if net_vega > 0 else -c.vega
            return c.delta if net_delta > 0 else -c.delta
        # candidates that actually help (positive relief); if none, fall back to
        # the lowest-edge trade to make progress
        helpers = [c for c in approved if relief(c) > 0]
        pool = helpers if helpers else approved
        drop = max(pool, key=lambda c: relief(c) / max(c.edge * c.capital, 1.0))
        approved.remove(drop)
        rejected.append((drop, f"sized down to relieve {kind} limit"))
        total_cap, net_vega, net_delta = state(approved)

    # 3. soft monitoring flags
    if abs(net_vega) > 0.8 * limits.vega_abs_max:
        flags.append(f"net vega {net_vega:.0f} near limit; watch vol expansion")
    sectors: dict[str, int] = {}
    for c in approved:
        sectors[c.sector] = sectors.get(c.sector, 0) + 1
    for s, k in sectors.items():
        if k > limits.max_per_sector:
            flags.append(f"sector {s} has {k} trades (cap {limits.max_per_sector})")

    # 4. optional qualitative LLM overlay (no-op here)
    flags.extend(llm_review(volta_result))

    passed = len(approved) > 0
    note = (f"approved {len(approved)}/{len(selected)}; "
            f"cap={total_cap:.0f} vega={net_vega:.0f} delta={net_delta:.0f}")

    return RiskDecision(
        approved=approved,
        rejected=rejected,
        flags=flags,
        passed=passed,
        note=note,
        trace={"net_vega": net_vega, "net_delta": net_delta,
               "total_capital": total_cap},
    )

# ==============================================================================
# Pipeline orchestrator
# ==============================================================================

"""Derivatives-branch pipeline orchestrator.

Chains the stages into one call and returns a structured result carrying every
stage's output, so the whole run is auditable end to end:

    candidates (engines)  ->  RMT clean (corr over selected underlyings)
                          ->  TDA regime
                          ->  VOLTA selection
                          ->  risk review
                          ->  final dry-run plan

Inputs
------
market : dict with keys ``events``, ``instruments``, ``indices``, ``flows`` for
    the signal engines (see candidates.engines), plus optionally
    ``returns`` : an (N, T) array of daily returns for the universe used to
    build the cleaned correlation matrix and detect the regime. If ``returns``
    is absent, RMT/TDA are skipped and the regime defaults to ``range``.

Nothing here executes trades. The output is a plan for dry-run / paper review.
"""






@dataclass
class PipelineResult:
    candidates: list[Candidate]
    regime: str
    regime_confidence: float
    regime_note: str
    volta: VoltaResult
    risk: RiskDecision
    final_plan: list[Candidate]
    trace: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Stage 1 (candidates): {len(self.candidates)} proposed",
            f"Stage 3 (regime):     {self.regime} (conf {self.regime_confidence})",
            f"                      {self.regime_note}",
            f"Stage 4 (VOLTA):      selected {len(self.volta.selected)}, "
            f"net vega {self.volta.net_vega:.0f}, capital {self.volta.total_capital:.0f}",
            f"Stage 5 (risk):       {self.risk.note}",
            f"FINAL PLAN:           {len(self.final_plan)} trades",
        ]
        for c in self.final_plan:
            lines.append(f"   - {c.id} {c.structure.value} {c.underlying} "
                         f"edge={c.edge:.1%} cap={c.capital:.0f}  [{c.note}]")
        if self.risk.flags:
            lines.append("FLAGS:")
            lines.extend(f"   ! {x}" for x in self.risk.flags)
        return "\n".join(lines)


def _corr_for_candidates(
    candidates: list[Candidate], returns: np.ndarray, universe: list[str]
) -> np.ndarray | None:
    """Build a candidate-indexed correlation matrix from the cleaned universe corr.

    ``returns`` rows are aligned to ``universe`` (list of underlying names).
    Returns an (n_cand, n_cand) matrix whose (i,j) entry is the cleaned
    correlation between candidate i's and candidate j's underlyings, or None if
    alignment is impossible.
    """
    if returns is None or not universe:
        return None
    idx = {name: k for k, name in enumerate(universe)}
    cleaned = rmt_clean(returns).cleaned
    n = len(candidates)
    out = np.zeros((n, n))
    for i in range(n):
        ui = idx.get(candidates[i].underlying)
        for j in range(n):
            uj = idx.get(candidates[j].underlying)
            if ui is not None and uj is not None:
                out[i, j] = cleaned[ui, uj]
    return out, cleaned


def run(
    market: dict,
    volta_cfg: VoltaConfig | None = None,
    risk_limits: RiskLimits | None = None,
) -> PipelineResult:
    """Run the full derivatives pipeline on a market snapshot."""
    trace: dict = {}

    # Stage 1: candidate generation
    candidates = generate_all(market)
    trace["n_candidates"] = len(candidates)

    # Stages 2+3: RMT clean + TDA regime (only if returns provided)
    returns = market.get("returns")
    universe = market.get("universe", [])
    regime_res: RegimeResult
    corr_for_cand = None
    if returns is not None and len(universe) > 0:
        aligned = _corr_for_candidates(candidates, np.asarray(returns), universe)
        if aligned is not None:
            corr_for_cand, cleaned_corr = aligned
            regime_res = detect_regime(cleaned_corr)
        else:
            regime_res = RegimeResult(Regime.RANGE, 0.5, {}, "no alignment; default")
    else:
        regime_res = RegimeResult(Regime.RANGE, 0.5, {},
                                  "no returns supplied; defaulting to range")
    trace["regime_features"] = regime_res.features

    # Stage 4: VOLTA selection
    volta_res = solve(candidates, regime=regime_res.regime.value,
                      corr=corr_for_cand, cfg=volta_cfg)

    # Stage 5: risk review
    risk_res = review(volta_res, limits=risk_limits)

    return PipelineResult(
        candidates=candidates,
        regime=regime_res.regime.value,
        regime_confidence=regime_res.confidence,
        regime_note=regime_res.note,
        volta=volta_res,
        risk=risk_res,
        final_plan=risk_res.approved,
        trace=trace,
    )


# --- aliases so the concatenated orchestrator resolves names ---
rmt_clean = clean
RegimeResult = RegimeResult  # noqa

# ==============================================================================
# DEMO  --  full pipeline on a synthetic market snapshot
# ==============================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(2)
    universe = ["MARKSANS","COFORGE","TRIVENI","PAINTCO","NIFTY","BANKNIFTY","REFEX","DIVIS"]
    n, T = len(universe), 250
    mkt = rng.standard_normal((1, T))
    returns = 0.8*np.ones((n,1))@mkt + rng.standard_normal((n,T))
    market = {
        "universe": universe, "returns": returns,
        "events": [dict(underlying="MARKSANS", sector="Pharma", expiry="2026-06-26",
                        spot=230, iv_percentile=0.72, call_strike=240, put_strike=220,
                        est_premium=18000, est_margin=85000, vega_per_lot=-380, days_to_event=4)],
        "instruments": [dict(underlying="TRIVENI", sector="Engg", expiry="2026-06-26",
                            fundamental_score=0.7, technical_score=0.5, est_margin=145000,
                            est_move=0.04, delta_per_lot=75),
                        dict(underlying="PAINTCO", sector="Paint", expiry="2026-06-26",
                            fundamental_score=-0.6, technical_score=-0.4, est_margin=98000,
                            est_move=0.03, delta_per_lot=70)],
        "indices": [dict(underlying="NIFTY", expiry="2026-06-26", spot=23000, vix=14.5,
                        expected_move=300, wing_width=200, est_credit=9000,
                        est_margin=180000, vega_per_lot=-500)],
        "flows": [dict(underlying="BANKNIFTY", expiry="2026-06-26", cum_fii=2400,
                      consistency=0.8, est_margin=160000, est_move=0.02, delta_per_lot=75)],
    }
    result = run(market, volta_cfg=VoltaConfig(capital_max=1_500_000, num_reads=2000, seed=1))
    print(result.summary())
