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

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from ripser import ripser


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
