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

from __future__ import annotations

from dataclasses import dataclass, field

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler

from quantflow.branches.derivatives.candidates.schema import Candidate, StrategyType


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
