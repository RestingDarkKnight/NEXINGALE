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

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from quantflow.spine.rmt import clean as rmt_clean
from quantflow.spine.tda.detect import detect_regime, Regime, RegimeResult
from quantflow.branches.derivatives.candidates.engines import generate_all
from quantflow.branches.derivatives.candidates.schema import Candidate
from quantflow.branches.derivatives.volta.optimizer import solve, VoltaConfig, VoltaResult
from quantflow.branches.derivatives.agents.risk import review, RiskLimits, RiskDecision


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
