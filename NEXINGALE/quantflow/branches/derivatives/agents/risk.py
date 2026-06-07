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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from quantflow.branches.derivatives.candidates.schema import Candidate
from quantflow.branches.derivatives.volta.optimizer import VoltaResult


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
