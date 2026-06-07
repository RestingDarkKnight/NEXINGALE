"""Common types for the derivatives branch.

A ``Candidate`` is the contract between the signal engines (which propose trades)
and VOLTA (which selects a subset). Every engine, no matter how different its
logic, emits the same shape: a structure with an estimated edge, the capital it
ties up, its Greek exposures, and tags VOLTA uses for diversification penalties.

This is the piece the v2 walkthrough hand-waved -- the C01..C12 candidates
"appeared". Here they are produced explicitly, with the fields VOLTA's QUBO needs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
