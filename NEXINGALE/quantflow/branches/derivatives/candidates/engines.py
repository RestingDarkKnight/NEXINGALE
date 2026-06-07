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

from __future__ import annotations

from .schema import Candidate, OptionLeg, Structure, StrategyType


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
