"""Tests for the derivatives branch: engines, VOLTA, risk, and the full pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from quantflow.branches.derivatives.candidates import engines
from quantflow.branches.derivatives.candidates.schema import Candidate, Structure, StrategyType
from quantflow.branches.derivatives.volta.optimizer import solve, VoltaConfig
from quantflow.branches.derivatives.agents.risk import review, RiskLimits
from quantflow.branches.derivatives.pipeline import run


# ---- engines ----

def test_vol_crush_skips_low_iv():
    events = [dict(underlying="X", sector="P", expiry="E", spot=100,
                   iv_percentile=0.50, call_strike=110, put_strike=90,
                   est_premium=1000, est_margin=50000)]
    assert engines.vol_crush(events) == []  # below 0.60 floor


def test_vol_crush_fires_on_high_iv():
    events = [dict(underlying="X", sector="P", expiry="E", spot=100,
                   iv_percentile=0.80, call_strike=110, put_strike=90,
                   est_premium=8000, est_margin=50000)]
    out = engines.vol_crush(events)
    assert len(out) == 1 and out[0].strategy_type == StrategyType.VOL_CRUSH


def test_directional_skips_disagreeing_signals():
    ins = [dict(underlying="X", sector="E", expiry="E",
                fundamental_score=0.7, technical_score=-0.5, est_margin=100000,
                est_move=0.04)]
    assert engines.directional_futures(ins) == []


def test_theta_harvest_skips_high_vix():
    ix = [dict(underlying="NIFTY", expiry="E", spot=23000, vix=25.0,
               expected_move=300, wing_width=200, est_credit=9000, est_margin=180000)]
    assert engines.theta_harvest(ix) == []


def test_flow_follow_direction_matches_sign():
    f = [dict(underlying="BN", expiry="E", cum_fii=-3000, consistency=0.9,
              est_margin=160000, est_move=0.02)]
    out = engines.flow_follow(f)
    assert len(out) == 1 and out[0].structure == Structure.SHORT_FUTURE


# ---- VOLTA ----

def _mk(id, edge, cap, vega, sector, stype, under="U"):
    return Candidate(id=id, underlying=under, structure=Structure.LONG_FUTURE,
                     strategy_type=stype, sector=sector, edge=edge, capital=cap,
                     vega=vega, delta=0.0, theta=0.0)

def test_volta_respects_capital():
    cands = [_mk(f"C{i}", 0.03, 600000, 0, f"S{i}", StrategyType.DIRECTIONAL)
             for i in range(5)]
    res = solve(cands, regime="trend", cfg=VoltaConfig(capital_max=1_000_000, num_reads=500))
    assert res.total_capital <= 1_000_000  # hard repair guarantees the cap


def test_volta_never_empty_when_affordable():
    cands = [_mk("C0", 0.02, 50000, 0, "S", StrategyType.DIRECTIONAL)]
    res = solve(cands, regime="range", cfg=VoltaConfig(num_reads=200))
    assert len(res.selected) >= 1


def test_volta_regime_tilt_prefers_theta_in_range():
    theta = _mk("T", 0.02, 100000, -100, "Index", StrategyType.THETA_HARVEST)
    direc = _mk("D", 0.02, 100000, 0, "Other", StrategyType.DIRECTIONAL)
    # in range, theta is tilted up; with identical raw edge it should be favoured
    res = solve([theta, direc], regime="range",
                cfg=VoltaConfig(capital_max=100000, num_reads=1000, sector_cap=1))
    ids = {c.id for c in res.selected}
    assert "T" in ids


# ---- risk ----

def test_risk_vetoes_oversized_single_trade():
    big = _mk("BIG", 0.05, 600000, 0, "S", StrategyType.DIRECTIONAL)
    from quantflow.branches.derivatives.volta.optimizer import VoltaResult
    vr = VoltaResult([big], 0, 600000, 0, 0, 0, "")
    dec = review(vr, RiskLimits(capital_max=1_000_000, max_single_capital_frac=0.30))
    assert big in [c for c, _ in dec.rejected]


def test_risk_sizes_down_to_vega_limit():
    from quantflow.branches.derivatives.volta.optimizer import VoltaResult
    cands = [_mk(f"C{i}", 0.03, 100000, -300, f"S{i}", StrategyType.VOL_CRUSH)
             for i in range(4)]  # net vega -1200
    vr = VoltaResult(cands, 0, 400000, -1200, 0, 0, "")
    dec = review(vr, RiskLimits(vega_abs_max=500))
    assert abs(dec.trace["net_vega"]) <= 500


# ---- full pipeline ----

def test_full_pipeline_runs_and_is_auditable():
    rng = np.random.default_rng(0)
    universe = ["A", "B", "C", "NIFTY"]
    returns = 0.7 * np.ones((4, 1)) @ rng.standard_normal((1, 200)) + rng.standard_normal((4, 200))
    market = {
        "universe": universe, "returns": returns,
        "events": [dict(underlying="A", sector="P", expiry="E", spot=100,
                        iv_percentile=0.75, call_strike=110, put_strike=90,
                        est_premium=8000, est_margin=80000, vega_per_lot=-380)],
        "indices": [dict(underlying="NIFTY", expiry="E", spot=23000, vix=14,
                         expected_move=300, wing_width=200, est_credit=9000,
                         est_margin=180000, vega_per_lot=-500)],
    }
    res = run(market, volta_cfg=VoltaConfig(num_reads=500))
    assert res.regime in {"trend", "range", "sector_rotation", "stress"}
    assert isinstance(res.final_plan, list)
    assert res.trace["n_candidates"] >= 1
    assert isinstance(res.summary(), str)
