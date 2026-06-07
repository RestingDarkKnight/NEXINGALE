# quantflow — Status, Workflow & Deployment Guide

*As of the current build. Branch: derivatives (quantflow-D). Mode: dry-run only.*

---

## 1. Where we are

The project splits into two branches sharing one spine. We are building the
**derivatives branch (quantflow-D) first** — it is the live trading edge, the
domain easiest to sanity-check, and the one where VOLTA's QUBO selection is
native. The equities branch (quantflow-E) is deferred.

### What is built and tested

| Stage | Component | State | Notes |
|---|---|---|---|
| Spine 2 | RMT correlation cleaning | **done, locked** | Ledoit-Wolf shrinkage; can't degenerate; improves conditioning on every sample |
| Spine 3 | TDA regime detection | **done** | Real persistent homology (ripser); rule-based 4-regime classifier |
| D-1 | Candidate generation | **done** | 4 signal engines (vol-crush, directional, theta-harvest, flow-follow) |
| D-4 | VOLTA QUBO optimizer | **done** | Real dimod/neal annealer; regime tilts; hard capital-cap repair |
| D-5 | Risk agent | **done** | Deterministic rules; constraint-targeted size-down; LLM hook (off) |
| — | Pipeline orchestrator | **done** | Chains all stages; full reasoning trace |
| — | Single standalone file | **done** | `quantflow_d_standalone.py`, runs top-to-bottom |

23 tests pass. The full pipeline runs end-to-end on a synthetic market snapshot.

### What is NOT built (and why it matters)

| Missing | Consequence |
|---|---|
| **Real data adapter** | The pipeline runs only on hand-made / synthetic `market` dicts. It has never seen your actual Kite / SOVRENN data. |
| **Backtest harness** | We have **no evidence the system has edge.** Nothing has been scored against history. |
| Spine 1 — pattern memory | No accumulated context yet (off the critical path; compounds later). |
| Live broker execution | By design — nothing trades. Dry-run only. |
| Equities branch | Deferred until D is validated. |

**The honest headline: the machinery works; whether it makes money is completely
unproven.** That is the next phase, not this one.

---

## 2. The final workflow

```
   YOUR DATA (Kite OHLCV, option chain, SOVRENN, FII/DII flows)
        |
        |  [data adapter — NOT BUILT YET; today you hand-build this dict]
        v
   market = {
       "universe":   [list of underlying names],
       "returns":    (N, T) array of daily returns for those underlyings,
       "events":     [vol-crush event rows],
       "instruments":[directional futures rows],
       "indices":    [iron-condor index rows],
       "flows":      [FII/DII flow rows],
   }
        |
        v
   STAGE 1  signal engines        -> list of Candidates (each: edge, capital, Greeks, tags)
        |
        v
   STAGE 2  RMT clean             -> denoised correlation matrix over the universe
        |
        v
   STAGE 3  TDA regime            -> regime label (trend / range / sector_rotation / stress)
        |
        v
   STAGE 4  VOLTA select          -> optimal subset under capital/vega/sector/corr penalties,
        |                            with regime-conditional edge tilts + hard capital cap
        v
   STAGE 5  risk review           -> veto oversized trades, size down to Greek limits
        |
        v
   FINAL PLAN  (dry-run)          -> approved trades + entry rationale + flags + full trace
```

Each stage's output is carried in the result object, so any decision is
auditable back to the data that produced it.

---

## 3. How to run it

### Option A — the standalone file (simplest)

```bash
pip install numpy dimod dwave-neal ripser
python quantflow_d_standalone.py        # runs the built-in demo
```

To run on your own snapshot, edit the `market = {...}` block at the bottom of
the file and re-run.

### Option B — the package (for development)

```bash
unzip quantflow_v0.2.zip && cd quantflow
pip install -e .            # or: pip install numpy dimod dwave-neal ripser
pip install pytest && python -m pytest tests/ -q     # 23 tests should pass
```

```python
from quantflow.branches.derivatives import run
from quantflow.branches.derivatives.volta.optimizer import VoltaConfig

market = { ... }   # see the input spec below
result = run(market, volta_cfg=VoltaConfig(capital_max=1_500_000))
print(result.summary())
```

### Input data spec (what each engine consumes)

Every row is a plain dict. The data adapter (when built) produces these from
broker data; today you build them by hand.

- **events** (vol-crush): `underlying, sector, expiry, spot, iv_percentile (0-1),
  call_strike, put_strike, est_premium, est_margin, vega_per_lot, days_to_event`
- **instruments** (directional): `underlying, sector, expiry, fundamental_score
  (-1..1), technical_score (-1..1), est_margin, est_move, delta_per_lot`
- **indices** (theta-harvest): `underlying, expiry, spot, vix, expected_move,
  wing_width, est_credit, est_margin, vega_per_lot`
- **flows** (flow-follow): `underlying, expiry, cum_fii (crore, signed),
  consistency (0-1), est_margin, est_move, delta_per_lot`
- **returns**: `(N, T)` numpy array of daily returns, rows aligned to `universe`.
  Optional — without it, RMT/TDA are skipped and the regime defaults to `range`.

---

## 4. Deployment guide

**Deployment here means paper / dry-run only.** This system must not touch real
capital until it has a validated backtest track record. The phases:

### Phase 0 — now: synthetic validation (done)
The pipeline runs and is internally consistent. No real data, no claims of edge.

### Phase 1 — real data adapter (next)
Build `spine/data/adapters/` to turn your Kite exports, option-chain snapshots,
SOVRENN issues, and FII/DII data into the `market` dict above. Sanity-check that
the candidates it produces look like trades you would actually consider. This is
where your trading intuition is the test.

### Phase 2 — backtest harness
Build `backtest/` to feed the pipeline a rolling historical window — one `market`
snapshot per day — collect the plans, and score realised P&L, Sharpe, and max
drawdown against a benchmark. **This is the gate.** If there is no edge here,
nothing else matters. Reserve the most recent ~12 months as untouched hold-out.

### Phase 3 — paper trading
Run the pipeline daily on live data, log the plans, but place no orders. Compare
the plans against what actually happened for a meaningful period (months, not
weeks). Wire the `llm_review` hook to the Claude API here for a qualitative
overlay if desired.

### Phase 4 — live, small, gated (only if Phases 2-3 pass)
Reuse the NEXINGALE risk discipline: `DRY_RUN` flag, `MAX_DAILY_LOSS`,
`MAX_OPEN_POSITIONS`, `MAX_CAPITAL_PER_TRADE`. Start with capital you can lose.
The risk agent's hard rules are the inviolable floor; never bypass them.

### Safety rules that always hold
- The capital cap is enforced by hard repair — never overridden.
- The risk agent's Greek/concentration limits are the floor, not a suggestion.
- No order is placed by any code in this repo. Execution is a separate,
  deliberate, later decision.

---

## 5. Honest limitations

See `MATH_LEDGER.md` for the full per-component list of where we chose
robustness over mathematical optimality and what the rigorous upgrade is. The
big ones:

- **Edge estimates are hand-formulas, not calibrated to history.** The single
  most important upgrade before trusting any P&L.
- **Regime classifier is rule-based**, not trained on labelled history.
- **Greeks are supplied by the data**, not computed by a pricer.
- **No proof of edge exists yet** — that is Phase 2.

---

## 6. Immediate next step

Pick one:
- **(a) Backtest harness on synthetic data** — proves the daily loop works,
  needs no real data.
- **(b) Real data adapter** — turns your Kite/SOVRENN exports into the `market`
  dict so everything runs on real numbers (recommended: the system is only as
  good as its inputs, and real candidates are far easier to sanity-check).
