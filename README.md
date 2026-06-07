# Nexingale (quantflow)

A quantitative research and trading pipeline, split into two branches that share
one spine. The split exists because long-horizon equity selection and dynamic
derivatives trading want genuinely different machinery — forcing one architecture
to do both compromises both.

## Two branches, one spine

```
                    shared spine
   data adapters -> pattern memory -> RMT cleaning -> TDA regime
                                                         |
                        +--------------------------------+
                        |                                |
                 quantflow-E (equities)         quantflow-D (derivatives)
                 long-horizon, factor-based      dynamic, structure-based
                 Miner / Screener / Trader        candidates -> VOLTA QUBO
                 (AlphaCrafter-native)            -> lean rational agents
```

The **spine** is built once and both branches consume it:

- **data/** — adapters normalising heterogeneous sources (Kite, Yahoo, SOVRENN,
  news, filings) into a common internal format.
- **memory/** — the pattern-memory graph (associative, compounds over time).
  Not on any trade's critical path; read-mostly for the branches.
- **rmt/** — RMT correlation cleaning. **Built and locked.** See below.
- **tda/** — topological regime detection; emits a regime label both branches use.

**quantflow-D (derivatives) is being built first** — it is the user's live edge
(options/futures), the domain where bad output is easiest to sanity-check, and
VOLTA's QUBO subset-selection is native to multi-leg Greek-balanced construction.
All work is dry-run / paper only until the pipeline is validated end to end.

**quantflow-E (equities)** comes second on the proven spine. It adopts the
AlphaCrafter pattern (Miner generates and curates cross-sectional factors,
Screener builds a regime-conditioned ensemble, Trader executes a reference
strategy). This is where the bull/bear debate layer lives — slow cadence makes
the latency affordable.

## Stage 2: RMT correlation cleaning (locked)

`quantflow/spine/rmt/filter.py`

Sample correlation matrices from finite series are mostly noise when T is not
≫ N. Cleaning is the cheapest reliable improvement available to anything that
consumes correlations.

- `clean(returns)` — **default: Ledoit-Wolf linear shrinkage.** Closed-form,
  data-driven intensity; no bandwidth, no grid, no cross-validation, nothing
  that can misfire. Guarantees, verified in the test suite:
  - always returns a valid correlation matrix (symmetric, unit diagonal, PSD);
  - shrinkage intensity stays in [0, 1];
  - the cleaned matrix is **never worse-conditioned than the raw matrix** — on
    every sample. This is the property downstream inverses depend on (a q = 2
    case goes from condition number ~3e13, numerically singular, to ~1e2).
  - closer to the population matrix in **expected** Frobenius loss (an average
    over noise, not a per-sample certainty — that is LW's actual theorem).
- `clean_rie(returns, eta=...)` — opt-in Ledoit-Péché rotationally-invariant
  estimator. Frobenius-optimal among RIEs in the high-dim limit, but needs a
  tuned bandwidth and does not ship as default (see the ledger).
- `estimate_n_factors(eigenvalues)` — Onatski (2010) factor count, kept as a
  diagnostic (a future TDA regime descriptor), not used to drive cleaning.

Run the tests: `python -m pytest tests/spine/test_rmt.py -q`

