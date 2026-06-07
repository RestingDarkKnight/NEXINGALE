# Mathematical Improvements Ledger

A running, honest record of where quantflow trades mathematical optimality for
robustness, simplicity, or build speed. Each entry states what we shipped, why,
what the more rigorous choice would be, and what it would cost. Nothing here is a
bug — these are deliberate, documented trade-offs. Revisit when a component
proves load-bearing enough to justify the upgrade.

---

## Stage 2 — RMT correlation cleaning

### 2.1 Default estimator: Ledoit-Wolf linear shrinkage (vs nonlinear)

**Shipped.** Linear shrinkage toward a constant-correlation target with a
closed-form intensity. Cannot degenerate; improves conditioning unconditionally.

**Why.** During the build we implemented, in order: naive Marchenko-Pastur
clipping, Bouchaud-Potters σ²-fitted clipping, Onatski-counted clipping, the
Ledoit-Péché RIE with a fixed bandwidth, and the RIE with cross-validated
bandwidth. Each more "optimal" estimator introduced a tunable that could (and
did) misfire:
  - clipping flattens sub-cliff structure → hurt Frobenius distance;
  - RIE bandwidth η at the textbook N^(-1/2) beat raw in only ~5/25 configs;
  - CV on Frobenius over-smooths (boundary optimum at large η);
  - CV on min-variance portfolio risk overfits one portfolio, wrecking the
    matrix elsewhere.
The lesson: the "best" η is **use-dependent**, so no single self-tuned objective
generalises. Linear LW sidesteps the entire problem.

**Rigorous upgrade.** Ledoit-Wolf **analytical nonlinear shrinkage** (Ledoit &
Wolf 2015/2020, the QuEST-function approach): shrink each eigenvalue by a
data-driven amount derived from the asymptotic Marchenko-Pastur fixed-point,
with no free bandwidth. It is the genuinely optimal rotationally-invariant
estimator *and* has no tunable. Cost: implementing or vendoring the QuEST
numerical inversion (nontrivial; ~150-300 lines or a dependency). This is the
correct eventual replacement for both `clean` and `clean_rie`.

### 2.2 Shrinkage target: constant-correlation (vs single-index / factor)

**Shipped.** Target F = equicorrelation matrix (mean off-diagonal correlation on
the off-diagonals, ones on the diagonal).

**Why.** Simplest target with a clean closed-form intensity; appropriate when a
single positive common factor dominates (true for broad equity baskets).

**Rigorous upgrade.** A **single-index (market-model) target** or a
multi-factor target (Ledoit-Wolf 2003 use the single-index model). Better when
the cross-section has strong sector block structure the equicorrelation target
cannot represent. Cost: estimate market betas, build F from them, adjust the π/γ
estimators accordingly (~40 lines). Worth it if TDA later shows persistent block
regimes.

### 2.3 RIE bandwidth: fixed / CV (vs analytic)

**Shipped (opt-in).** `clean_rie` defaults η = N^(-1/3), beats raw in the
majority of T ≫ N configs but is not guaranteed.

**Rigorous upgrade.** Same as 2.1 — the analytic nonlinear-shrinkage bandwidth
removes the parameter entirely. Until then, RIE stays opt-in/research only.

### 2.4 Onatski factor count: ED estimator (vs full distributional test)

**Shipped.** Onatski (2010) Eigenvalue-Difference estimator as a *diagnostic*
only. Robust to dominant factors (finds the spectral cliff).

**Why.** It is fast, parameter-light, and we only use the count as a future
regime feature, not to drive cleaning — so its occasional ±1 miscount is
harmless.

**Rigorous upgrade.** Onatski's full **edge-distribution test** with a calibrated
significance level, or the Passemier-Yao bias-corrected estimator, or parallel
analysis (Horn). Cost: modest. Only worth it once TDA actually consumes the count
as a load-bearing feature.

### 2.5 Standardisation: sample mean/std (vs robust / EWMA)

**Shipped.** Plain per-asset demeaning and unit-variance scaling over the window.

**Rigorous upgrade.** (a) **Robust** location/scale (median / MAD) to blunt fat
tails and jumps that distort correlations; (b) **EWMA** weighting so recent
observations count more, matching the non-stationarity the whole pipeline is
premised on. Cost: small. The EWMA version is likely worth doing before live use
because it directly addresses regime drift.

### 2.6 Stationarity within the window (assumed, not tested)

**Shipped.** Cleaning treats the T columns as i.i.d. draws from one distribution.

**Reality.** Markets are non-stationary inside any window long enough to make
q < 1. We currently ignore this.

**Rigorous upgrade.** Either shorten windows and accept higher q (then *require*
the nonlinear estimator, which handles q ≥ 1 gracefully), or model the drift
explicitly (e.g. DCC-GARCH style dynamic correlation). Cost: significant; this is
a research thread, not a patch. Flag for the publishable-work track.

---

## Template for future entries

### N.M — <component> — <one-line trade-off>
**Shipped.** what we did.
**Why.** the robustness/speed reason.
**Rigorous upgrade.** the better math, and its cost.

---

## Stage 1 — Candidate generation (signal engines)

### 1.1 Rule-based edge estimates (vs calibrated models)
**Shipped.** Each engine computes `edge` from a simple transparent formula
(e.g. vol-crush edge = (IV percentile - floor) x premium/margin).
**Why.** Readable, tunable to the user's book, no training data needed yet.
**Rigorous upgrade.** Calibrate edges to realised post-event returns from
history (e.g. fit the actual IV-crush distribution per underlying); replace the
hand formulas with empirical expected-edge models. This is the derivatives
analog of AlphaCrafter's Miner validation loop (IC/ICIR/decay).

### 1.2 Greeks supplied, not computed (vs a pricing model)
**Shipped.** Engines take vega/delta/theta as inputs from the data dict.
**Why.** Avoids embedding an options pricer in v1; the broker snapshot already
carries Greeks.
**Rigorous upgrade.** A Black-Scholes / Bachelier pricer to compute Greeks and
fair value internally, enabling vol-surface arbitrage candidates and removing
dependence on vendor Greeks.

## Stage 3 — TDA regime detection

### 3.1 Rule-based classifier (vs trained model)
**Shipped.** Thresholds over interpretable persistence features (H1/H2 energy,
avg correlation, top-mode share) -> one of 4 regimes.
**Why.** Auditable and needs no labelled history; you can read why a regime was
chosen.
**Rigorous upgrade.** Train a classifier (gradient-boosted trees) on historical
windows with hindsight regime labels, as in the design doc; validate detection
*latency* against an HMM baseline (the publishable result).

### 3.2 Full Vietoris-Rips persistence (vs scalable approximations)
**Shipped.** Exact ripser persistence on the full distance matrix.
**Why.** Correct and fine for N up to a few hundred.
**Rigorous upgrade.** Sparse/approximate filtrations (e.g. witness complexes) if
the universe grows to thousands of instruments.

## Stage 4 — VOLTA

### 4.1 Soft penalties + hard repair (vs exact constrained solver)
**Shipped.** QUBO with soft quadratic penalties, auto-scaled to edge magnitude,
plus a deterministic post-solve repair that strictly enforces the capital cap.
**Why.** Soft QUBO penalties cannot enforce a hard inequality; the repair makes
the money limit inviolable regardless of penalty calibration.
**Rigorous upgrade.** A proper constrained solver (e.g. CQM via dimod's
ConstrainedQuadraticModel, or branch-and-bound for small N) that handles
inequalities natively; or penalty weights tuned by backtest rather than the
edge-magnitude heuristic.

### 4.2 Regime tilt table (vs learned regime-conditional edges)
**Shipped.** Hand-set multipliers (e.g. theta-harvest x1.4 in range regimes).
**Why.** Encodes obvious trading priors transparently.
**Rigorous upgrade.** Estimate regime-conditional edges from history per the
design doc's ei(r) = alpha*mu_i(r) + (1-alpha)*mu_base formulation.

## Stage 5 — Risk agent

### 5.1 Deterministic rules + LLM hook (vs live LLM judgement)
**Shipped.** Hard rule-based risk review (concentration, Greek limits,
constraint-targeted size-down) with a no-op llm_review hook.
**Why.** This environment makes no external calls, and the D-branch deliberately
favours rational rules over debate chatter.
**Rigorous upgrade.** Wire llm_review to the Claude API for a qualitative overlay
(news/narrative sanity) on top of the hard rules; keep the rules as the
inviolable floor.
