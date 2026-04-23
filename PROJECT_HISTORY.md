# Project History: OFDM-ISAC DQN Adaptive Pilot Allocation

**B.Tech Final Year Project — NIT Silchar, May 2026**
**Student:** Abhinash Dutta

This document consolidates all planning notes, AI-session handoffs, and experiment trial records into a single historical narrative. The individual files (`ENDGAME_PLAN.md`, `IMPLEMENTATION_DIVERGENCE_LOG.md`, `PROJECT_HANDOFF.md`, etc.) and the `artifacts/` directory have been archived in this document and removed from the repo to keep the codebase clean. The canonical reference model is preserved in `REFERENCE_MODEL_FINAL.md`.

---

## 1. Project Origin

The project aimed to convert a professor's draft paper on OFDM-ISAC resource allocation into a runnable simulation and validate the proposed DQN-based adaptive pilot allocation approach. The starting point was the document `REFERENCE_MODEL_FOR_GITHUB.md` — a draft model specification describing:

- An OFDM system acting as an integrated sensor and communicator.
- An RL agent choosing pilot ratio `ρ_t` and pilot power fraction `β_t` at each frame.
- A three-component reward: communication rate, synchronization penalty, sensing penalty.

The project was treated from the outset as a **proof-of-concept simulation**, not a hardware deployment.

---

## 2. Critical Discrepancies Found in the Original Reference Model

Before any experiments could be run, a review of the original paper (aided by Gemini and Claude) identified nine significant errors. These were catalogued in `CHANGELOG_REFERENCE_MODEL.md` and `DISCREPANCIES_NOTED_BY_GEMINI.md`, and resolved in `REFERENCE_MODEL_FINAL.md`.

### 2.1 Causality Paradox in the MDP (Critical)
The original state `s_t` included the current frame's CFO estimate `ε̂_t` and channel gain `|H_t|²`. But these quantities can only be computed *after* the frame is received, which requires the action `(ρ_t, β_t)` to have already been chosen (since pilot placement depends on it). This was a chicken-and-egg paradox.

**Fix:** The state was redefined to use strictly delayed observations: `s_t = (ε̂_{t-1}, 𝓔_{t-1}, SINR̄_{t-1}, SNR_pilot,{t-1}, |Ĥ_{t-1}|², ρ_{t-1}, β_{t-1})`. This makes the MDP causally valid.

### 2.2 Unobservable State Variable
The original state included `ε̃_t² = (ε_t - ε̂_t)²` — the true residual CFO error. But `ε_t` is not observable at the receiver. If it were, the estimation problem would be trivially solved. This variable was also used in the reward.

**Fix:** Replaced with the observable synchronization mismatch metric:
```
𝓔_t = (1/|P_t|N) Σ_{k∈P_t} Σ_{n=1}^{N-1} |Y_t[k,n+1] - Y_t[k,n] e^{j2πε̂_t T_s}|²
```
This quantifies residual phase inconsistency after CFO compensation and is computable from received pilots.

### 2.3 Biased CFO Estimator
The original estimator divided by `N` but summed only `N-1` phase differences (from `n=1` to `N-1`), introducing a factor-of-`(N-1)/N` systematic bias.

**Fix:** Changed denominator from `N` to `N-1`.

### 2.4 ICI Power Missing Channel Gain
The ICI power `σ²_ICI` was scaled by transmit power `P` but not by channel gain `|H_t|²`, while the desired signal correctly included the fading gain. Since ICI propagates through the same channel as the signal, this was physically inconsistent.

**Fix:** Updated to `σ²_ICI ≈ C · ε̃_t² · (1-β_t)P|H_t|²`.

### 2.5 Observable SINR Proxy
The original SINR term in the state was the analytical expression depending on the hidden `ε̃_t²`. After the ICI fix, a measurable proxy was derived from `𝓔_t`:
```
ε̃_t²_hat = max(0, (𝓔_t - 2σ²) / ((2πT_s)² · p_pilot · |Ĥ_t|²))
σ²_ICI_hat = C · ε̃_t²_hat · (1-β_t)P|Ĥ_t|²
SINR̄_t = (1-β_t)P|Ĥ_t|² / (σ² + σ²_ICI_hat)
```
This is an approximation, not an identity, relying on repeated pilots and quasi-static channel assumptions. In the thesis it is described honestly as an observable estimate.

### 2.6 Sensing Metric (CRLB → Empirical)
The original sensing metric `Var(τ̂_t)` was a Cramér-Rao Lower Bound — a theoretical quantity not computable online. The final paper explicitly moved away from CRLB-based sensing.

**Fix:** Replaced with the empirical sidelobe-to-peak ratio of a pilot-domain delay-correlation profile:
```
S_t = mean sidelobe power / peak power
```
Lower `S_t` = better sensing. This is observable from received pilots without ground-truth knowledge.

### 2.7 Sensing Model Missing Pilot Density
The original sensing SNR `= β_t P / σ²` ignored pilot density `ρ_t`, contradicting the claim that both variables jointly affect sensing.

**Fix:** `SNR_pilot = β_t P |H_t|² / (ρ_t K σ²)`.

### 2.8 Time-Varying Power `P_t` Undefined
The original used `P_t` with no transition dynamics, making the system open-loop and the SNR state variable trivially static.

**Fix:** Replaced with constant `P`. Removed `SNR_{t-1}` from the dynamic state.

### 2.9 Convergence Claims Too Strong
The original made near-guarantee claims about DQN convergence. Standard DQNs have no formal convergence guarantee with nonlinear function approximation.

**Fix:** Reworded to claim only empirical convergence, justified by experience replay and target networks.

---

## 3. Implementation Choices Not Specified by the Paper

The paper left many numerical values open. The chosen values (confirmed acceptable by the professor):

| Parameter | Value | Reason |
|---|---|---|
| Subcarriers `K` | 40 | Action ratios map cleanly to integers |
| OFDM symbols per frame `N` | 8 | Small, practical frame size |
| Average power `P` | 1.0 | Normalized |
| Noise variance `σ²` | 0.05 | Estimator behavior visible without trivial channel |
| Symbol duration `T_s` | 66.67 µs | 15 kHz subcarrier spacing scale |
| CFO mean reversion `α_c` | 0.95 | Bounded AR(1) drift |
| CFO innovation std | 5 Hz | Noticeable but not explosive drift |
| Channel correlation `α` | 0.98 | Slowly varying block fading |
| Target delay | 1.2 µs | Makes sensing proxy meaningful |
| Episode length | 64 frames | Training/evaluation convention |

DQN hyperparameters (engineering choices, not from paper):

| Hyperparameter | Value |
|---|---|
| Hidden layers | (48, 48) |
| Activation | ReLU |
| Learning rate | 1e-3 |
| Discount factor | 0.98 |
| Replay capacity | 4000 |
| Batch size | 32 |
| Epsilon schedule | 1.0 → 0.05 over 2000 steps |
| Warmup steps | 128 |
| Target update interval | 100 |

---

## 4. Experiment Trail (Artifacts History)

The `artifacts/` directory accumulated 25 subdirectories across the project lifecycle. A chronological summary:

### Phase 1 — Reference Model Smoke Tests
**`reference_model_smoke`, `reference_model_smoke2`, `reference_model_traincheck`**
Very early runs to verify the pipeline. Schema was different (no `policy_results`). Purpose: check environment and DQN execute without error. No meaningful learning expected.

### Phase 2 — Empirical Sensing Adoption
**`reference_model_empirical_smoke`, `…smoke2`, `…smoke3`**
Runs after switching `S_t` from CRLB-based to the empirical sidelobe-to-peak metric. Confirmed the empirical metric was computable and produced sensible values. Training was short (smoke-test budget).

### Phase 3 — SINR Observable Fix
**`reference_model_revised_smoke`, `…smoke2`**
After the causality/observability fixes (Section 2), the environment changed substantially. These runs verified the corrected SINR proxy and observable state worked end-to-end. Old artifacts became stale at this point.

### Phase 4 — ICI Fix Verification
**`ici_fix_smoke`**
Targeted verification of the ICI `|H_t|²` correction (Discrepancy 2.4). Confirmed the reward and state were consistent after the fix.

### Phase 5 — Observable SINR Integration
**`sinr_observable_smoke`**
First run using the full observable SINR proxy chain (𝓔_t → ε̃_t²_hat → σ²_ICI_hat → SINR̄_t). Simulator-truth diagnostics (`sinr_true`, `ici_power_true`) were retained internally for comparison.

### Phase 6 — Weight Sweep Exploration
**`weight_sweep_smoke`, `weight_sweep_hybrid_smoke`, `weight_sweep`**
First systematic reward-weight sweeps. Short budgets (40 episodes, 32 frames) over a wide grid `λ_sync ∈ {0.25, 0.5, 1.0, 2.0}`, `λ_sensing ∈ {0.25, 0.5, 1.0, 2.0}`. The winner under hybrid scoring was `(λ_sync=0.25, λ_sensing=0.5)` with a low mean reward of ~16.7 (expected: short training). These were **exploratory only**, not used in final thesis.

**`final_weight_sweep_medium`**
More serious sweep: 100 episodes, 64 frames, {0.25, 0.5, 1.0, 2.0} × {0.25, 0.5, 1.0, 2.0}. Identified the same `(0.25, 0.5)` cluster as consistently strong. Mean reward ~40.8.

**`final_weight_sweep_medium_gap_penalty`**
Variant adding an explicit gap-to-best-fixed penalty term to the hybrid score. Used to stress-test the weight selection robustness.

**`final_refine_loca`**
Additional refinement run narrowing down candidates. Helped identify the final shortlist of three.

### Phase 7 — Final Candidate Confirmation
**`final_compare_l025_l025`, `final_compare_l025_l05`, `final_compare_l05_l05`**
Quick comparison of three near-optimal settings. These fed the shortlisting step.

**`final_confirm_l04_l05` (Candidate A)**
λ_sync=0.4, λ_sensing=0.5. Mean DQN reward: ~28.8. Stability score: 2.51. Rejected: lowest mean reward, most erratic.

**`final_confirm_l025_l05` (Candidate B)**
λ_sync=0.25, λ_sensing=0.5. Mean DQN reward: ~40.8. Stability score: 2.92. Strong hybrid score but lower mean reward than C.

**`final_confirm_l02_l04` (Candidate C — Winner)**
λ_sync=0.2, λ_sensing=0.4. Mean DQN reward: ~46.9. Stability score: 3.61. Selected as winner: highest mean DQN reward and strongest stability score.

### Phase 8 — Extended Training Run
**`extended_500ep/winner`**
500-episode run with Candidate C weights to check if more training helps. Result: reward plateaued near 45 after episode 100. Confirmed the ceiling was not under-training but a fundamental limit of the stationary environment — the best fixed policy at (ρ=0.1, β=0.2) dominates because the rate term overwhelms sync and sensing incentives.

### Phase 9 — Professor Report and Thesis Rerun
**`professor_final_report`**
Pre-generated figures and a PROFESSOR_FINAL_SUMMARY.md sent to the supervisor.

**`thesis_rerun` (candidates A, B, C)**
Fresh multi-seed sweep using the NumPy-refactored DQN (21× faster than the original pure-Python implementation). Used as the authoritative data source for all thesis figures and numbers. Numeric drift vs prior runs was small but non-trivial (see Section 5 below).

---

## 5. Key Decisions Made During the Project

### 5.1 Reward-Weight Selection Method
The project used a **stability-aware hybrid scoring** rule rather than selecting by mean reward alone:
```
hybrid_score = stability_score × baseline_gap_factor
stability_score = mean_reward / std_reward   (over last 20 episodes, across 5 seeds)
baseline_gap_factor = function of |DQN - best_fixed_baseline|
```
Product form enforces AND: both stability and policy quality must be good. A sum would allow one to compensate for a weak other.

### 5.2 Candidate C as Winner Despite B's Hybrid Lead
In the thesis-rerun, Candidate B led on hybrid score (0.609 vs C's 0.365). This was because B's best-fixed baseline, evaluated under B's weights (higher λ_sensing=0.5), scored lower (~49 vs C's ~54), compressing the gap. C was still chosen because:
- C had the highest mean DQN reward (46.45 vs 45.32).
- C had the highest stability score (3.175 vs 2.892).
- B's hybrid lead came from a weaker baseline, not a stronger DQN.
The thesis selection rationale was updated to "highest mean reward and strongest stability" rather than "highest hybrid score".

### 5.3 DQN Implementation: Pure Python → NumPy
The original DQN used a pure-Python forward/backward pass (loop over neurons). This was later refactored to fully vectorised NumPy operations, achieving a 21× speedup. All final thesis results use the NumPy implementation.

### 5.4 Sensing Metric: Empirical Sidelobe-to-Peak
Chose observable empirical `S_t` over CRLB, consistent with the paper's measurable-only philosophy. The CRLB mode was kept in the codebase as a legacy/comparison option but not used in the main thesis results.

### 5.5 Final Reward Weights
```
λ_rate = 1.0,  λ_sync = 0.2,  λ_sensing = 0.4
```
λ_rate fixed at 1.0 as the dominant term. Smaller sync and sensing weights reflect the tradeoff: too high and the DQN ignores the rate term (which drives most of the reward).

### 5.6 Honest Evaluation
The project explicitly identified and reported that:
- The best fixed policy (ρ=0.1, β=0.2) beats the DQN by ~6.6 reward units.
- This is because the stationary AR(1) environment allows a well-chosen fixed action to dominate.
- The DQN still beats the random policy (+8.2) and heuristic (+14.3).
- The root cause (stationary environment favouring constant allocation) is identified as the main limitation and motivates Phase 2a (regime-switching CFO).

---

## 6. Questions Resolved During the Project

| Question | Resolution |
|---|---|
| How to describe the state SINR term? | "Observable proxy derived from 𝓔_t, pilot power, noise variance, and estimated channel power" — an estimate, not an exact measurement |
| How to select reward weights? | Stability-aware hybrid score: stability_score × baseline_gap_factor |
| Does CRLB sensing appear in the thesis? | No — empirical S_t only in main results; CRLB kept as legacy code only |
| Final training budget? | 100 episodes × 64 frames × 5 seeds = 32,000 environment steps per candidate |
| Are the physical parameter values acceptable? | Confirmed acceptable by professor |
| Is the proof-of-concept framing acceptable? | Yes |

---

## 7. Project Close State (Original Scope)

The original scope — convert the professor's paper to a working experiment and validate the DQN approach — is complete. The repository in this commit is the checkpoint.

**What was delivered:**
- Fully observable, causally consistent OFDM-ISAC simulator (NumPy-based).
- DQN agent with 12-action discrete policy, vectorised in NumPy.
- Reward-weight sweep methodology with hybrid stability scoring.
- Comparison against 12 fixed policies, random, and heuristic baseline.
- Complete B.Tech thesis (6 chapters, 2 appendices).
- All figures generated from reproducible scripts (`ofdm_isac_rl/thesis_figures.py`).

**Known performance ceiling:**
DQN reward ~46.5 (seeds {7,17,27,37,47}), best fixed ~53.1. Gap is not under-training (confirmed by 500-episode run plateauing at ~45). It is a structural property of the stationary environment.

**Next step (Phase 2a — separate branch):**
Implement regime-switching CFO (low-drift / high-drift Markov chain). This creates genuine non-stationarity where the optimal fixed action changes over regimes, motivating adaptive control and giving the DQN a real chance to outperform all fixed baselines.
