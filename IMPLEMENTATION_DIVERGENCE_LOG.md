# Implementation Divergence Log

## Scope

Reference document:

- `REFERENCE_MODEL_FINAL.md`

Purpose of this log:

- record every known place where the implementation differs from the paper
- record numerical values chosen because the paper does not specify them
- record engineering choices added for training or evaluation
- separate "paper-faithful default behavior" from "optional comparison modes"

This document should be updated whenever the implementation changes.

## Current Default Position

The implementation should be treated as:

- aligned to the final paper **by default**
- extended with extra engineering choices needed to run experiments
- extended with optional modes for comparison when the paper history is ambiguous

Current recommended default:

- sensing metric mode: `empirical`
- reward-term normalization: `enabled`
- DQN compared against fixed, random, and heuristic baselines

## A. Direct Mismatches or Extensions Relative to `REFERENCE_MODEL_FINAL.md`

| Area | Paper Says | Implementation Does | Type | Status |
| --- | --- | --- | --- | --- |
| Sensing metric `S_t` | `S_t` should be an observable empirical proxy based on delay-correlation sharpness or measured delay estimation error | Supports both `empirical` and `crlb` modes; default is now `empirical` | Extension | Keep `empirical` as main mode, keep `crlb` only as comparison/legacy mode |
| Exact empirical formula for `S_t` | Paper leaves `S_t` open-ended | Uses sidelobe-to-peak ratio of a pilot-domain delay-correlation profile | Design choice | Acceptable unless professor specifies a different observable proxy |
| Reward formula | Paper uses raw measurable metrics `r_t = lambda_1 R_t - lambda_2 E_t - lambda_3 S_t` | Implementation can normalize each term by a fixed reference before applying weights | Engineering extension | Keep, but record clearly in report |
| State observability of SINR term | Paper says the state should contain measurable quantities from the previous frame | State now uses an observable SINR proxy built from `\mathcal{E}_t`, pilot power, noise variance, and `|\hat{H}_t|^2`; simulator-truth SINR is kept only for diagnostics | Engineering interpretation | Reasonable proof-of-concept fix, but should be documented as a receiver-side estimator rather than an exact measurement |
| Flat-fading channel granularity | Paper uses scalar block-fading `H_t` | Implementation injects a delay-dependent phase slope across pilot subcarriers so empirical sensing can be formed | Modeling extension | Reasonable for sensing, but should be documented explicitly |
| Sensing section uses observable-only philosophy | Paper explicitly moved away from CRLB-based sensing in the final version | Optional `crlb` mode reintroduces a theoretical quantity | Historical extension | Use only for ablation/comparison, not as main final-paper mode |

## B. Numerical Values Chosen Because the Paper Does Not Fix Them

These are not paper facts. They are implementation assumptions.

| Parameter | Current Value | Why It Exists |
| --- | --- | --- |
| Number of subcarriers `K` | `40` | Chosen to make action ratios map cleanly to integer pilot counts |
| OFDM symbols per frame `N` | `8` | Practical small frame size for simulation |
| Average power `P` | `1.0` | Normalized power convention |
| Noise variance `sigma^2` | `0.05` | Chosen to keep estimator behavior visible without making the channel trivial |
| Symbol duration `T_s` | `66.666e-6` | Corresponds to `15 kHz` subcarrier spacing scale |
| Subcarrier spacing `Delta f` | `15 kHz` | Generic OFDM-like assumption |
| CFO mean reversion `alpha_c` | `0.95` | Stable bounded CFO dynamics for RL |
| CFO innovation std | `5 Hz` | Produces noticeable but not explosive drift |
| Channel correlation `alpha` | `0.98` | Slowly varying block fading |
| Target delay | `1.2e-6 s` | Needed to make the sensing proxy meaningful |
| Delay search window | `+/- 3e-6 s` | Chosen around the target delay |
| Delay bins | `41` | Small but usable grid resolution |
| Episode length | `64` frames | Training/evaluation convention |

## C. Reward-Design Choices Not Fixed by the Paper

| Item | Current Implementation | Why |
| --- | --- | --- |
| Default reward weights | `lambda_rate = 1.0`, `lambda_sync = 1.0`, `lambda_sensing = 0.5` | Placeholder starting point only |
| Reward normalization | Enabled by default | Makes terms comparable before weight tuning |
| Rate reference `R_ref` | Nominal high-rate reference from the action set | Keeps normalized rate near order-1 |
| Sync reference `E_ref` | `2 * sigma^2` | Natural scale for the mismatch of two noisy pilot observations |
| Sensing reference `S_ref` | `1.0` in empirical mode, CRLB at default action `(rho=0.2, beta=0.4)` in CRLB mode | Matches the scale of the selected sensing metric |

Critical note:

The paper does not tell us how to choose `lambda_1`, `lambda_2`, and `lambda_3`. Therefore any specific values are implementation choices and must be justified experimentally.

## D. Sensing-Metric Decision Record

### Paper-faithful recommendation

Use:

- `S_t` as an observable empirical proxy
- derived from the delay-correlation function
- reported as something like peak sharpness, sidelobe-to-peak ratio, or mainlobe width

### Current implemented empirical choice

The current empirical mode computes:

1. CFO-compensated pilot-domain channel estimates
2. a delay-correlation profile over a delay grid
3. `S_t = mean sidelobe power / peak power`

Interpretation:

- lower `S_t` is better
- smaller sidelobes relative to the main peak imply better delay resolvability

### Why this is recommended

- observable from received pilots
- consistent with the final draft's observable-only philosophy
- usable in both simulation and a future proof-of-concept hardware setting

### What is not recommended as the main final-paper choice

Using CRLB as the main `S_t` in the final implementation is not ideal if the goal is strict observability, because the final draft explicitly moved away from that.

Best practice:

- use empirical `S_t` as the main implementation
- optionally keep CRLB as an ablation or earlier-draft comparison

## E. RL / Optimization Engineering Choices Outside the Paper

| Item | Current Value / Choice | Type |
| --- | --- | --- |
| Network type | Pure-Python MLP | Engineering choice |
| Hidden layers | `(48, 48)` | Engineering choice |
| Activation | ReLU | Engineering choice |
| Optimizer | Adam | Engineering choice |
| Replay capacity | `4000` | Engineering choice |
| Batch size | `32` | Engineering choice |
| Discount factor | `0.98` | Engineering choice |
| Epsilon schedule | `1.0` to `0.05` over `2000` steps | Engineering choice |
| Warmup steps | `128` | Engineering choice |
| Target update interval | `100` | Engineering choice |
| State scaling | feature scaling, log-scaling, clipping before the network | Engineering choice |

These are normal RL implementation details, but they are not specified by the paper.

## F. Evaluation Additions Not in the Paper

The following were added for proof-of-concept evaluation:

- fixed-action baselines for all 12 `(rho, beta)` combinations
- random baseline
- heuristic baseline
- markdown summary tables
- learned action-distribution report

These are not divergences in the negative sense. They are extra evaluation scaffolding.

## G. Additional Modeling Simplifications

| Area | Simplification | Note |
| --- | --- | --- |
| OFDM data plane | Data subcarriers are not fully simulated symbol-by-symbol for decoding | Rate is computed analytically from the impairment model |
| Channel estimation | Channel power estimate is formed from pilot-domain averages | Acceptable for this simulator, but not a full estimator study |
| Pilot placement | Pilots are placed on evenly spaced subcarriers | Paper does not specify pilot-placement pattern |
| Delay target dynamics | Delay target is static in the current code | Paper does not define delay dynamics |

## H. Items That Need a Final Decision Before Locking the Project

1. Whether the final reported implementation should use:
   - empirical `S_t`
   - CRLB `S_t`
   - both, with one treated as ablation

2. Whether the final report should present the observable SINR proxy as:
   - the main communication-quality state variable
   - a measurable estimate of Equation (9)
   - or a proof-of-concept substitute for Equation (9)

3. How `lambda_1`, `lambda_2`, and `lambda_3` should be chosen:
   - by a small grid search
   - by heuristic tuning
   - by explicit priority constraints

4. Which metrics will be primary in the final report:
   - average reward
   - average spectral efficiency
   - synchronization mismatch
   - sensing metric

## I. Recommended Decision Path

If the goal is to stay closest to `REFERENCE_MODEL_FINAL.md`, the safest path is:

1. keep empirical `S_t` as the default
2. keep reward normalization, but disclose it as an implementation aid
3. tune reward weights by grid search after normalization
4. compare DQN against fixed, random, and heuristic baselines
5. document every non-paper value from Section B above in the final report

## J. Short Critical Summary

The current implementation is:

- structurally close to the final paper
- not numerically unique, because the paper leaves several values unspecified
- extended with practical RL machinery
- extended with optional comparison modes

The biggest points to watch are:

- the exact definition of `S_t`
- the exact justification of the observable SINR proxy derived from `\mathcal{E}_t`
- the justification for reward weighting
