# Project Handoff

## 1. What This Repository Is

This repository is a proof-of-concept implementation of the model described in `REFERENCE_MODEL_FINAL.md`.

The project studies adaptive pilot-resource control for an OFDM-ISAC system using reinforcement learning. At each frame, the controller chooses:

- pilot ratio `rho_t`
- pilot power fraction `beta_t`

The objective is to trade off:

- communication performance
- synchronization quality
- sensing quality

The current RL controller is a DQN.

This codebase is intended as:

- a runnable simulation implementation
- a reference for experiments and paper writing
- a starting point for further student work

It is **not** a hardware-ready system.

## 2. Who Should Read This

This document is written so that:

- a professor can understand the current project status and assumptions
- a student can continue the work without prior context
- an AI agent can enter the repository and know where to look first

If someone only reads one file before working on the repo, it should be this one.

## 3. Recommended Reading Order

For a new person entering the project, read in this order:

1. `PROJECT_HANDOFF.md`
2. `REFERENCE_MODEL_FINAL.md`
3. `README.md`
4. `IMPLEMENTATION_DIVERGENCE_LOG.md`
5. `ENDGAME_PLAN.md`
6. `QUESTIONS_FOR_PROFESSOR.md`

Purpose of each:

- `PROJECT_HANDOFF.md`: one-stop overview of the project
- `REFERENCE_MODEL_FINAL.md`: target paper/model specification
- `README.md`: quick run/test commands
- `IMPLEMENTATION_DIVERGENCE_LOG.md`: exact list of paper-vs-code differences
- `ENDGAME_PLAN.md`: what remains to finish the project
- `QUESTIONS_FOR_PROFESSOR.md`: clarification checklist for supervisor/author

## 4. Project Goal in Plain Language

The goal is to show that a DQN-based controller can adapt pilot allocation and pilot power in an OFDM-ISAC system, rather than using a fixed pilot design.

At each frame:

- the receiver observes the previous frame's measurements
- the DQN chooses one action `(rho_t, beta_t)`
- the system then produces communication, synchronization, and sensing outcomes
- the reward combines those outcomes

The scientific purpose is to demonstrate adaptive pilot/power control in a proof-of-concept setting.

## 5. Current High-Level Status

### What is already implemented

The following are already working:

- environment simulator
- CFO estimation from repeated pilots
- observable synchronization metric `mathcal{E}_t`
- empirical sensing metric `S_t`
- communication-rate model
- DQN training loop
- replay buffer, target network, epsilon-greedy exploration
- fixed, random, and heuristic baselines
- training CLI
- reward-weight sweep CLI
- SVG chart generation without external plotting libraries
- unit tests

### What is not yet final

The following are still project decisions rather than locked results:

- how the paper will describe the observable SINR term in the state
- how reward weights will be selected in the final report
- which final experiment budget will be used
- whether CRLB sensing appears at all, and if so only as an ablation or historical comparison

### Practical status summary

The core system model and DQN are implemented well enough to move into final experiment design and paper assembly.

The project is **not** blocked on missing core code.

## 6. Main Files and What They Do

### Reference and documentation

- `REFERENCE_MODEL_FINAL.md`
  - target model description
- `README.md`
  - quick start and commands
- `IMPLEMENTATION_DIVERGENCE_LOG.md`
  - all known differences, assumptions, and engineering choices
- `ENDGAME_PLAN.md`
  - closing-phase experiment and reporting plan
- `QUESTIONS_FOR_PROFESSOR.md`
  - clarification checklist

### Core code

- `ofdm_isac_rl/config.py`
  - physical and RL configuration
- `ofdm_isac_rl/environment.py`
  - environment dynamics, measurements, reward, state construction
- `ofdm_isac_rl/dqn.py`
  - DQN agent and pure-Python MLP
- `ofdm_isac_rl/baselines.py`
  - fixed, random, and heuristic policies
- `ofdm_isac_rl/train.py`
  - main train/evaluate entry point
- `ofdm_isac_rl/sweep.py`
  - reward-weight sweep and paper-pack generation
- `ofdm_isac_rl/reporting.py`
  - SVG plot generation

### Tests

- `tests/test_reference_model.py`
  - smoke tests and a regression check for the ICI/SINR equations

## 7. Model Summary

### Hidden environment dynamics

The environment includes:

- mean-reverting CFO process
- correlated block fading channel
- additive noise

These evolve frame by frame.

### Action space

The action is the pair `(rho_t, beta_t)` with:

- `rho in {0.1, 0.2, 0.3, 0.4}`
- `beta in {0.2, 0.4, 0.6}`

Total action count:

- `12`

### State used by the DQN

The current state contains:

- estimated CFO
- observable synchronization mismatch
- communication-quality term labeled as SINR in code
- pilot-domain SNR
- estimated channel power
- previous pilot ratio
- previous pilot power fraction

Important:

- the state uses delayed frame information
- the network input is numerically scaled before entering the DQN
- scaling is only for learning stability, not a physics change

### Reward

The reward combines:

- communication rate
- synchronization penalty
- sensing penalty

The code supports reward normalization before weighting so that the terms are on comparable scales.

## 8. Important Scientific Decisions and Their Current Status

### A. Synchronization metric

Current position:

- use the observable mismatch metric `mathcal{E}_t`

This aligns with the final draft's measurable-only philosophy.

### B. Sensing metric

Current default:

- empirical sensing metric

Implemented form:

- build a delay-correlation profile from CFO-compensated pilot-domain channel estimates
- define `S_t` as mean sidelobe power divided by peak power

Interpretation:

- lower `S_t` is better

Why this was chosen:

- it is observable
- it fits the final draft better than CRLB
- it is usable in a proof-of-concept receiver

### C. CRLB sensing mode

The code still supports a `crlb` sensing mode.

Reason it exists:

- earlier drafts and discussion considered CRLB-style sensing

Current recommendation:

- do not use CRLB as the main result
- use it only if needed for ablation or historical comparison

### D. SINR term in the state

This is one of the most important points in the repo.

Problem:

- the paper wants measurable state variables
- the analytical SINR equation depends on residual CFO and channel terms that are not directly known in exact form online

Current implementation decision:

- use an observable SINR proxy derived from:
  - synchronization mismatch `mathcal{E}_t`
  - known pilot power
  - known noise variance
  - estimated channel power

Why this matters:

- this is a reasonable proof-of-concept engineering interpretation
- but it should be described honestly in the paper as an observable estimate or proxy, not as an exact direct measurement of the analytical SINR

### E. Reward weights

Current default code values:

- `lambda_rate = 1.0`
- `lambda_sync = 1.0`
- `lambda_sensing = 0.5`

These are placeholders, not final research conclusions.

Open project question:

- how the final reward weights should be selected

## 9. Numerical Values in the Current Implementation

These are current implementation assumptions, not necessarily paper-mandated values:

- number of subcarriers `K = 40`
- OFDM symbols per frame `N = 8`
- average power `P = 1.0`
- noise variance `sigma^2 = 0.05`
- symbol duration `T_s = 66.666e-6`
- subcarrier spacing `Delta f = 15 kHz`
- CFO mean reversion `0.95`
- CFO innovation standard deviation `5 Hz`
- channel correlation `0.98`
- episode length `64` frames by default

The professor later indicated that these values are acceptable for the project unless a better benchmark setup is provided.

## 10. RL Implementation Summary

The RL controller is a DQN with:

- pure-Python MLP
- replay memory
- target network
- epsilon-greedy action selection
- feature scaling before the network

Current default training choices:

- hidden layers `(48, 48)`
- learning rate `1e-3`
- discount factor `0.98`
- replay capacity `4000`
- batch size `32`
- epsilon decay from `1.0` to `0.05`
- warmup steps `128`
- target update interval `100`

These are engineering choices. They are not claimed to come directly from the paper.

## 11. Baselines

The code includes three baseline families:

- all fixed `(rho, beta)` actions
- random policy
- heuristic policy

These were added so the DQN can be compared against non-adaptive and simple adaptive alternatives.

## 12. Current Output and Artifact System

### Training command

Use:

```bash
python3 -m ofdm_isac_rl.train --episodes 80 --frames 64 --eval-episodes 5
```

Outputs include:

- history
- learned model parameters
- policy evaluation summary
- markdown summary

### Sweep command

Use:

```bash
python3 -m ofdm_isac_rl.sweep --output-dir artifacts/weight_sweep
```

Outputs include:

- reward-weight sweep tables
- best-policy summary table
- reward curves
- baseline-comparison charts
- action-distribution chart
- paper-writing pack

### Tests

Use:

```bash
python3 -m unittest discover -s tests
```

## 13. Important Warning About Existing Artifacts

Not all previously generated artifacts should be treated as final paper results.

Why:

- the implementation evolved during development
- the ICI/SINR formulation was corrected
- the project has not yet locked the final reward-weight selection criterion

Therefore:

- old sweep outputs are useful for exploration
- they should not be cited as final results until the final experiment protocol is fixed and rerun

## 14. What Is Already Decided vs What Still Needs Decision

### Already decided enough to proceed

- use DQN as the controller
- use the discrete `(rho, beta)` action space
- use an observable synchronization metric
- use empirical sensing as the default mode
- include fixed, random, and heuristic baselines
- treat the project as proof-of-concept rather than deployment-ready hardware work

### Still needs explicit final decision

- how to describe the observable SINR term in the paper
- how to rank reward weights
- whether CRLB appears anywhere in the final report
- what final training/sweep budget will be used

## 15. Recommended Endgame

The repository already contains a fuller version of this in `ENDGAME_PLAN.md`.

The short version is:

1. lock the reward-weight selection criterion
2. lock the wording for the observable SINR term
3. run the final weight sweep
4. evaluate against all baselines
5. save the final tables and figures
6. write the paper with all assumptions disclosed

## 16. What Still Needs to Be Done in Code

The remaining coding work is mainly experimental and reporting work, not core model construction.

Likely remaining tasks:

1. add or finalize stability-based sweep ranking if the professor wants reward-curve smoothness to drive weight selection
2. rerun the sweep after the final criterion is agreed
3. regenerate the paper-pack artifacts
4. improve figure quality if required for submission

## 17. Known Limitations

These should be stated honestly:

1. The project is a simulation proof-of-concept, not a hardware deployment.
2. The exact empirical sensing metric is an implementation choice because the paper leaves it open.
3. The state SINR is an observable proxy/estimate, not a literal exact online measurement of the analytical SINR formula.
4. Reward weights are not yet finalized by a locked experimental protocol.
5. The pure-Python implementation is convenient and dependency-free, but not optimized for large experiment budgets.

## 18. If a New Person Needs to Continue the Work

Recommended first actions:

1. read `PROJECT_HANDOFF.md`
2. read `REFERENCE_MODEL_FINAL.md`
3. read `IMPLEMENTATION_DIVERGENCE_LOG.md`
4. run the tests
5. run one short training command
6. inspect the current artifact outputs

After that, the next question to answer is:

- "What exact criterion are we using to choose reward weights?"

That decision determines the final sweep design.

## 19. If a Professor Wants a One-Paragraph Summary

This repository is a proof-of-concept OFDM-ISAC reinforcement-learning implementation based on `REFERENCE_MODEL_FINAL.md`, where a DQN adaptively selects pilot ratio and pilot power fraction to trade off communication rate, synchronization quality, and sensing quality. The simulator, DQN, baselines, training pipeline, sweep pipeline, and plotting are already implemented. The main remaining work is not core modeling but final experiment design: specifically, locking the reward-weight selection rule, deciding how to describe the observable SINR proxy in the paper, rerunning the final sweeps, and packaging the final tables and plots.

## 20. Related Documents

Use these for deeper detail:

- `IMPLEMENTATION_DIVERGENCE_LOG.md` for exact paper-vs-code differences
- `ENDGAME_PLAN.md` for final-phase execution planning
- `QUESTIONS_FOR_PROFESSOR.md` for unresolved clarification items
