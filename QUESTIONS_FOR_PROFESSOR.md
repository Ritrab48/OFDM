# Questions for Professor / Paper Author

## Purpose

This document is the **current** supervisor-facing checklist.

It is not a history of every question that ever existed. It only records:

- what has already been resolved
- what is still open
- what should be confirmed before final experiments and paper writing

For the full implementation context, read:

- `PROJECT_HANDOFF.md`
- `IMPLEMENTATION_DIVERGENCE_LOG.md`
- `ENDGAME_PLAN.md`

## 1. What Has Already Been Resolved

The following project decisions are already settled enough to proceed:

### Environment setup

The current implementation assumptions are acceptable for now:

- `K = 40`
- `N = 8`
- `P = 1.0`
- `sigma^2 = 0.05`
- `T_s = 66.666e-6`
- `Delta f = 15 kHz`
- CFO mean reversion `0.95`
- CFO innovation standard deviation `5 Hz`
- channel correlation `0.98`

These are still implementation choices, but they do not currently need clarification.

### Project scope

The project is being treated as:

- a proof-of-concept simulation study
- not a hardware deployment project

### Modeling assumptions

The following assumptions are acceptable for this project:

- repeated pilots across adjacent OFDM symbols
- quasi-static / block fading over the frame
- frame-level action updates

### Baselines

The project will compare the DQN against:

- all fixed `(rho, beta)` action policies
- random policy
- heuristic policy

### Reporting direction

The report should include:

- reward-related plots
- baseline comparison tables
- communication, synchronization, and sensing metrics
- action-distribution information

### Main project aim

The main goal is:

- to demonstrate adaptive pilot/power control

## 2. What Is Still Open

These are the main questions that still matter.

## 3. Questions Still Worth Asking

### A. Reward-weight selection criterion

This is the most important unresolved item.

Current issue:

- the project still needs a final rule for selecting `lambda_1`, `lambda_2`, and `lambda_3`

What needs confirmation:

1. Should reward weights be chosen mainly by:
   - how smoothly the reward-vs-episodes curve settles
   - the final reward value
   - performance against baselines
   - a hybrid of the above

2. If smooth convergence is the main criterion, how should it be measured?
   For example:
   - low variance in the last training window
   - low slope in the last training window
   - moving-average smoothness

3. Is reward normalization acceptable as an implementation aid before tuning the weights?
   Current implementation:
   - normalize reward terms by reference scales
   - then tune the lambdas on top of those normalized terms

### B. State SINR term

This is the second major unresolved item.

Current issue:

- the paper wants measurable state variables
- the exact analytical SINR in the paper is not directly measurable online in exact form

Current implementation:

- the state uses an observable SINR proxy derived from:
  - synchronization mismatch `mathcal{E}_t`
  - pilot power
  - noise variance
  - estimated channel power

What needs confirmation:

4. Is this observable SINR proxy acceptable for the project?

5. In the paper/report, should this term be described as:
   - a measurable estimate of Equation (9)
   - an observable proxy for communication quality
   - a proof-of-concept substitute for the exact analytical SINR

### C. Sensing metric wording

This is less urgent than the two items above, but still worth locking.

Current implementation:

- `S_t` is an empirical observable metric based on delay-correlation sharpness
- specifically:
  - build a delay-correlation profile
  - define `S_t = mean sidelobe power / peak power`

What needs confirmation:

6. Is this exact empirical form of `S_t` acceptable to report as the final sensing metric?

7. Should CRLB-based sensing appear at all in the writeup?
   Recommended options:
   - not at all
   - appendix only
   - ablation / comparison only

## 4. Suggested Short Version to Send

If a short message is needed, use this:

---

I have the proof-of-concept implementation working, and most modeling choices are already fixed. I only need confirmation on the remaining final-report decisions:

1. How should the reward weights `lambda_1`, `lambda_2`, and `lambda_3` be selected?
   Should we prioritize smooth settling of the reward-vs-episodes curve, final reward, baseline comparison, or a hybrid rule?

2. Is the current observable SINR term acceptable?
   The implementation uses a measurable proxy derived from the synchronization mismatch, pilot power, noise variance, and estimated channel power, rather than an exact hidden-state SINR.

3. Is the current empirical sensing metric acceptable as the final `S_t`?
   Right now it is implemented as mean sidelobe power divided by correlation-peak power.

4. Should CRLB-based sensing appear anywhere in the final report, or should the report stay with the observable sensing metric only?

---

## 5. Bottom Line

This file used to be broader, but most of those earlier questions have already been resolved.

The only major issues still needing confirmation are:

1. reward-weight selection rule
2. observable SINR term
3. final wording and role of the sensing metric
