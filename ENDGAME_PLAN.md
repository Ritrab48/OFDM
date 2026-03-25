# Endgame Plan

## 1. Status Check

### Are the system model and DQN defined well enough to proceed?

Yes, for a proof-of-concept implementation they are defined well enough to move into the experiment and paper-writing phase.

What is already in place:

- system dynamics for CFO and block fading
- action space for adaptive pilot ratio `\rho_t` and pilot power fraction `\beta_t`
- observable synchronization metric `\mathcal{E}_t`
- empirical sensing metric `S_t`
- communication-rate model
- DQN agent, replay memory, target network, epsilon-greedy exploration
- fixed, random, and heuristic baselines
- training, evaluation, sweep, and artifact-generation scripts

### What is still not fully locked?

These are not blockers for moving forward, but they must be stated clearly in the paper:

- the final wording and justification of the observable SINR proxy used in the state
- the final criterion for choosing reward weights
- the final experiment budget: number of episodes, frames per episode, and seeds

### Practical conclusion

The core model is implemented. What remains is not "define the DQN from scratch" work. What remains is:

1. lock final experiment choices
2. run the right sweeps
3. save the right artifacts
4. write the paper around those artifacts

## 2. Reservations

These are the main reservations that should be visible before final experiments:

1. The current state SINR is an observable proxy, not a direct physical measurement of the exact analytical SINR in Equation (9).
2. The current sensing metric is a justified empirical proxy, but the paper leaves its exact formula open, so the chosen proxy must be disclosed.
3. The previous sweep artifacts under `artifacts/weight_sweep/` should be treated as exploratory only, not final, because:
   - the SINR implementation has since been corrected
   - the reward-weight success criterion is still being finalized with your professor
4. If the professor insists that reward weights should be chosen mainly by "smooth settling of reward vs episodes," then the sweep ranking logic should follow that criterion explicitly rather than using mean reward alone.

## 3. Final Decisions Still Needed

Before final experiments, the project should explicitly lock these decisions:

1. How the paper will describe the state communication-quality term:
   - measurable estimate of Equation (9)
   - observable proxy derived from `\mathcal{E}_t`
   - proof-of-concept substitute
2. How reward weights will be selected:
   - stability-first
   - performance-first
   - hybrid criterion
3. Whether CRLB-based sensing appears at all:
   - not used
   - ablation only
   - appendix/historical note only
4. Final training budget:
   - episodes per run
   - frames per episode
   - number of seeds

## 4. Reward-Weight Selection Plan

### Recommended approach

Do not choose reward weights by a single number alone. Use a two-part selection rule:

1. stability of the training curve
2. final policy quality against baselines

### Proposed selection procedure

For each candidate pair `(\lambda_sync, \lambda_sensing)` with `\lambda_rate = 1`:

1. train for the agreed number of episodes
2. repeat across multiple random seeds
3. save per-episode reward history
4. compute stability metrics on the last training window
5. evaluate the trained policy against baselines
6. rank the weight setting using a hybrid score

### Stability metrics to compute

For the last `W` episodes, compute:

- final-window mean reward
- final-window reward standard deviation
- final-window reward slope magnitude
- moving-average smoothness

Interpretation:

- higher final-window mean reward is better
- lower standard deviation is better
- lower absolute slope is better because it means the curve has settled
- smoother moving-average behavior is better

### Policy-quality metrics to compute

For the final trained policy, compute:

- average reward
- average rate `R_t`
- average synchronization metric `\mathcal{E}_t`
- average sensing metric `S_t`
- gap to best fixed baseline

### Final decision rule

Recommended rule:

- first filter out weight settings with unstable curves
- among the stable settings, choose the one with the best final reward and the best or near-best baseline gap

If your professor prefers a simpler rule, use:

- choose the setting with the smoothest settled reward curve
- then report whether that setting also gives competitive policy performance

## 5. Data That Must Be Saved

For each reward-weight setting and each seed, save:

- per-episode training reward
- per-episode average rate
- per-episode average synchronization metric
- per-episode average sensing metric
- per-episode average loss
- final evaluation metrics for DQN
- final evaluation metrics for every baseline
- action counts for DQN during evaluation

### Minimum file outputs

For the final experiment pack, the following should exist:

- sweep summary table
- per-seed results table
- best-setting training history
- best-setting baseline comparison table
- best-setting action-distribution table
- experiment configuration summary

### Strongly recommended file formats

- `csv` for tables
- `json` for detailed histories and metadata
- `svg` or `png` for plots
- one markdown summary file that explains what each artifact means

## 6. Tables Needed in the Paper

### Table 1: Simulation and RL configuration

Include:

- `K`, `N`, `P`, `\sigma^2`, `T_s`, `\Delta f`
- CFO dynamics parameters
- channel correlation
- action space values
- DQN hyperparameters
- episode length and seed count

### Table 2: Reward-weight sweep summary

Each row should contain:

- `\lambda_rate`
- `\lambda_sync`
- `\lambda_sensing`
- mean final reward
- final-window reward standard deviation
- final-window slope magnitude
- mean gap to best fixed baseline

### Table 3: Final baseline comparison

Compare:

- DQN
- best fixed baseline
- random baseline
- heuristic baseline

Columns:

- average reward
- average rate
- average sync metric
- average sensing metric

### Table 4: Best fixed-action policies

Optional but useful:

- all fixed `(rho, beta)` policies
- ranked by average reward

## 7. Plots Needed in the Paper

### Plot 1: Reward vs episodes

Purpose:

- show whether learning stabilizes
- support reward-weight selection

Recommendation:

- use moving-average reward curves
- include one curve per candidate weight setting for shortlisted settings only

### Plot 2: Final reward-weight comparison

Purpose:

- show which weight settings perform best overall

Format:

- bar chart of final mean reward or hybrid ranking score

### Plot 3: Baseline comparison

Purpose:

- show whether adaptive control beats fixed and non-adaptive strategies

Format:

- grouped bar charts for reward, rate, sync metric, and sensing metric

### Plot 4: Action distribution of learned policy

Purpose:

- show that the learned policy is adaptive rather than collapsing to one fixed action

### Plot 5: Optional metric curves

If space allows:

- rate vs episodes
- sync metric vs episodes
- sensing metric vs episodes

These help explain which objective the DQN is trading off.

## 8. Metrics to Discuss in the Paper

Primary metrics:

- average reward
- average spectral efficiency / rate
- average synchronization metric `\mathcal{E}_t`
- average sensing metric `S_t`

Secondary metrics:

- reward stability in the final training window
- gap to best fixed baseline
- learned action diversity

Important reporting rule:

- never report reward alone without also reporting rate, sync, and sensing

## 9. Recommended Experiment Sequence

### Phase A: Lock decisions

1. confirm professor's preferred reward-weight criterion
2. confirm how the SINR proxy should be described
3. confirm whether CRLB appears only as historical context or not at all

### Phase B: Final weight-selection sweep

1. run the weight sweep using the locked criterion
2. save all per-seed histories
3. shortlist the best settings

### Phase C: Final comparison runs

1. rerun the selected best setting with the full training budget
2. evaluate against all baselines
3. generate final tables and figures

### Phase D: Paper assembly

1. write system model and assumptions
2. write reward design and weight-selection method
3. present results
4. state limitations honestly

## 10. Remaining Implementation Work

If the current implementation direction is accepted, the remaining coding work is modest and mainly experimental:

1. add a stability-based ranking mode to the sweep script
2. regenerate the paper pack after the new sweep
3. optionally improve figure quality if needed for submission
4. optionally add extra exports for per-seed histories if the current outputs are not enough for the paper

## 11. Plain Answer

Are we done with defining the system model and DQN?

- For a proof-of-concept: essentially yes.
- For a locked final paper claim: not fully, because the SINR-state wording and reward-weight criterion still need to be fixed explicitly.

What is left?

- finalize the selection rule for reward weights
- run the final sweeps with the correct criterion
- save the final artifact set
- write the paper around those final artifacts
