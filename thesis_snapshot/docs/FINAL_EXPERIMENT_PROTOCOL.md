# Final Experiment Protocol

## Purpose

This document defines the exact final experiment sequence for the current proof-of-concept project.

It is based on the current state of the codebase and the latest comparison runs.

Current best candidate:

- `lambda_rate = 1.0`
- `lambda_sync = 0.25`
- `lambda_sensing = 0.5`

Current evidence:

- this setting beat the best fixed baseline in the longer 100-episode comparison run
- the other two shortlisted settings did not

## Locked Defaults

Use these defaults unless there is a deliberate decision to change them:

- sensing mode: `empirical`
- reward normalization: enabled
- state SINR term: observable proxy as currently implemented
- action space:
  - `rho in {0.1, 0.2, 0.3, 0.4}`
  - `beta in {0.2, 0.4, 0.6}`
- project scope: proof-of-concept simulation study

## Final Questions This Protocol Answers

This protocol is meant to answer:

1. Which reward weights should be used in the final report?
2. Does the DQN beat the best fixed baseline under the chosen weights?
3. What tables and plots should be saved for the paper?

## Experiment Structure

The final experiment has three phases:

1. local weight refinement
2. final weight confirmation
3. final report artifact generation

## Phase 1: Local Weight Refinement

### Goal

Refine the reward weights locally around the current best region instead of stopping at the coarse grid.

### Why this phase is needed

Yes, further tuning is appropriate here because:

- the current winner came from a coarse multiplicative grid
- the reward weights are continuous design choices, not fixed categorical settings
- the longer comparison runs showed that coarse-sweep ranking alone was not decisive

However, tuning should be limited to **one local refinement round**, not endless trial-and-error. Otherwise the project risks overfitting to this exact simulator setup and seed set.

### Region to refine

Use the current promising region:

- `lambda_sync in {0.2, 0.25, 0.3, 0.4}`
- `lambda_sensing in {0.4, 0.5, 0.6, 0.75}`

Keep:

- `lambda_rate = 1.0`

### Runtime budget

Use:

- `episodes = 60`
- `frames = 64`
- `eval_episodes = 3`
- `seeds = 7,17,27`
- `stability_window = 15`

### Command

Run:

```bash
cd ~/dev/FYP-8-SEM/FYP-8-SEM

python3 -m ofdm_isac_rl.sweep \
  --episodes 60 \
  --frames 64 \
  --eval-episodes 3 \
  --seeds 7,17,27 \
  --lambda-rate 1.0 \
  --lambda-sync 0.2,0.25,0.3,0.4 \
  --lambda-sensing 0.4,0.5,0.6,0.75 \
  --stability-window 15 \
  --selection-mode hybrid \
  --sensing-mode empirical \
  --reward-normalize \
  --output-dir artifacts/final_refine_local
```

### Output of Phase 1

From `artifacts/final_refine_local/sweep_results.csv`, choose the top **two** settings by:

1. positive or least-negative `mean_gap_to_best_fixed`
2. then `selection_score`
3. then `mean_dqn_reward`

Carry only those top two settings into Phase 2.

## Phase 2: Final Weight Confirmation

### Goal

Confirm the best reward-weight setting using longer runs and multiple seeds.

### Settings to compare

Run the top **two** settings from Phase 1.

If Phase 1 cannot be run for runtime reasons, use these fallback settings:

1. `lambda_sync = 0.25`, `lambda_sensing = 0.5`
2. `lambda_sync = 0.25`, `lambda_sensing = 0.25`
3. `lambda_sync = 0.5`, `lambda_sensing = 0.5`

### Runtime budget

Use:

- `episodes = 100`
- `frames = 64`
- `eval_episodes = 5`
- `seeds = 7,17,27,37,47`
- `stability_window = 20`

### Commands

Run these from:

```bash
cd ~/dev/FYP-8-SEM/FYP-8-SEM
```

#### Confirmation Run Template

```bash
python3 -m ofdm_isac_rl.sweep \
  --episodes 100 \
  --frames 64 \
  --eval-episodes 5 \
  --seeds 7,17,27,37,47 \
  --lambda-rate 1.0 \
  --lambda-sync <chosen_sync> \
  --lambda-sensing <chosen_sensing> \
  --stability-window 20 \
  --selection-mode hybrid \
  --sensing-mode empirical \
  --reward-normalize \
  --output-dir artifacts/final_confirm_<label>
```

### Fallback commands if skipping Phase 1

#### Candidate A

```bash
python3 -m ofdm_isac_rl.sweep \
  --episodes 100 \
  --frames 64 \
  --eval-episodes 5 \
  --seeds 7,17,27,37,47 \
  --lambda-rate 1.0 \
  --lambda-sync 0.25 \
  --lambda-sensing 0.5 \
  --stability-window 20 \
  --selection-mode hybrid \
  --sensing-mode empirical \
  --reward-normalize \
  --output-dir artifacts/final_confirm_l025_l05
```

#### Candidate B

```bash
python3 -m ofdm_isac_rl.sweep \
  --episodes 100 \
  --frames 64 \
  --eval-episodes 5 \
  --seeds 7,17,27,37,47 \
  --lambda-rate 1.0 \
  --lambda-sync 0.25 \
  --lambda-sensing 0.25 \
  --stability-window 20 \
  --selection-mode hybrid \
  --sensing-mode empirical \
  --reward-normalize \
  --output-dir artifacts/final_confirm_l025_l025
```

#### Candidate C

```bash
python3 -m ofdm_isac_rl.sweep \
  --episodes 100 \
  --frames 64 \
  --eval-episodes 5 \
  --seeds 7,17,27,37,47 \
  --lambda-rate 1.0 \
  --lambda-sync 0.5 \
  --lambda-sensing 0.5 \
  --stability-window 20 \
  --selection-mode hybrid \
  --sensing-mode empirical \
  --reward-normalize \
  --output-dir artifacts/final_confirm_l05_l05
```

## Phase 2 Decision Rule

### Primary rule

Choose the final weight setting using this order:

1. positive `mean_gap_to_best_fixed`
2. highest `mean_dqn_reward`
3. highest `stability_score`

### Interpretation

- if a setting has a positive `mean_gap_to_best_fixed`, it means the DQN beats the best fixed baseline on the reward objective
- that is more important than raw training reward alone
- if more than one setting has a positive gap, choose the one with the better reward and stable behavior

### If no setting has a positive gap

Then choose the setting with:

1. the least negative `mean_gap_to_best_fixed`
2. acceptable stability
3. the best overall reward among those near the top

And report honestly that the DQN did not consistently beat the best fixed baseline.

## Phase 3: Final Report Artifact Generation

### Goal

Once the final weight setting is chosen, keep the artifact directory from that winning run as the final paper pack.

No additional rerun is required unless:

- the winning run was interrupted
- the artifact set is incomplete
- a larger seed count is later requested

### Final artifact directory

The winning directory should contain:

- `paper_pack.md`
- `sweep_results.csv`
- `sweep_results.json`
- `best_config.json`
- `best_history.json`
- `best_policy_summary.csv`
- `best_policy_summary.json`
- `best_reward_curve.svg`
- `best_policy_reward.svg`
- `best_policy_rate.svg`
- `best_policy_sync.svg`
- `best_policy_sensing.svg`
- `best_action_distribution.svg`

## What To Compare After Phase 2

For each candidate directory from Phase 2, compare these files:

- `paper_pack.md`
- `best_config.json`
- `best_policy_summary.csv`

### Metrics to compare

The key metrics are:

1. `mean_gap_to_best_fixed`
2. `mean_dqn_reward`
3. `avg_rate_mean` for DQN
4. `avg_sync_mean` for DQN
5. `avg_sensing_mean` for DQN
6. DQN action distribution

### What counts as a good result

The desired pattern is:

- DQN reward at or above the best fixed baseline
- strong communication rate
- sync mismatch not worse in a meaningful way
- sensing metric still acceptable
- action distribution not completely collapsed to one obviously inferior fixed action

## Exact Files To Use In The Paper

Use the winning directory's:

- `paper_pack.md` for summary language
- `best_policy_summary.csv` for the main policy-comparison table
- `best_reward_curve.svg` for reward-vs-episode behavior
- `best_policy_reward.svg` for reward comparison
- `best_policy_rate.svg` for rate comparison
- `best_policy_sync.svg` for synchronization comparison
- `best_policy_sensing.svg` for sensing comparison
- `best_action_distribution.svg` for adaptation behavior

## Suggested Paper Claims

These are acceptable only if supported by the final winning run:

### Safe claims

- the DQN learns a nontrivial adaptive policy over pilot ratio and pilot power
- the policy can outperform simple baselines under the chosen reward weights
- the tradeoff between rate, synchronization, and sensing is weight-dependent

### Claims to avoid unless strongly supported

- universal superiority over all fixed policies in all regimes
- hardware readiness
- exact observability of the analytical SINR expression

## What To Tell The Professor After Running This

Use this structure:

1. We ran a local refinement sweep around the current best weight region.
2. We selected the top two candidates from that sweep.
3. We reran those candidates with longer training and five seeds.
4. We selected the final setting by:
   - whether DQN beat the best fixed baseline
   - then reward
   - then stability
5. The winning setting was `(...)`.
6. Under that setting:
   - DQN reward was `...`
   - best fixed reward was `...`
   - rate was `...`
   - sync was `...`
   - sensing was `...`

## If You Want The Short Answer

Yes, further tuning is appropriate, but only locally and only once.

Do:

1. one local refinement sweep around `(0.25, 0.5)`
2. one longer confirmation round on the top two settings

Do not:

1. keep expanding the lambda search indefinitely
2. keep changing seeds and grids until one result looks good

That would weaken the credibility of the final result.

## If Runtime Is Too High

Use this fallback only if needed:

1. skip Phase 1
2. run the three fallback settings already identified
3. keep `episodes = 100`
4. reduce seeds to `7,17,27`

Do not reduce below this unless absolutely necessary, because the final result becomes much less convincing.

## Final Recommendation

The strongest current candidate is still:

- `lambda_rate = 1.0`
- `lambda_sync = 0.25`
- `lambda_sensing = 0.5`

But the updated protocol now formally allows a last local refinement step before locking it in.
   - sync was `...`
   - sensing was `...`
5. Therefore the final report will use that setting.

## If Runtime Is Too High

Use this fallback only if needed:

- keep the same three candidate settings
- keep `episodes = 100`
- reduce seeds to `7,17,27`

Do not reduce below this unless absolutely necessary, because the final result becomes much less convincing.

## Final Recommendation

Based on current evidence, the expected winner is:

- `lambda_rate = 1.0`
- `lambda_sync = 0.25`
- `lambda_sensing = 0.5`

But the final decision should still be made by Phase 1 using the commands above.
