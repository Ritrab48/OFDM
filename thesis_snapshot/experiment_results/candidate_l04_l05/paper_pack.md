# Paper Writing Pack

## Experimental Setup

- Sensing mode: `empirical`
- Reward normalization: `True`
- Episodes per run: `100`
- Frames per episode: `64`
- Evaluation episodes per run: `5`
- Candidate `lambda_sync` values: `[0.4]`
- Candidate `lambda_sensing` values: `[0.5]`

## Best Weight Setting

- Selection mode: `hybrid`
- Stability window: `20` episodes
- `lambda_rate = 1.0`
- `lambda_sync = 0.4`
- `lambda_sensing = 0.5`
- Mean DQN reward across seeds: `28.8483`
- Final-window mean reward: `32.1680`
- Final-window reward std: `5.8679`
- Final-window reward slope abs: `0.2902`
- Stability score: `2.5121`
- Reward gap to best fixed baseline: `-10.5675`
- Baseline-gap factor: `0.0864`

## Interpretation Notes

- In `selection_mode=hybrid`, unstable curves are penalized and negative reward gaps to the best fixed baseline are penalized rather than ignored.
- The best setting is chosen by the configured selection rule, not by mean reward alone unless `selection_mode=reward` is used.
- Empirical `S_t` is the paper-faithful default because it preserves the observable-only philosophy.
- If the DQN does not beat the best fixed baseline under the current sweep, that should be stated honestly as a proof-of-concept limitation.

## Main Tables

- `sweep_results.csv`: all weight combinations ranked by the configured selection score
- `best_policy_summary.csv`: DQN and all baselines for the selected weight setting
- `best_history.json`: mean training history across seeds for the selected weight setting

## Main Figures

- `weight_sweep_reward.svg`
- `best_reward_curve.svg`
- `best_policy_reward.svg`
- `best_policy_rate.svg`
- `best_policy_sync.svg`
- `best_policy_sensing.svg`
- `best_action_distribution.svg`

## Remaining Work

- Finalize how the observable SINR proxy should be described in the paper.
- Run a larger multi-seed sweep once faster numerical tooling is available or more runtime is acceptable.
- Decide whether CRLB sensing should appear only as an ablation or also as a historical comparison.
- If the best fixed baseline still beats DQN, improve training length, state design, or reward tuning before claiming adaptive superiority.

## Links to Audit Notes

- See `IMPLEMENTATION_DIVERGENCE_LOG.md` for a full paper-vs-implementation audit.
