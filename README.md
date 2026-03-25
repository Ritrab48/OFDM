# OFDM-ISAC Reference Implementation

Start with `PROJECT_HANDOFF.md` if you are new to the repository and need the full project context.

This repository now includes a runnable reference implementation of the model described in [REFERENCE_MODEL_FINAL.md](REFERENCE_MODEL_FINAL.md). The code is intentionally pure standard-library Python so it runs without extra package installation.

The implementation covers:

- A frame-level OFDM-ISAC simulator with mean-reverting CFO dynamics, correlated block fading, discrete pilot/power actions, and observable state variables.
- CFO estimation from repeated pilots, the residual pilot-domain mismatch metric, and two sensing modes:
  - `empirical`: pilot-domain delay-correlation sharpness proxy
  - `crlb`: CRLB-style sensing lower bound override requested by the supervisor
- Reward-term normalization so weight search is based on tradeoff preference rather than raw unit mismatch.
- A DQN agent with replay memory, target network, epsilon-greedy exploration, JSON export of the learned network weights, and baseline comparisons against fixed, random, and heuristic policies.

## Run

```bash
python3 -m ofdm_isac_rl.train --episodes 80 --frames 64 --eval-episodes 5
```

## Sweep

```bash
python3 -m ofdm_isac_rl.sweep --output-dir artifacts/weight_sweep
```

Artifacts are written to `artifacts/reference_model/` by default:

- `history.json`
- `model.json`
- `policy_results.json`
- `summary.md`

The sweep command writes a paper-pack artifact set:

- `sweep_results.csv`
- `best_policy_summary.csv`
- `paper_pack.md`
- SVG graphs for reward curves, policy comparisons, and action distribution

## Test

```bash
python3 -m unittest discover -s tests
```

## Notes

`REFERENCE_MODEL_FINAL.md` defines the sensing term as an empirical observable proxy. The current implementation also supports a `crlb` sensing mode because the project direction later requested using a CRLB-style lower bound instead.

The default training command now uses:

- empirical sensing mode
- normalized reward terms
- baseline comparisons against fixed, random, and heuristic policies

Useful overrides:

```bash
python3 -m ofdm_isac_rl.train --sensing-mode crlb --no-reward-normalize
python3 -m ofdm_isac_rl.train --lambda-rate 1.0 --lambda-sync 0.5 --lambda-sensing 0.25
```
