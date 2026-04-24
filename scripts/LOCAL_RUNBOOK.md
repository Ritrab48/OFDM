# Local Oracle Sweep Runbook

This branch is intended to run locally on a spare machine.

## Prerequisites

- `python3`
- `python3 -m venv`
- internet access only if cloning the repository

## Setup

```bash
git clone <repo-url>
cd FYP-8-SEM
git checkout analysis/phase2b-oracle-diagnostics
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy
python -m unittest discover -s tests
```

## Smoke Test

```bash
./scripts/run_oracle_sweep.sh smoke
```

This should finish quickly and write:

- `artifacts/ec2_oracle_sweep_smoke/`

## Focused Screening Run

Use the spare laptop for this run:

```bash
ORACLE_SWEEP_WORKERS=5 ./scripts/run_oracle_sweep.sh focused_e10
```

For a Ryzen 5 5600H, `5` workers is a reasonable starting point. If the laptop stays responsive, try `6`. If it heats up too much or becomes sluggish, drop to `4`.

This writes:

- `artifacts/phase2b_oracle_sweep_focused_e10/`

## After The Run

Open:

- `artifacts/phase2b_oracle_sweep_focused_e10/oracle_sweep_report.md`
- `artifacts/phase2b_oracle_sweep_focused_e10/oracle_sweep_results.csv`

Use those to choose the top few environment-weight combinations worth confirming at higher evaluation budgets or training with DQN.
