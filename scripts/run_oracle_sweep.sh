#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
PROFILE="${1:-smoke}"
if [[ $# -gt 0 ]]; then
  shift
fi

case "$PROFILE" in
  smoke)
    ARGS=(
      --frames 8
      --eval-episodes 1
      --seeds 7
      --weights 1.0:0.2:0.4
      --regime-switch-prob 0.05
      --noise-variance 0.05
      --cfo-high-drift-std 100
      --cfo-high-drift-alpha 0.7
      --workers 1
      --output-dir artifacts/ec2_oracle_sweep_smoke
    )
    ;;
  focused_e10)
    ARGS=(
      --frames 64
      --eval-episodes 10
      --seeds 7,17,27,37,47
      --weights 1.0:0.2:0.4,1.0:0.25:0.5,1.0:0.4:0.5,1.0:0.5:0.4
      --regime-switch-prob 0.02,0.05
      --noise-variance 0.05,0.1
      --cfo-high-drift-std 300,600,1000
      --cfo-high-drift-alpha 0.7
      --workers "${ORACLE_SWEEP_WORKERS:-4}"
      --output-dir artifacts/phase2b_oracle_sweep_focused_e10
    )
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Supported profiles: smoke, focused_e10" >&2
    exit 2
    ;;
esac

if [[ $# -gt 0 ]]; then
  ARGS+=("$@")
fi

exec "$PYTHON_BIN" -m ofdm_isac_rl.oracle_sweep "${ARGS[@]}"
