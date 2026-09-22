#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=./output/.mplconfig
mkdir -p "$MPLCONFIGDIR"

python3 -m monkey_ethology run \
  --config configs/demo_study.yaml \
  --input data/sample/demo_cohort/raw \
  --pattern "*.csv" \
  --cage default \
  --animals 562,883 \
  --days 1,2,3 \
  --output ./output/demo_viz

echo
echo "结果目录: $(pwd)/output/demo_viz/figures"
ls -1 output/demo_viz/figures 2>/dev/null | head -80
