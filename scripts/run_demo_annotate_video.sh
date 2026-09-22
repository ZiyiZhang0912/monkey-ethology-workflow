#!/usr/bin/env bash
# 用已有 sessions 分类结果，在原始视频上叠加 Behavior 标注。
# 默认只导出 562-day1 前 750 帧预览；去掉 --max-frames 可整段导出。
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src:${PYTHONPATH:-}
export MPLCONFIGDIR=./output/.mplconfig
mkdir -p "$MPLCONFIGDIR" data/sample/demo_cohort/videos

STEM="${1:-562-day1}"
MAX_FRAMES="${2:-750}"

python3 -m monkey_ethology annotate-video \
  --config configs/demo_study.yaml \
  --output ./output/demo_viz \
  --video-dir data/sample/demo_cohort/videos \
  --data-dir ./output/demo_viz/sessions \
  --stem "$STEM" \
  --max-frames "$MAX_FRAMES"

echo
echo "标注视频: $(pwd)/output/demo_viz/annotated_videos/"
ls -la output/demo_viz/annotated_videos/ 2>/dev/null | head -20
