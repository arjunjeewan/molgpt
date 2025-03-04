#!/usr/bin/env bash
# Sequential, idempotent multi-seed training driver for the arch ablation.
# Reuses existing seed-42 ckpts (unconditional_moses{,_baseline}.pt); trains seeds 1 & 2
# for both archs, plus one longer (25-epoch) modernized run. Skips any ckpt already present.
set -u
# Activate your environment first (see ../environment.yml):  conda activate molgpt
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
W=../cond_gpt/weights
LOG=experiments/logs

# arch seed epochs run_name
RUNS=(
  "baseline 1  10 unconditional_moses_baseline_s1"
  "modern   1  10 unconditional_moses_modern_s1"
  "baseline 2  10 unconditional_moses_baseline_s2"
  "modern   2  10 unconditional_moses_modern_s2"
  "modern   42 25 unconditional_moses_modern_e25"
)

for spec in "${RUNS[@]}"; do
  read -r arch seed epochs name <<< "$spec"
  if [[ -f "$W/$name.pt" ]]; then
    echo "[skip] $name.pt exists"
    continue
  fi
  echo "[start] arch=$arch seed=$seed epochs=$epochs -> $name  ($(date '+%F %T'))"
  python experiments/train_seeded.py --arch "$arch" --seed "$seed" --epochs "$epochs" \
      --run_name "$name" > "$LOG/train_$name.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[FAIL] $name rc=$rc  -- see $LOG/train_$name.log"; exit $rc
  fi
  echo "[done]  $name  ($(date '+%F %T'))"
done
echo "ALL TRAINING DONE ($(date '+%F %T'))"
