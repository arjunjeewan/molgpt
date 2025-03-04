#!/usr/bin/env bash
# Idempotent GuacaMol training matrix for the component ablation.
#   Part A: 5 configs (baseline,+rope,+swiglu,+rmsnorm,modern) x A_SEEDS, UNCONDITIONAL
#   Part B: {baseline,modern} x conditioning modes x B_SEEDS
# Skips any run whose checkpoint+sidecar already exist. Edit A_SEEDS / B_SEEDS to scope.
set -u
# Activate your environment first (see ../environment.yml):  conda activate molgpt
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
W=../cond_gpt/weights
LOG=experiments/logs
mkdir -p "$LOG"
EPOCHS=10

A_CONFIGS=(baseline rope swiglu rmsnorm modern)
A_SEEDS=(42 1 2)
B_CONFIGS=(baseline modern)
B_SEEDS=(42)                     # <-- set to "42 1 2" for 3-seed Part B
# conditioning modes for Part B: "label num_props props(_join) scaffold"
B_MODES=(
  "logp     1 logp no"
  "scaf     0 -    yes"
  "scaflogp 1 logp yes"
)

# Part A is seed-major so a complete single-seed Table A (all 5 configs) lands first.
RUNS=()
for s in "${A_SEEDS[@]}"; do for cfg in "${A_CONFIGS[@]}"; do
  RUNS+=("$cfg $s uncond 0 - no")
done; done
for cfg in "${B_CONFIGS[@]}"; do for s in "${B_SEEDS[@]}"; do for m in "${B_MODES[@]}"; do
  read -r ml np pr sc <<< "$m"; RUNS+=("$cfg $s $ml $np $pr $sc")
done; done; done

echo "TOTAL RUNS: ${#RUNS[@]}  ($(date '+%F %T'))"
for spec in "${RUNS[@]}"; do
  read -r cfg seed mlabel nprops props scaf <<< "$spec"
  name="guaca_${mlabel}_${cfg}_s${seed}"
  if [[ -f "$W/$name.pt" && -f "$W/$name.json" ]]; then echo "[skip] $name"; continue; fi
  scafflag=""; [[ "$scaf" == "yes" ]] && scafflag="--scaffold"
  propflag=""; [[ "$nprops" != "0" ]] && propflag="--num_props $nprops --props ${props//_/ }"
  echo "[start] $name ($(date '+%F %T'))"
  python experiments/train_ablate.py --config "$cfg" --seed "$seed" --epochs "$EPOCHS" \
      --run_name "$name" $propflag $scafflag > "$LOG/train_$name.log" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then echo "[FAIL] $name rc=$rc -- see $LOG/train_$name.log"; exit $rc; fi
  echo "[done]  $name ($(date '+%F %T'))"
done
echo "ALL TRAINING DONE ($(date '+%F %T'))"
