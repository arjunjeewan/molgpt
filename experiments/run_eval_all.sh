#!/usr/bin/env bash
# Evaluate every (arch,seed) checkpoint:
#   1) pure-temperature decoding frontier (fast metrics) -> datasets/sweep_seed_<tag>.csv
#   2) full MOSES metrics at T=1.0 (FCD/SNN/Frag/Scaf/...) -> datasets/moses_metrics_<tag>_T1.0.json
# Idempotent: skips outputs that already exist and ckpts that don't.
set -u
# Activate your environment first (see ../environment.yml):  conda activate molgpt
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
W=../cond_gpt/weights
LOG=experiments/logs
TEMPS="0.7 0.9 1.0 1.2 1.6"
FULLT=1.0

# tag  arch  ckpt_basename
SPECS=(
  "base_s42   baseline unconditional_moses_baseline"
  "modern_s42 modern   unconditional_moses"
  "base_s1    baseline unconditional_moses_baseline_s1"
  "modern_s1  modern   unconditional_moses_modern_s1"
  "base_s2    baseline unconditional_moses_baseline_s2"
  "modern_s2  modern   unconditional_moses_modern_s2"
  "modern_e25 modern   unconditional_moses_modern_e25"
)

for spec in "${SPECS[@]}"; do
  read -r tag arch name <<< "$spec"
  ckpt="$W/$name.pt"
  if [[ ! -f "$ckpt" ]]; then echo "[skip-missing] $tag ($ckpt)"; continue; fi
  bflag=""; [[ "$arch" == "baseline" ]] && bflag="--baseline"

  sweepout="datasets/sweep_seed_$tag.csv"
  if [[ -f "$sweepout" ]]; then echo "[skip] sweep $tag"; else
    echo "[sweep] $tag ($(date '+%T'))"
    python experiments/sweep_decode.py $bflag --ckpt "$ckpt" --temps "$TEMPS" --out "$sweepout" \
        > "$LOG/sweep_$tag.log" 2>&1 || { echo "[FAIL sweep] $tag"; }
  fi

  mjson="datasets/moses_metrics_${tag}_T${FULLT}.json"
  if [[ -f "$mjson" ]]; then echo "[skip] moses $tag"; else
    echo "[moses] $tag T=$FULLT ($(date '+%T'))"
    python experiments/gen_eval_moses.py $bflag --ckpt "$ckpt" --temp "$FULLT" --tag "$tag" \
        > "$LOG/moses_$tag.log" 2>&1 || { echo "[FAIL moses] $tag"; }
  fi
done
echo "ALL EVAL DONE ($(date '+%F %T'))"
