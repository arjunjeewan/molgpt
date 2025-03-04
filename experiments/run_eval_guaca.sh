#!/usr/bin/env bash
# Evaluate every trained GuacaMol checkpoint (enumerated from its sidecar .json).
#   unconditional -> validity/unique/novelty + KL-div + FCD
#   conditional/scaffold -> per-condition validity/unique/novelty + MAD + scaffold-match
# Idempotent: skips runs whose metrics json already exists.
set -u
# Activate your environment first (see ../environment.yml):  conda activate molgpt
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
W=../cond_gpt/weights
LOG=experiments/logs
TEMP=1.0

shopt -s nullglob
for sidecar in "$W"/guaca_*.json; do
  name=$(basename "$sidecar" .json)
  mjson="datasets/guaca_metrics_${name}_T${TEMP}.json"
  if [[ -f "$mjson" ]]; then echo "[skip] $name"; continue; fi
  echo "[eval] $name T=$TEMP ($(date '+%T'))"
  python experiments/eval_guaca.py --run_name "$name" --temp "$TEMP" > "$LOG/eval_$name.log" 2>&1 \
      || echo "[FAIL eval] $name -- see $LOG/eval_$name.log"
done
echo "ALL EVAL DONE ($(date '+%F %T'))"
