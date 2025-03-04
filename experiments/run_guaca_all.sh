#!/usr/bin/env bash
# End-to-end GuacaMol ablation: train matrix -> eval all -> aggregate tables.
# Idempotent throughout (each phase skips already-finished work), so safe to re-run.
set -u
# Activate your environment first (see ../environment.yml):  conda activate molgpt
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1

echo "============ GUACA MATRIX START $(date '+%F %T') ============"
bash experiments/run_train_guaca.sh
echo "============ TRAIN PHASE DONE  $(date '+%F %T') ============"
bash experiments/run_eval_guaca.sh
echo "============ EVAL PHASE DONE   $(date '+%F %T') ============"
python experiments/aggregate_guaca.py
echo "============ AGGREGATE DONE    $(date '+%F %T') ============"
