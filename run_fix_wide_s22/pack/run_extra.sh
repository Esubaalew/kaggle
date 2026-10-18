#!/usr/bin/env bash
# Finish the wide (1024-slot) arm on one Kaggle T4x2 session.
#   GPU0: bpe_aux seed 22 (the run Colab could not finish)
#   GPU1: gmm seed 33 then bpe_aux seed 33 (third seed), unless SEED33=0
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
LABELS=analyser_wide; EPOCHS="${EPOCHS:-60}"; SEED33="${SEED33:-1}"
one () { local v=$1 s=$2 tag="${1}_analyser_wide_s${2}"
  echo "================================================================"
  echo "run=$tag  variant=$v  labels=$LABELS  seed=$s"
  echo "================================================================"
  python3 -u train.py --variant "$v" --seed "$s" --data-dir data \
    --tokenizer data/tokenizer.json --label-cache "data/labels_${LABELS}.json" \
    --tgt-label-map "data/labels_tgt_${LABELS}.json" --run-dir "runs/${tag}" --epochs "$EPOCHS"
  rm -f "runs/${tag}/last.pt"; }
( export CUDA_VISIBLE_DEVICES=0; one bpe_aux 22 ) 2>&1 | sed -u 's/^/[gpu0] /' &
if [ "$SEED33" = "1" ]; then
  ( export CUDA_VISIBLE_DEVICES=1; one gmm 33; one bpe_aux 33 ) 2>&1 | sed -u 's/^/[gpu1] /' &
fi
wait
