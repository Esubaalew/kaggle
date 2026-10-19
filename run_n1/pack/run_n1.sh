#!/usr/bin/env bash
# N1: NLLB-200-distilled-600M on the repaired AGE pairs with the feature-bundle labels (1,024 root slots,
# 128 feature-bundle slots). Arms: ft (no labels), aux (labels as side heads), gmm with the gate trained open
# for every word with a root or bundle label, and the same gmm with one epoch in which NLLB is frozen and only
# the memory trains. Seeds 11/22/33; GPU0 runs seed 11 and half of seed 33, GPU1 seed 22 and the other half.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-/kaggle/working/nllb_n1}"; mkdir -p "$OUT"
L=analyser_wide_feat
run () { python3 -u nllb_mt.py "$@" --labels $L --out "$OUT" --save-weights 0 || echo "FAILED: $*"; }
arm () { case $1 in
  ft) run --system ft --seed $2 ;;
  aux) run --system aux --seed $2 ;;
  gmm) run --system gmm --gate-target any --seed $2 ;;
  gmm_frz) run --system gmm --gate-target any --freeze-epochs 1 --seed $2 ;;
esac; }
q0 () { for a in gmm gmm_frz aux ft; do arm $a 11; done; arm gmm 33; arm aux 33; }
q1 () { for a in gmm gmm_frz aux ft; do arm $a 22; done; arm gmm_frz 33; arm ft 33; }
( export CUDA_VISIBLE_DEVICES=0; q0 ) 2>&1 | sed -u 's/^/[gpu0] /' | tee "$OUT/log_gpu0.txt" &
( export CUDA_VISIBLE_DEVICES=1; q1 ) 2>&1 | sed -u 's/^/[gpu1] /' | tee "$OUT/log_gpu1.txt" &
wait
echo "N1 QUEUE FINISHED"
