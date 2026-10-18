#!/usr/bin/env bash
# Re-run after the autocast fix: grounded gmm only (gmm_free s11/s22 already
# finished in version 1 and were fetched). GPU0 seed 11, GPU1 seed 22.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-/kaggle/working/nllb_fix}"; mkdir -p "$OUT"
WIDE="${WIDE:-1}"
run () { python3 -u nllb_mt.py "$@" --out "$OUT" --save-weights 0; }
q () { local s=$1
       run --system gmm --labels analyser --seed $s
       if [ "$WIDE" = "1" ]; then run --system gmm --labels analyser_wide --seed $s; fi; }
( export CUDA_VISIBLE_DEVICES=0; q 11 ) 2>&1 | sed -u 's/^/[gpu0] /' | tee "$OUT/log_gpu0.txt" &
( export CUDA_VISIBLE_DEVICES=1; q 22 ) 2>&1 | sed -u 's/^/[gpu1] /' | tee "$OUT/log_gpu1.txt" &
wait
