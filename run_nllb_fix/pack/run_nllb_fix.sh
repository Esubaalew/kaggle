#!/usr/bin/env bash
# Fixed NLLB-600M grounded memory (drfn_nllb_fix). Two GPU queues in parallel,
# progress streamed to the notebook log with a [gpuN] prefix.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT="${OUT:-/kaggle/working/nllb_fix}"; mkdir -p "$OUT"
run () { python3 -u nllb_mt.py "$@" --out "$OUT" --save-weights 0; }
q0 () { run --system gmm --labels analyser --seed 11
        run --system gmm_free --labels analyser --seed 11
        run --system gmm --labels analyser_wide --seed 11; }
q1 () { run --system gmm --labels analyser --seed 22
        run --system gmm_free --labels analyser --seed 22
        run --system gmm --labels analyser_wide --seed 22; }
( export CUDA_VISIBLE_DEVICES=0; q0 ) 2>&1 | sed -u 's/^/[gpu0] /' | tee "$OUT/log_gpu0.txt" &
( export CUDA_VISIBLE_DEVICES=1; q1 ) 2>&1 | sed -u 's/^/[gpu1] /' | tee "$OUT/log_gpu1.txt" &
wait
