#!/bin/bash
# Controls for step 3. GPU0: gmm_free s11 then aux (weight 0.1) s11; GPU1: gmm_free s22.
# Progress goes to the notebook log as well as to files.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/kaggle/working/nllb}
mkdir -p $OUT
q0 () { CUDA_VISIBLE_DEVICES=0 python -u nllb_mt.py --system gmm_free --seed 11 --out $OUT && \
        CUDA_VISIBLE_DEVICES=0 python -u nllb_mt.py --system aux --aux-weight 0.1 --seed 11 --out $OUT; }
q1 () { CUDA_VISIBLE_DEVICES=1 python -u nllb_mt.py --system gmm_free --seed 22 --out $OUT; }
q0 2>&1 | sed -u 's/^/[gpu0] /' | tee $OUT/log2_gpu0.txt &
q1 2>&1 | sed -u 's/^/[gpu1] /' | tee $OUT/log2_gpu1.txt &
wait
