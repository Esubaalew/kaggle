#!/bin/bash
# Two GPU queues in parallel. Same data, labels (analyser), schedule, decoding.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${OUT:-/kaggle/working/nllb}
mkdir -p $OUT
q0="zeroshot:0 ft:11 gmm:11 aux:22"
q1="aux:11 gmm:22 ft:22"
run_queue () {
  local gpu=$1; shift
  for job in "$@"; do
    sys=${job%%:*}; seed=${job##*:}
    echo "=== GPU$gpu $sys seed $seed  $(date +%H:%M:%S) ==="
    CUDA_VISIBLE_DEVICES=$gpu python -u nllb_mt.py --system $sys --seed $seed --out $OUT
  done
}
N=$(python -c "import torch;print(torch.cuda.device_count())")
if [ "$N" -ge 2 ]; then
  run_queue 0 $q0 > $OUT/log_gpu0.txt 2>&1 &
  run_queue 1 $q1 > $OUT/log_gpu1.txt 2>&1 &
  wait
else
  run_queue 0 $q0 $q1 > $OUT/log_gpu0.txt 2>&1
fi
grep -h -v "^  ep " $OUT/log_gpu*.txt
