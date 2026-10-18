#!/bin/bash
# Re-decode + slot tests on all existing checkpoints, two GPUs in parallel.
set -u
OUT=${OUT:-/kaggle/working/eval2}
mkdir -p $OUT
N=$(python -c "import torch;print(max(torch.cuda.device_count(),1))")
echo "GPUs: $N"
for ((i=0; i<N; i++)); do
  CUDA_VISIBLE_DEVICES=$i python -u eval2.py --ckpt-root /kaggle/input --out $OUT \
     --shard $i --n-shards $N > $OUT/log_shard$i.txt 2>&1 &
done
wait
cat $OUT/log_shard*.txt
