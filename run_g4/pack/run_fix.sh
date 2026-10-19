#!/usr/bin/env bash
# Fixed Grounded Morphological Memory (drfn_fix) runner.
# Environment variables:
#   SEEDS      (default "11 22")
#   LABELS     ("analyser" for 256-slot clean cache, "analyser_wide" for 1024-slot wide cache)
#   VARIANTS   (default "bpe_aux cmm_spec gmm")
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

SEEDS="${SEEDS:-11 22}"
LABELS="${LABELS:-analyser}"
VARIANTS="${VARIANTS:-bpe_aux cmm_spec gmm}"
EPOCHS="${EPOCHS:-60}"

CACHE="data/labels_${LABELS}.json"
TGT_MAP="data/labels_tgt_${LABELS}.json"
TOK="data/tokenizer.json"

# REVIEW FIX 2026-10-14: one seed per GPU in parallel (Kaggle T4 x2), output
# streamed with a [gpuN] prefix so progress is visible in the notebook log.
NGPU=$(python3 -c "import torch;print(max(torch.cuda.device_count(),1))")
run_seed () {
  local s=$1
  for v in $VARIANTS; do
    tag="${v}_${LABELS}_s${s}"
    if [ "$v" = "plain_bpe" ]; then
      tag="plain_bpe_s${s}"
    fi
    echo "================================================================"
    echo "run=$tag  variant=$v  labels=$LABELS  seed=$s"
    echo "================================================================"
    python3 -u train.py \
      --variant "$v" \
      --seed "$s" \
      --data-dir data \
      --tokenizer "$TOK" \
      --label-cache "$CACHE" \
      --tgt-label-map "$TGT_MAP" \
      --run-dir "runs/${tag}" \
      --epochs "$EPOCHS"
    rm -f "runs/${tag}/last.pt"
  done
}
i=0
for s in $SEEDS; do
  g=$(( i % NGPU ))
  ( export CUDA_VISIBLE_DEVICES=$g; run_seed $s ) 2>&1 | sed -u "s/^/[gpu$g s$s] /" &
  i=$(( i + 1 ))
  if [ "$i" -ge "$NGPU" ]; then wait; i=0; fi
done
wait
