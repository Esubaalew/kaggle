#!/bin/bash
# Deciding grid, seed 11. All four runs share ONE BPE tokenizer so the only
# difference between them is where the supervision goes and whether the labels
# are real.
#
#   bpe_aux      real labels -> external heads (the current best system)
#   bpe_placebo  shuffled labels -> external heads
#   gmm          real labels -> memory routing
#   gmm_placebo  shuffled labels -> memory routing
#
# Read (gmm - gmm_placebo) against (bpe_aux - bpe_placebo). The head-side gap
# is the known +0.47 chrF++. The placebo permutes labels ACROSS WORDS, so two
# forms of one root land in different slots and the grounded memory cannot
# share between them -- which is what makes this the decisive comparison.
set -u
mkdir -p runs/shared
for V in ${VARIANTS:-bpe_aux bpe_placebo gmm gmm_placebo}; do
  echo "=== $V ==="
  python -u train.py --variant $V --seed ${SEED:-11} \
    --data-dir data --tokenizer runs/shared/tokenizer.json --label-cache data/geezrpa_label_cache_k256.json \
    --run-dir runs/${V}_s${SEED:-11} || exit 1
done
