#!/bin/bash
# Decoder-memory grid on the cleaned data. Same data, tokenizer, and source
# labels as drfn_clean, so `gmm_<labels>_s11` from that grid is the baseline.
set -u
LABELS=${LABELS:-analyser}
for V in ${VARIANTS:-gmm_dec gmm_dec_free gmm_dec_sep}; do
  echo "=== $V  labels=$LABELS ==="
  python -u train.py --variant $V --seed ${SEED:-11} \
    --data-dir data --tokenizer data/tokenizer.json \
    --label-cache data/labels_${LABELS}.json \
    --tgt-labels data/labels_tgt_${LABELS}.json \
    --run-dir runs/${V}_${LABELS}_s${SEED:-11} || exit 1
done
