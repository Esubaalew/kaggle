#!/bin/bash
# Clean-data grid. LABELS selects the label cache (geezrpa | analyser).
# All runs share data/tokenizer.json, built once from the cleaned training
# text, so both notebooks use the identical tokenizer.
#
# After the last epoch each run decodes the full dev set from best.pt and
# writes dev_hyps.json + final_eval.json (char chrF, routing probes), so
# fetch_small.py is enough -- no checkpoint downloads.
set -u
LABELS=${LABELS:-geezrpa}
for V in ${VARIANTS}; do
  ES=""
  case $V in cmm_spec|cmm) ES="--entropy-sign 1.0";; esac
  echo "=== $V  labels=$LABELS ==="
  python -u train.py --variant $V --seed ${SEED:-11} $ES \
    --data-dir data --tokenizer data/tokenizer.json \
    --label-cache data/labels_${LABELS}.json \
    --run-dir runs/${V}_${LABELS}_s${SEED:-11} || exit 1
done
