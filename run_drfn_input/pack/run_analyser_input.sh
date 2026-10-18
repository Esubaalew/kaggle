#!/bin/bash
# The Ge'ez ByT5 root analyser as the Ge'ez-side tokenizer (the Amharic side stays BPE). Run from this folder after
# build_analyser_cache.py has written runs/shared/analyser_cache.json and analyser_labels.json.
#   input_plain_s<seed>   plain model; Ge'ez input = [root] + [morphology] for each word whose root occurs >= 2 times in
#                         train, BPE pieces for every other word. Compare: Plain BPE (advisor-update-2019).
#   input_labels_s11      same input plus root + pattern heads on the analyser labels (LABELS_RUN=1, seed 11 only).
#                         The root and the pattern are already in the input, so these heads are close to trivial;
#                         this run only checks that they do not hurt.
# Transformer body fixed to the 2019 widths (FFN 1186 plain, 1169 with heads); only the Ge'ez embedding table is larger.
# Everything else is the 2019 setup: version D trainer, configs/base.json, 60 epochs, early stop 8, same pairs.
set -u
C=runs/shared/analyser_cache.json
L=runs/shared/analyser_labels.json
for f in "$C" "$L"; do [ -f "$f" ] || { echo "missing $f (run build_analyser_cache.py first)"; exit 1; }; done
for S in ${SEEDS:-11 22 33}; do
  python train.py --variant plain_bpe --seed $S --data-dir data \
    --src-tokenizer analyser_input --analyser-cache $C --ffn-dim 1186 --run-dir runs/input_plain_s$S
done
if [ "${LABELS_RUN:-1}" = "1" ]; then
  python train.py --variant bpe_aux --seed 11 --data-dir data --label-cache $L \
    --src-tokenizer analyser_input --analyser-cache $C --ffn-dim 1169 --run-dir runs/input_labels_s11
fi
