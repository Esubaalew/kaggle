#!/bin/bash
# The Ge'ez ByT5 root analyser in the 2019 BPE setup (advisor-update-2019). Run from this folder after
# build_analyser_cache.py has written runs/shared/analyser_labels.json. Everything else is the 2019 setup:
# version D trainer, configs/base.json, 60 epochs, early stop 8, same pairs, Amharic side BPE.
#
#  1-3  labels_both_s11/22/33   BPE input, root + pattern heads trained on ANALYSER labels.
#                               Compare: "Both labels" rows (same heads, geezRPA labels) and "Plain BPE" rows.
#  4    rootin_plain_s11        BPE pieces + one analyser root token per word as INPUT, no heads.
#                               Compare: "Plain BPE" seed 11. Exploratory; the analyser is then needed at test time.
set -u
L=runs/shared/analyser_labels.json
[ -f "$L" ] || { echo "missing $L (run build_analyser_cache.py first)"; exit 1; }
for S in ${SEEDS:-11 22 33}; do
  python train.py --variant bpe_aux --seed $S --data-dir data --label-cache $L --run-dir runs/labels_both_s$S
done
if [ "${INPUT_RUN:-1}" = "1" ]; then
  python train.py --variant plain_bpe --seed 11 --data-dir data \
    --src-tokenizer bpe_root --analyser-labels $L --run-dir runs/rootin_plain_s11
fi
