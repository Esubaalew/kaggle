#!/bin/bash
# One run. train.py computed  total = total - entropy_reg * ent, which
# MAXIMIZES routing entropy and pushes every token toward uniform assignment
# over slots -- the failure mode section 3.6 names as (i), diffuse routing.
# The protocol says the opposite: Eq.(12) adds +alpha*L_ent and section 3.6
# says "Minimizing L_ent sharpens each token's routing toward few slots".
#
# This reruns the published cmm configuration with the sign flipped and
# nothing else changed. Compare metadata.json routing_consistency_root
# against the published 0.337 (permutation baseline 0.330).
# Run 3 asks a different question: the published `cmm` module is NOT the CMM
# of protocol section 3.4. cmm.py ties keys to values and drops the query
# projection (softmax(h B^T) B), where Eqs.(4-5) specify Q = H W_Q with
# separate M_K and M_V, and Eq.(9) a scalar gate. `cmm_spec` is the spec
# version, with the SAME auxiliary-head supervision as `cmm`, so the only
# difference is module fidelity. Until this run exists, the thesis cannot say
# whether the negative result is about the idea or about the implementation.
set -u
mkdir -p runs/shared
python -u train.py --variant cmm --seed 11 --entropy-sign -1.0 \
  --data-dir data --tokenizer runs/shared/tokenizer.json --label-cache data/geezrpa_label_cache_k256.json \
  --run-dir runs/cmm_s11_diffuse
python -u train.py --variant cmm --seed 11 --entropy-sign 1.0 \
  --data-dir data --tokenizer runs/shared/tokenizer.json --label-cache data/geezrpa_label_cache_k256.json \
  --run-dir runs/cmm_s11_sharpen
python -u train.py --variant cmm_spec --seed 11 --entropy-sign 1.0 \
  --data-dir data --tokenizer runs/shared/tokenizer.json --label-cache data/geezrpa_label_cache_k256.json \
  --run-dir runs/cmmspec_s11_sharpen
