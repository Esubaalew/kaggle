#!/usr/bin/env bash
# Build the Run A2 input dataset from committed code and data plus Run A's best weights, then upload it.
# Run from this folder after committing geez-morphology-model:  bash make_dataset.sh
set -euo pipefail
R=~/is/researcher
MC=$(git -C $R/geez-morphology-model rev-parse HEAD); DC=$(git -C $R/geez-dataset-v1 rev-parse HEAD)
test -z "$(git -C $R/geez-morphology-model status --porcelain -- dataset_loader.py train_tpu.py training_mixture.py)" || { echo "commit geez-morphology-model first"; exit 1; }
rm -rf dataset && mkdir dataset
git -C $R/geez-morphology-model archive --format=tar.gz --prefix=model/ -o "$PWD/dataset/model.tar.gz" $MC
git -C $R/geez-dataset-v1 archive --format=tar.gz --prefix=geez-dataset/ -o "$PWD/dataset/geez-dataset.tar.gz" $DC
echo $MC > dataset/MODEL_COMMIT; echo $DC > dataset/DATA_COMMIT
(cd $R/kaggle/geez_gpu_v2/out_v6_full && tar -cf "$OLDPWD/dataset/runA_best.tar" -s ",^best,runA_best," best)
cat > dataset/dataset-metadata.json <<META
{"title": "geez-morphology-gpu-v2-a2", "id": "esubalewchekol/geez-morphology-gpu-v2-a2", "licenses": [{"name": "CC-BY-SA-4.0"}], "isPrivate": true}
META
echo "MODEL $MC  DATA $DC"; du -sh dataset/*
kaggle datasets create -p dataset --dir-mode tar
