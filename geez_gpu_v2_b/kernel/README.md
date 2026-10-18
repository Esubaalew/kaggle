# geez-byt5-gpu-v2-b (Run B, Kaggle GPU T4 x2)

Run B = span-corruption pretraining on the Ge'ez corpora of the dataset, then the Run A fine-tune.

Input: dataset `esubechekol/geez-morphology-gpu-v2` (model code, dataset, MODEL_COMMIT, DATA_COMMIT). Works
with the tarballs or with the folders Kaggle extracts them to.

Phases (both fp32, 512 rows per optimizer step, 32 rows x 2 GPUs x 8 accumulation; 32 x 16 on one GPU):

1. pretrain: `train_tpu.py --stage dapt` data. Corpus from `build_dapt_corpus.py` (about 16,181 lines,
   1.41 MB; lines holding a held-out form dropped), 230-byte chunks (6,204 per epoch), 15% noise, mean span 20,
   inputs 256 / targets 64 bytes, re-noised each epoch. 6 epochs = 72 steps (12 per epoch). Adafactor lr 1e-3,
   warmup min(500, steps/10) = 7, linear decay to 10%. Starts from google/byt5-small. Weights end in `pretrained/`.
2. finetune: same as Run A, from the pretrained weights with a fresh optimizer. Adafactor lr 1e-3, 500 warmup,
   linear decay to 10%, 10 epochs = 10,980 steps (1,098 per epoch), dev eval every 1000 steps, best/ on
   (dev_root + dev_exact) / 2.

Load check before training: 562,403 train, 3,472 dev, leak audit CLEAN; otherwise the run stops.
Log every 50 steps (steps/s, ETA). Checkpoints in `ckpt/` every 30 minutes and when best/ improves, keep 2;
each records its phase. Clean stop about 11h15m after the session start.

Resume: add `"kernel_sources": ["esubechekol/geez-byt5-gpu-v2-b"]` to kernel-metadata.json and run a new
version. The newest checkpoint is loaded and training continues in the recorded phase at the recorded batch.

Output: `best/` (bf16 safetensors), `pretrained/`, `ckpt/`, `eval_log.jsonl`, `dev_report.json`,
`run_info.json`, `leak_audit.txt`, `mixture_report.json`, `dapt_report.json`.
