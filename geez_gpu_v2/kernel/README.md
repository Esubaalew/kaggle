# geez-byt5-gpu-v2-a

Ge'ez ByT5 v2, Run A (fine-tune only) on Kaggle GPU T4 x2 (machine_shape "NvidiaTeslaT4").
Model code a1ad2b6, dataset 04f45a7. DDP on both GPUs, fp32, 32 rows x 2 GPUs x 8 accumulation = 512 rows
per step, 10 epochs = 10,980 steps. Checkpoints go to /kaggle/working/ckpt/ every 30 minutes; the session
stops cleanly about 11h15m after start with a final checkpoint and best/.

First session: push as is (kernel_sources is empty).

Resume session (after a version finished with "SESSION DONE ... checkpoint in ckpt/"):
JSON has no comments, so edit kernel-metadata.json by hand and set
    "kernel_sources": ["esubechekol/geez-byt5-gpu-v2-a"],
then push again. The previous version's output (ckpt/, best/, eval_log.jsonl) is attached under
/kaggle/input/geez-byt5-gpu-v2-a/ and the script resumes from its newest checkpoint. Repeat until the log
shows "TRAINING DONE". The model to use is best/.
