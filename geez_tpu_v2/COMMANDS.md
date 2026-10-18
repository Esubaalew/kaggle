> Historical v2 launcher. The new bounded, tested pilot is in `CMM/research_20261017/tpu/`; see its research plan before launching another long run. The old launcher can silently restart on one core and continue after NaN.

# TPU v2 runs: exact commands (Mac)

Folder: copy this `kaggle/` folder to the Mac (e.g. `~/is/researcher/kaggle/geez_tpu_v2/`). Layout:

    dataset/                 dataset-metadata.json, model.tar.gz (model 1a8e7ad), geez-dataset.tar.gz (data 970f815), *_COMMIT
    run_a/kernel/            kernel-metadata.json + geez_byt5_tpu_run_a.py   (TPU v5e-8, private, internet on)
    run_b/kernel/            kernel-metadata.json + geez_byt5_tpu_run_b.py
    fetch_small.py           pulls logs/reports (and optionally best/)
    TRAIN_COUNTS.md, leak_audit_run_*.txt, mixture_report_run_a_and_b.json, dapt_corpus_report.json

## 0. Token

    export KAGGLE_API_TOKEN=$(cat ~/.kaggle_esubalewchekol/access_token)

## 1. Dataset (private)

    cd ~/is/researcher/kaggle/geez_tpu_v2
    kaggle datasets create -p dataset
    kaggle datasets status esubalewchekol/geez-morphology-tpu-v2      # wait for "ready"
    # later code/data updates (same slug):
    # kaggle datasets version -p dataset -m "model <hash>, data <hash>"

## 2. Kernels (both at once; Kaggle runs 2 TPU sessions in parallel)

    kaggle kernels push -p run_a/kernel
    kaggle kernels push -p run_b/kernel
    kaggle kernels status esubalewchekol/geez-byt5-tpu-v2-a
    kaggle kernels status esubalewchekol/geez-byt5-tpu-v2-b

The kernel log prints `torch / torch_xla / transformers` versions and `xla devices 8` first. If it shows
`ModuleNotFoundError: torch_xla` the session did not get the TPU image: open the kernel in the web editor once,
Settings -> Accelerator -> TPU v5e-8, Save Version (keep `machine_shape: TpuV5E8`, no `docker_image` field).

## 3. Watch and fetch

    kaggle kernels logs esubalewchekol/geez-byt5-tpu-v2-a | tail -40      # if your CLI has `logs`
    python3 fetch_small.py a
    python3 fetch_small.py b
    python3 fetch_small.py a --model      # best/ only when the run is done
    cat out_a/run_a/finetune/eval_log.jsonl  out_a/run_a/finetune/dev_report.json

## 4. Resume (if a run stops on the time budget)

Add the finished kernel's output as input (kernel_sources: ["esubalewchekol/geez-byt5-tpu-v2-a"]) and append
`"--resume_from", "/kaggle/input/geez-byt5-tpu-v2-a/run_a/finetune"` to the finetune stage in the kernel .py; push again.
