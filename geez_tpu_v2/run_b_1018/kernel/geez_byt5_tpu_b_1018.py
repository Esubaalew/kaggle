# Kaggle TPU kernel: Ge'ez ByT5 Run B (dapt -> finetune), SMOKE TEST (300 optimizer steps total).
# dapt 140 steps -> finetune 140 steps (init from dapt/best, eval) -> resume finetune +20 steps (to 160).
import glob, os, re, subprocess, sys, time

RUN = "run_b_1018_smoke"
OUT = "/kaggle/working/" + RUN
T0 = time.time()
COMMON = ["--model_name", "google/byt5-small", "--per_core_batch", "16", "--gradient_accumulation", "4",
          "--bf16", "auto", "--lr", "1e-3", "--expected_world_size", "8", "--log_every", "25"]
STAGES = [
    ("dapt", "dapt", ["--stage", "dapt", "--max_steps", "140", "--warmup_steps", "20", "--save_every", "70",
                      "--time_budget_hours", "1.0"]),
    ("finetune", "finetune", ["--max_steps", "140", "--warmup_steps", "20", "--eval_every", "140", "--save_every", "140",
                              "--dev_cap", "500", "--skip_final_report", "--unified_upsample", "3",
                              "--rare_class_min", "400", "--init_from", OUT + "/dapt/best", "--time_budget_hours", "1.0"]),
    ("finetune_resume", "finetune", ["--max_steps", "160", "--warmup_steps", "20", "--eval_every", "160", "--save_every", "160",
                                     "--dev_cap", "500", "--skip_final_report", "--unified_upsample", "3",
                                     "--rare_class_min", "400", "--init_from", OUT + "/dapt/best", "--time_budget_hours", "0.5"]),
]

if not glob.glob("/kaggle/input/**/train_tpu.py", recursive=True):
    os.makedirs("/kaggle/tmp/src", exist_ok=True)
    for tgz in glob.glob("/kaggle/input/**/*.tar.gz", recursive=True):
        subprocess.run(["tar", "-xzf", tgz, "-C", "/kaggle/tmp/src"], check=True)

def find(name):
    for root in ("/kaggle/input", "/kaggle/tmp/src"):
        hits = sorted(glob.glob(f"{root}/**/{name}", recursive=True))
        if hits:
            return os.path.dirname(hits[0])
    sys.exit(f"{name} not found")

code = find("train_tpu.py")
data = os.path.dirname(os.path.dirname(find("merho_ges_gold_train.jsonl")))
print("code", code, "data", data, flush=True)
for f in ("MODEL_COMMIT", "DATA_COMMIT", "MANIFEST.txt"):
    p = next(iter(glob.glob(f"/kaggle/input/**/{f}", recursive=True)), "")
    if os.path.exists(p):
        print(f, open(p).read().strip(), flush=True)
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.setdefault("PJRT_DEVICE", "TPU")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
subprocess.run("python -c \"import torch, torch_xla, transformers; print('torch', torch.__version__, 'torch_xla', "
               "torch_xla.__version__, 'transformers', transformers.__version__)\"", shell=True)
subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, sys.argv[1]); import train_tpu; "
                "print('CHIP_COUNT detected', train_tpu.detect_chip_count(), flush=True)", code])
os.makedirs(OUT, exist_ok=True)

logs = {}
for name, outdir, extra in STAGES:
    stage_out = os.path.join(OUT, outdir)
    cmd = [sys.executable, os.path.join(code, "train_tpu.py"), "--dataset_dir", data, "--output_dir", stage_out] + COMMON + extra
    print("RUN", name, " ".join(cmd), flush=True)
    buf = []
    with open(os.path.join(OUT, f"{name}.log"), "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); log.write(line); log.flush(); buf.append(line)
        rc = p.wait()
    logs[name] = "".join(buf)
    print(f"STAGE {name} exit {rc} after {(time.time() - T0) / 3600:.2f} h", flush=True)
    if rc != 0:
        print(f"SMOKE FAILED in {name}", flush=True)
        sys.exit(rc)

checks = {
    "world8": " x8," in logs["dapt"] and " x8," in logs["finetune"],
    "dapt_steps": "step 140/140" in logs["dapt"],
    "finite_loss": all(re.search(r"loss (nan|inf)", l, re.I) is None for l in logs.values()),
    "eval_ran": "eval {" in logs["finetune"],
    "best_saved": os.path.isdir(OUT + "/finetune/best") and os.path.isdir(OUT + "/dapt/best"),
    "resumed_at_140": "at step 140" in logs["finetune_resume"],
    "reached_160": "step 160/160" in logs["finetune_resume"],
}
for k, v in checks.items():
    print("CHECK", k, "OK" if v else "FAIL", flush=True)
print(f"FINISHED smoke {(time.time() - T0) / 3600:.2f} h", flush=True)
print("SMOKE OK" if all(checks.values()) else "SMOKE FAILED checks", flush=True)
