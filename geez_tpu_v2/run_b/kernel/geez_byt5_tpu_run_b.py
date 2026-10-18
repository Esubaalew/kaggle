# Kaggle TPU kernel: Ge'ez morphology ByT5, run_b. Generated from run_kernel_template.py.
import glob
import os
import subprocess
import sys
import time

RUN = "run_b"
STAGES = [["dapt", ["--stage", "dapt", "--model_name", "google/byt5-small", "--per_core_batch", "32", "--lr", "1e-3", "--epochs", "60", "--warmup_steps", "100", "--save_every", "200", "--time_budget_hours", "1.2"]], ["finetune", ["--model_name", "google/byt5-small", "--per_core_batch", "64", "--lr", "1e-3", "--epochs", "10", "--warmup_steps", "500", "--eval_every", "1000", "--save_every", "1000", "--unified_upsample", "3", "--rare_class_min", "400", "--init_from", "{OUT}/dapt/best"]]]
OUT = "/kaggle/working/" + RUN
T0 = time.time()
SESSION_HOURS = 8.5      # stop training with margin before the 9 h session limit


# The dataset holds model.tar.gz and geez-dataset.tar.gz (Kaggle may have unpacked them already).
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
for f in ("MODEL_COMMIT", "DATA_COMMIT"):
    p = next(iter(glob.glob(f"/kaggle/input/**/{f}", recursive=True)), "")
    if os.path.exists(p):
        print(f, open(p).read().strip(), flush=True)
# Kaggle presets TPU_PROCESS_ADDRESSES for one process, which breaks the 8-process launch
# ("Expected 8 worker addresses, got 1"); drop it before anything touches the TPU.
os.environ.pop("TPU_PROCESS_ADDRESSES", None)
os.environ.setdefault("PJRT_DEVICE", "TPU")
subprocess.run("python -c \"import torch, torch_xla, transformers; print('torch', torch.__version__, 'torch_xla', "
               "torch_xla.__version__, 'transformers', transformers.__version__)\"", shell=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs(OUT, exist_ok=True)

for name, extra in STAGES:
    left = SESSION_HOURS - (time.time() - T0) / 3600
    stage_out = os.path.join(OUT, name)
    budget_arg = [] if "--time_budget_hours" in extra else ["--time_budget_hours", f"{max(0.2, left - 0.35):.2f}"]
    cmd = [sys.executable, os.path.join(code, "train_tpu.py"), "--dataset_dir", data, "--output_dir", stage_out] + budget_arg + [x.replace("{OUT}", OUT) for x in extra]
    print("RUN", " ".join(cmd), flush=True)
    with open(os.path.join(OUT, f"{name}.log"), "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line)
            log.write(line)
            log.flush()
        rc = p.wait()
    print(f"stage {name} exit {rc} after {(time.time() - T0) / 3600:.2f} h", flush=True)
    if rc != 0:
        sys.exit(rc)
print("done", flush=True)
