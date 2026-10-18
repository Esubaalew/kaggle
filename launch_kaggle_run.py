#!/usr/bin/env python3
"""Start a ByT5 training run on Kaggle from the current commits of both repos.

  python3 launch_kaggle_run.py          prepare everything locally and show what would change (nothing uploaded)
  python3 launch_kaggle_run.py --go     upload the bundles as a new dataset version and start the notebook
  python3 launch_kaggle_run.py --status show the run status
  python3 launch_kaggle_run.py --fetch  download the run's output to ./geezmodel/ (or ./<--run_dir>/)

Options for a run: --seed N (training seed, default 42) and --arm books|core (books: gold + generated + all book rows,
the default; core: gold + generated + dictionary + words without root, no book rows). The run name carries both:
byt5_<code7>_<data7>_<arm>_s<seed>. Checkpoints are chosen on the dev set (gold val + unified book val) for both arms.

The notebook runs as a Kaggle batch job ("Save & Run All"), so it keeps running with the browser closed.
"""
import argparse, hashlib, json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
MORPH_REPO = os.path.join(RESEARCH, "geez-morphology-model")
DATA_REPO = os.path.join(RESEARCH, "geez-dataset-v1")
KERNEL = "esubaalew/geez-morphology-byt5-training"
INPUTS = "esubaalew/geez-morphology-training-inputs"
NOTEBOOK = "geez-morphology-byt5-training.ipynb"


def sh(*cmd, cwd=None):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"[!] {' '.join(cmd)}\n{r.stdout}{r.stderr}")
    return r.stdout.strip()


def head(repo):
    if sh("git", "status", "--porcelain", cwd=repo):
        sys.exit(f"[!] {repo} has uncommitted changes; commit them first so the run is reproducible")
    return sh("git", "rev-parse", "HEAD", cwd=repo), sh("git", "branch", "--show-current", cwd=repo)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def make_bundles(morph, data, out):
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        os.remove(os.path.join(out, f))
    names = []
    for repo, (commit, branch), prefix in ((MORPH_REPO, morph, "geez-morphology"), (DATA_REPO, data, "geez-dataset")):
        name = f"{prefix}_{commit[:7]}.bundle"
        sh("git", "bundle", "create", os.path.join(out, name), branch, cwd=repo)
        sh("git", "bundle", "verify", os.path.join(out, name), cwd=repo)
        names.append(name)
    with open(os.path.join(out, "SHA256SUMS.txt"), "w") as f:
        for n in names:
            f.write(f"{sha256(os.path.join(out, n))}  {n}\n")
    json.dump({"id": INPUTS, "title": INPUTS.split("/")[1], "licenses": [{"name": "other"}]},
              open(os.path.join(out, "dataset-metadata.json"), "w"), indent=1)
    return names


BOOKS_CELL = '''# Score the best weights on every held-out set, with one prediction line per pair
import os, subprocess
out = "/kaggle/working/{run}"
best = os.path.join(out, "best_weights_by_val_score")
if os.path.isdir(best):
    for name, splits in (("gold", ["val", "test"]), ("unified", ["unified_val", "unified_test"])):
        subprocess.run(["python3", "evaluate_gold.py", "--system", "byt5", "--model_dir", best,
                        "--dataset_dir", "/tmp/geez_code/geez-dataset", "--split", *splits,
                        "--out_json", os.path.join(out, f"scores_{{name}}_details.json"),
                        "--predictions_out", os.path.join(out, f"predictions_{{name}}.jsonl")],
                       cwd="/tmp/geez_code/geez-morphology", check=True)
else:
    print("no best weights; held-out sets not scored")
'''


def patch_notebook(kdir, morph_commit, data_commit, patience=None, seed=42, arm="books"):
    sh("kaggle", "kernels", "pull", KERNEL, "-p", kdir, "-m")
    meta_path, nb_path = os.path.join(kdir, "kernel-metadata.json"), os.path.join(kdir, NOTEBOOK)
    meta, nb = json.load(open(meta_path)), json.load(open(nb_path))

    # Any checkpoint under /kaggle/input is resumed from, so only the bundles dataset may be attached.
    meta.update(dataset_sources=[INPUTS], kernel_sources=[], model_sources=[], competition_sources=[],
                enable_gpu=True, enable_internet=True)
    json.dump(meta, open(meta_path, "w"), indent=2)

    cfg = nb["cells"][1]
    src = "".join(cfg["source"])
    for key, val in (("MORPHOLOGY_COMMIT", morph_commit), ("DATASET_COMMIT", data_commit)):
        src, n = re.subn(rf'^({key}\s*=\s*)"[0-9a-f]*"', rf'\g<1>"{val}"', src, flags=re.M)
        if n != 1:
            sys.exit(f"[!] {key} not found once in the settings cell; the notebook changed, edit it by hand")
    if patience is not None:
        src, n = re.subn(r'^(PATIENCE\s*=\s*)\d+', rf'\g<1>{patience}', src, flags=re.M)
        if n != 1:
            sys.exit("[!] PATIENCE not found once in the settings cell")
    run = f"byt5_{morph_commit[:7]}_{data_commit[:7]}_{arm}_s{seed}"
    extra = f"--seed {seed}" + (" --no_unified --no_book_forms" if arm == "core" else "")
    for key, val in (("RUN_NAME", run), ("TRAINER_EXTRA_ARGS", extra)):
        src, n = re.subn(rf'^({key}\s*=\s*)"[^"]*"', lambda m, v=val: m.group(1) + '"' + v + '"', src, count=1, flags=re.M)
        if n != 1:
            sys.exit(f"[!] {key} not found once in the settings cell; the notebook changed, edit it by hand")
    checks = {"RUN_MODE": '"final"'}
    for key, want in checks.items():
        m = re.search(rf'^{key}\s*=\s*(\S+)', src, flags=re.M)
        if not m or m.group(1) != want:
            sys.exit(f"[!] settings cell: {key} is {m.group(1) if m else 'missing'}, expected {want}")
    if '"resume_from": ""' not in src:
        sys.exit('[!] settings cell: resume_from is not empty; this run must start from scratch')
    cfg["source"] = src.splitlines(keepends=True)

    # checkpoint choice on the dev set: gold val plus the unified book val rows (evaluate_gold.py --dev_unified)
    launcher = "".join(nb["cells"][3]["source"])
    old = '"--dataset_dir", self.data, "--split", "val", "--out_json", out_json]'
    new = old.replace('"val", ', '"val", "--dev_unified", ')
    if launcher.count(old) + launcher.count(new) != 1:
        sys.exit("[!] the launcher cell's val scoring command changed; edit the notebook by hand")
    nb["cells"][3]["source"] = launcher.replace(old, new).splitlines(keepends=True)
    books = BOOKS_CELL.format(run=run)
    nb["cells"] = [c for c in nb["cells"] if "held-out sets not scored" not in "".join(c["source"])
                   and "--split\", \"books\"" not in "".join(c["source"])]
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                        "source": books.splitlines(keepends=True)})
    json.dump(nb, open(nb_path, "w"), indent=1, ensure_ascii=False)
    return run, meta


def wait_dataset_ready():
    for _ in range(60):
        st = subprocess.run(["kaggle", "datasets", "status", INPUTS], capture_output=True, text=True).stdout.strip()
        print(f"  dataset status: {st}")
        if "ready" in st.lower():
            return
        time.sleep(20)
    sys.exit("[!] the dataset version did not become ready in 20 minutes; check it on kaggle.com, then rerun --go")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="upload and start the run")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--seed", type=int, default=42, help="training seed of this run (default 42)")
    ap.add_argument("--arm", choices=["books", "core"], default="books",
                    help="books: gold + generated + all book rows (default); core: no book rows (the ablation arm)")
    ap.add_argument("--run_dir", default="geezmodel", help="--fetch: folder under this one for the output")
    ap.add_argument("--patience", type=int, default=None,
                    help="val passes without a gain before early stopping (notebook default 4)")
    a = ap.parse_args()

    if a.status:
        print(sh("kaggle", "kernels", "status", KERNEL))
        return
    if a.fetch:
        out = os.path.join(HERE, a.run_dir)
        print(sh("kaggle", "kernels", "output", KERNEL, "-p", out, "-o"))
        return

    morph, data = head(MORPH_REPO), head(DATA_REPO)
    print(f"geez-morphology {morph[1]} {morph[0]}\ngeez-dataset    {data[1]} {data[0]}")
    stage = os.path.join(HERE, "staging")
    bdir, kdir = os.path.join(stage, "inputs"), os.path.join(stage, "kernel")
    names = make_bundles(morph, data, bdir)
    print(f"bundles: {', '.join(names)} in {bdir}")
    os.makedirs(kdir, exist_ok=True)
    run, meta = patch_notebook(kdir, morph[0], data[0], a.patience, a.seed, a.arm)
    print(f"notebook: commits set, books scoring cell added, inputs {meta['dataset_sources']}, "
          f"machine {meta.get('machine_shape')}, internet {meta['enable_internet']}")
    print(f"arm {a.arm}, seed {a.seed}")
    print(f"output folder on Kaggle: /kaggle/working/{run}")

    if not a.go:
        print("\nNothing uploaded. Run again with --go to upload the bundles and start training.")
        return
    print("\nuploading a new version of", INPUTS)
    print(sh("kaggle", "datasets", "version", "-p", bdir, "-m", f"bundles {morph[0][:7]} {data[0][:7]}"))
    wait_dataset_ready()
    print("starting the notebook")
    print(sh("kaggle", "kernels", "push", "-p", kdir))
    print(f"\nStarted. It runs on Kaggle with your browser closed (about 4 hours).\n"
          f"  status: python3 {os.path.basename(__file__)} --status\n"
          f"  output: python3 {os.path.basename(__file__)} --fetch")


if __name__ == "__main__":
    main()
