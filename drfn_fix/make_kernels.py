"""Generate the fixed small-model Kaggle notebooks for a given Kaggle username.

    python3 make_kernels.py KAGGLE_USERNAME

Writes:
  - pack/dataset-metadata.json (`<username>/drfn-fix-pack`)
  - kernel_clean/ (`<username>/drfn-fix-clean`: bpe_aux, cmm_spec, gmm on 256-slot cleaned labels, seeds 11 & 22)
  - kernel_wide/  (`<username>/drfn-fix-wide`:  bpe_aux, gmm on 1024-slot wide cleaned labels, seeds 11 & 22)
"""
import json
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))

json.dump({"title": "drfn-fix-pack", "id": f"{OWNER}/drfn-fix-pack",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(HERE, "pack", "dataset-metadata.json"), "w"), indent=2)

SETUP = r'''import glob, os, shutil, subprocess, sys, json
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
pack_src = os.path.dirname(find('run_fix.sh'))
PACK = '/kaggle/working/pack'
if not os.path.exists(PACK): shutil.copytree(pack_src, PACK)
os.chdir(PACK)
print('pack', pack_src)
def stream(cmd, env=None):
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    for line in p.stdout:
        if line.startswith('{"step"') or line.startswith("{'step'"): continue
        print(line, end='')
    return p.wait()'''

SUMMARY = r'''rows = []
for d in sorted(glob.glob('runs/*_s*')):
    lg, fe = os.path.join(d, 'train_log.jsonl'), os.path.join(d, 'final_eval.json')
    if not os.path.exists(lg): continue
    ev = [json.loads(l) for l in open(lg) if '"dev_loss"' in l]
    if not ev: continue
    b = min(ev, key=lambda e: e['dev_loss'])
    f = json.load(open(fe)) if os.path.exists(fe) else {}
    b5 = f.get('beam5_nr4') or {}
    pr = (f.get('probe_root') or {}).get('all_types') or {}
    gp = (f.get('probe_root') or {}).get('gate_probe') or {}
    rows.append((os.path.basename(d), len(ev), b['dev_loss'], b['epoch'],
                 b5.get('chrF_char'), b5.get('BLEU'),
                 (f.get('probe_root') or {}).get('unseen_slot_accuracy'),
                 gp.get('mean_gate_rooted'), gp.get('mean_gate_rootless')))
fmt = lambda v, s='%.2f': 'n/a' if v is None else s % v
print('%-28s %4s %9s %4s %10s %8s %11s %15s' % (
    'run', 'ep', 'best loss', '@ep', 'chrF(b5)', 'BLEU(b5)', 'root unseen', 'gate(rt/null)'))
for r in rows:
    print('%-28s %4d %9.4f %4d %10s %8s %11s %15s' % (
        r[0], r[1], r[2], r[3], fmt(r[4]), fmt(r[5]),
        ('%.1f%%' % (100 * r[6])) if r[6] is not None else 'n/a',
        (fmt(r[7], '%.2f') + '/' + fmt(r[8], '%.2f')) if r[7] is not None else 'n/a'))
open('/kaggle/working/summary.txt', 'w').write(repr(rows))'''


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


ARMS = {
    "clean": ("analyser", "11 22", "bpe_aux cmm_spec gmm",
              "256-slot geez-dataset-verified labels (seeds 11 & 22)"),
    "wide": ("analyser_wide", "11 22", "bpe_aux gmm",
             "1024-slot wide geez-dataset-verified labels (seeds 11 & 22)"),
}

for arm, (labels, seeds, variants, desc) in ARMS.items():
    kdir = os.path.join(HERE, f"kernel_{arm}")
    os.makedirs(kdir, exist_ok=True)
    slug = f"drfn-fix-{arm}"
    intro = f"# {slug}: {desc}\n\nIncludes all 4 small-model fixes: geez-dataset-verified labels, Null-Safe Gate Grounding (`BCE(lambda_t, root >= 1)`), parameter-free Value Grounding (`c @ M_V.T`), and Layer-4 readout retention (`has_aux = True`). Decodes full dev with Beam 5 + 4-gram blocking."
    json.dump(nb([
        md(intro),
        code(f'SEEDS = "{seeds}"\nLABELS = "{labels}"\nVARIANTS = "{variants}"'),
        code("!pip install -q tokenizers sacrebleu\n"
             "import torch; print(torch.__version__, torch.cuda.is_available())"),
        code(SETUP),
        code("env = dict(os.environ, SEEDS=SEEDS, LABELS=LABELS, VARIANTS=VARIANTS)\n"
             "print('exit', stream(['bash', 'run_fix.sh'], env=env))"),
        code(SUMMARY),
    ]), open(os.path.join(kdir, f"{slug}.ipynb"), "w"), indent=1)
    json.dump({"id": f"{OWNER}/{slug}", "title": slug,
               "code_file": f"{slug}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
               "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
               "dataset_sources": [f"{OWNER}/drfn-fix-pack"], "kernel_sources": [],
               "competition_sources": [], "model_sources": [],
               "machine_shape": "NvidiaTeslaT4"},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: pack/dataset-metadata.json, kernel_clean/, kernel_wide/ written")
