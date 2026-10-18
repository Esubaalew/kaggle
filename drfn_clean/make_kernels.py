"""Generate the two clean-grid Kaggle notebooks for a given Kaggle username.

    python3 make_kernels.py KAGGLE_USERNAME

Writes pack/dataset-metadata.json, kernel_geezrpa/ and kernel_analyser/ so
that the dataset and both notebooks belong to KAGGLE_USERNAME. Rerun with a
different username to switch accounts; nothing else changes.
"""
import json
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))

json.dump({"title": "drfn-clean-pack", "id": f"{OWNER}/drfn-clean-pack",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(HERE, "pack", "dataset-metadata.json"), "w"), indent=2)

SETUP = r'''import glob, os, shutil, subprocess, sys, json
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
pack_src = os.path.dirname(find('run_clean.sh'))
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
    pr = (f.get('probe_root') or {}).get('all_types') or {}
    pp = (f.get('probe_pattern') or {}).get('all_types') or {}
    rows.append((os.path.basename(d), len(ev), b['dev_loss'], b['epoch'],
                 f.get('chrF_char_full_dev'), pr.get('consistency'), pr.get('permutation'),
                 (f.get('probe_root') or {}).get('unseen_slot_accuracy'),
                 pp.get('consistency'), pp.get('permutation')))
fmt = lambda v, s='%.3f': 'n/a' if v is None else s % v
print('%-26s %4s %9s %4s %10s %15s %10s %15s' % ('run', 'ep', 'best loss', '@ep',
      'chrF full', 'root vs chance', 'root unseen', 'pattern vs chance'))
for r in rows:
    print('%-26s %4d %9.4f %4d %10s %15s %10s %15s' % (r[0], r[1], r[2], r[3], fmt(r[4], '%.2f'),
          fmt(r[5]) + '/' + fmt(r[6]), ('%.1f%%' % (100 * r[7])) if r[7] is not None else 'n/a',
          fmt(r[8]) + '/' + fmt(r[9])))
print('\nchrF full = character chrF on the FULL dev set (1,741 sentences), invisible characters removed.')
print('Not comparable with earlier numbers: the data and tokenizer changed. Compare runs within this grid.')
open('/kaggle/working/summary.txt', 'w').write(repr(rows))'''


def nb(cells):
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
            "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


INTRO = """# drfn clean grid: {labels} labels (seed 11)

Cleaned AGE text (invisible zero-width characters removed from Amharic,
punctuation detached from words), a BPE that keeps word boundaries, and
`gmm` with the balance regulariser off. Details: `build_report.json`.

| run | what it is |
|---|---|
{rows}

After the last epoch each run decodes the full dev set from `best.pt` and
writes `dev_hyps.json` and `final_eval.json` (character chrF on full dev,
surface-form routing probes at root and pattern level), so the results can be
fetched without weights."""

ROWS = {
    "plain_bpe": "| `plain_bpe` | no labels: the reference point for how much labels help |",
    "bpe_aux": "| `bpe_aux` | labels on two side heads |",
    "cmm_spec": "| `cmm_spec` | the section 3.4 CMM plus side heads; best translator on the old data |",
    "gmm": "| `gmm` | labels supervise the memory's own routing (grounded) |",
}
ARMS = {
    "geezrpa": ("plain_bpe bpe_aux cmm_spec gmm", "geezRPA"),
    "analyser": ("bpe_aux cmm_spec gmm", "ByT5 analyser"),
}
for arm, (variants, label_name) in ARMS.items():
    kdir = os.path.join(HERE, f"kernel_{arm}")
    os.makedirs(kdir, exist_ok=True)
    rows = "\n".join(ROWS[v] for v in variants.split())
    json.dump(nb([
        md(INTRO.format(labels=label_name, rows=rows)),
        code(f'SEED = 11\nLABELS = "{arm}"\nVARIANTS = "{variants}"'),
        code("!pip install -q tokenizers sacrebleu\n"
             "import torch; print(torch.__version__, torch.cuda.is_available())"),
        code(SETUP),
        code("env = dict(os.environ, SEED=str(SEED), LABELS=LABELS, VARIANTS=VARIANTS)\n"
             "print('exit', stream(['bash', 'run_clean.sh'], env=env))"),
        code(SUMMARY),
    ]), open(os.path.join(kdir, f"drfn-clean-{arm}.ipynb"), "w"), indent=1)
    json.dump({"id": f"{OWNER}/drfn-clean-{arm}", "title": f"drfn-clean-{arm}",
               "code_file": f"drfn-clean-{arm}.ipynb", "language": "python",
               "kernel_type": "notebook", "is_private": True, "enable_gpu": True,
               "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
               "dataset_sources": [f"{OWNER}/drfn-clean-pack"], "kernel_sources": [],
               "competition_sources": [], "model_sources": [],
               "machine_shape": "NvidiaTeslaT4"},
              open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: pack/dataset-metadata.json, kernel_geezrpa/, kernel_analyser/ written")
