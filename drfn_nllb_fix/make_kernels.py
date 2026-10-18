"""Build the drfn-nllb-fix dataset metadata + notebook for a Kaggle username.

    python3 make_kernels.py KAGGLE_USERNAME
"""
import json
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))

json.dump({"title": "drfn-nllb-fix-pack", "id": f"{OWNER}/drfn-nllb-fix-pack",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(HERE, "pack", "dataset-metadata.json"), "w"), indent=2)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}


def code(s):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": s.splitlines(True)}


INTRO = """# drfn-nllb-fix: Fixed Grounded Morphological Memory inside NLLB-200-600M

Fixes all 5 NLLB issues:
1. `geez-dataset`-verified labels (`labels_analyser.json` 256 slots and `labels_analyser_wide.json` 1024 slots).
2. Memory placed after Encoder Layers 6 and 12 (`MEM_LAYERS = (5, 11)`) so the top memory feeds directly into decoder cross-attention.
3. Three-part Grounded Memory loss (Key routing CE + Value alignment `c @ M_V.T` + Null-Safe Gate `BCE(lambda_t, root >= 1)`).
4. Separate gradient norm clipping (`clip_grad_norm_(base, 1.0)` and `clip_grad_norm_(new_params, 1.0)`).
"""

SETUP = r'''import glob, os, shutil, subprocess
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
pack_src = os.path.dirname(find('run_nllb_fix.sh'))
PACK = '/kaggle/working/pack'
if not os.path.exists(PACK): shutil.copytree(pack_src, PACK)
os.chdir(PACK)
from huggingface_hub import snapshot_download
print(snapshot_download('facebook/nllb-200-distilled-600M'))'''

RUN = "print('exit', subprocess.call(['bash', 'run_nllb_fix.sh']))"

SUMMARY = r'''import json
for p in sorted(glob.glob('/kaggle/working/nllb_fix/*/eval.json')):
    r = json.load(open(p)); d = r['dev_beam5_nr4']; pr = r.get('probe') or {}; sw = r.get('slot_swap') or {}
    print('%-16s loss %.4f  chrF %.2f chrF++ %.2f BLEU %.2f  ep %s  unseen-slot %s  gate(rt/null) %s/%s  swap %s/%s' % (
        r['run'], r.get('best_dev_loss', 0.0), d['chrF_char'], d['chrF++'], d['BLEU'], r.get('best_epoch'),
        ('%.3f' % pr['own_slot_unseen']) if pr else '-',
        ('%.2f' % pr['mean_gate_rooted']) if pr.get('mean_gate_rooted') is not None else '-',
        ('%.2f' % pr['mean_gate_rootless']) if pr.get('mean_gate_rootless') is not None else '-',
        ('%.3f' % sw['success_swap']) if sw.get('n_items') else '-',
        ('%.3f' % sw['success_self']) if sw.get('n_items') else '-'))'''

nb = {"cells": [md(INTRO),
                code("!pip install -q sacrebleu sentencepiece\nimport torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count())"),
                code(SETUP), code(RUN), code(SUMMARY)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
kdir = os.path.join(HERE, "kernel")
os.makedirs(kdir, exist_ok=True)
json.dump(nb, open(os.path.join(kdir, "drfn-nllb-fix.ipynb"), "w"), indent=1)
json.dump({"id": f"{OWNER}/drfn-nllb-fix", "title": "drfn-nllb-fix", "code_file": "drfn-nllb-fix.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
           "dataset_sources": [f"{OWNER}/drfn-nllb-fix-pack"], "kernel_sources": [],
           "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: pack/dataset-metadata.json, kernel/ written")
