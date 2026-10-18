"""Build the drfn-eval2 dataset metadata + notebook for a Kaggle username.

    python3 make_kernels.py KAGGLE_USERNAME
The notebook reads checkpoints from that user's four earlier notebooks
(attached as kernel_sources), so they must belong to the same account.
"""
import json, os, sys
if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
json.dump({"title": "drfn-eval2-pack", "id": f"{OWNER}/drfn-eval2-pack",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(HERE, "pack", "dataset-metadata.json"), "w"), indent=2)

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": s.splitlines(True)}
INTRO = """# drfn-eval2: better decoding + causal slot tests (no training)

Reads `best.pt` from the four earlier notebooks and, for every run:

1. **Re-decodes the full dev set** with beam 5 + no-repeat-4-gram blocking (fixed in
   advance), plus beam 5 without blocking and greedy + blocking. Greedy decoding
   looped in 49-68% of outputs.
2. **Slot swap** (grounded memory runs): force one Ge'ez word's root slot to another
   root and check whether the Amharic output switches to that root's words.
3. **Memory ablation**: switch the memory off and measure the loss increase on
   Amharic tokens linked to rare / unseen Ge'ez word forms."""
SETUP = r'''import glob, os, shutil, subprocess
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
pack_src = os.path.dirname(find('run_eval2.sh'))
PACK = '/kaggle/working/pack'
if not os.path.exists(PACK): shutil.copytree(pack_src, PACK)
os.chdir(PACK)
cks = sorted(glob.glob('/kaggle/input/**/best.pt', recursive=True))
print(len(cks), 'checkpoints'); print('\n'.join(cks))'''
RUN = "print('exit', subprocess.call(['bash', 'run_eval2.sh']))"
SUMMARY = r'''import json
rows = []
for p in sorted(glob.glob('/kaggle/working/eval2/*/eval2.json')):
    r = json.load(open(p)); d = r['decode']
    sw = r.get('slot_swap') or {}
    print('%-28s b5nr4 %.2f  b5 %.2f  b1nr4 %.2f  BLEU(b5nr4) %.2f  swap %s/%s  n=%s' % (
        r['run'], d['beam5_nr4']['chrF_char'], d['beam5']['chrF_char'], d['beam1_nr4']['chrF_char'],
        d['beam5_nr4']['BLEU'], ('%.3f' % sw['success_swap']) if sw else '-',
        ('%.3f' % sw['success_self']) if sw else '-', sw.get('n_items', '-')))
for f in glob.glob('/kaggle/working/pack/runs'): pass'''
nb = {"cells": [md(INTRO), code("!pip install -q tokenizers sacrebleu\nimport torch; print(torch.__version__, torch.cuda.device_count())"),
                code(SETUP), code(RUN), code(SUMMARY)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
kdir = os.path.join(HERE, "kernel"); os.makedirs(kdir, exist_ok=True)
json.dump(nb, open(os.path.join(kdir, "drfn-eval2.ipynb"), "w"), indent=1)
json.dump({"id": f"{OWNER}/drfn-eval2", "title": "drfn-eval2", "code_file": "drfn-eval2.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
           "dataset_sources": [f"{OWNER}/drfn-eval2-pack"],
           "kernel_sources": [f"{OWNER}/drfn-clean-geezrpa", f"{OWNER}/drfn-clean-analyser",
                              f"{OWNER}/drfn-clean-overnight", f"{OWNER}/drfn-dec"],
           "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: pack/dataset-metadata.json, kernel/ written")
