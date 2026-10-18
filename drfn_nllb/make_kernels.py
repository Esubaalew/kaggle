"""Build the drfn-nllb dataset metadata + notebook for a Kaggle username.

    python3 make_kernels.py KAGGLE_USERNAME
Needs internet ON (downloads facebook/nllb-200-distilled-600M from Hugging Face).
"""
import json, os, sys
if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
json.dump({"title": "drfn-nllb-pack", "id": f"{OWNER}/drfn-nllb-pack",
           "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(HERE, "pack", "dataset-metadata.json"), "w"), indent=2)
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": s.splitlines(True)}
INTRO = """# drfn-nllb: the grounded root memory inside a pretrained translator

Host: NLLB-200-distilled-600M (Ge'ez is not one of its languages; source tag tir_Ethi).
Same cleaned AGE data and analyser labels as the small-model grid.

| run | what |
|---|---|
| zeroshot | NLLB as released |
| ft | fine-tuned |
| aux | fine-tuned + root/pattern side heads |
| gmm | fine-tuned + grounded memory after encoder layers 4 and 8 (starts as an exact no-op) |

Seeds 11 and 22 for ft / aux / gmm. Decoding: beam 5, no-repeat 4-gram."""
SETUP = r'''import glob, os, shutil, subprocess
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
pack_src = os.path.dirname(find('run_nllb.sh'))
PACK = '/kaggle/working/pack'
if not os.path.exists(PACK): shutil.copytree(pack_src, PACK)
os.chdir(PACK)
from huggingface_hub import snapshot_download
print(snapshot_download('facebook/nllb-200-distilled-600M'))'''
RUN = "print('exit', subprocess.call(['bash', 'run_nllb.sh']))"
SUMMARY = r'''import json
for p in sorted(glob.glob('/kaggle/working/nllb/*/eval.json')):
    r = json.load(open(p)); d = r['dev_beam5_nr4']; pr = r.get('probe') or {}; sw = r.get('slot_swap') or {}
    print('%-10s chrF %.2f chrF++ %.2f BLEU %.2f  best_ep %s  unseen own-slot %s  swap %s vs %s (n=%s)' % (
        r['run'], d['chrF_char'], d['chrF++'], d['BLEU'], r.get('best_epoch'),
        ('%.3f' % pr['own_slot_unseen']) if pr else '-',
        ('%.3f' % sw['success_swap']) if sw.get('n_items') else '-',
        ('%.3f' % sw['success_self']) if sw.get('n_items') else '-', sw.get('n_items', '-')))'''
nb = {"cells": [md(INTRO),
                code("!pip install -q sacrebleu sentencepiece\nimport torch, transformers; print(torch.__version__, transformers.__version__, torch.cuda.device_count())"),
                code(SETUP), code(RUN), code(SUMMARY)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
kdir = os.path.join(HERE, "kernel"); os.makedirs(kdir, exist_ok=True)
json.dump(nb, open(os.path.join(kdir, "drfn-nllb.ipynb"), "w"), indent=1)
json.dump({"id": f"{OWNER}/drfn-nllb", "title": "drfn-nllb", "code_file": "drfn-nllb.ipynb",
           "language": "python", "kernel_type": "notebook", "is_private": True,
           "enable_gpu": True, "enable_tpu": False, "enable_internet": True, "keywords": ["gpu"],
           "dataset_sources": [f"{OWNER}/drfn-nllb-pack"], "kernel_sources": [],
           "competition_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: pack/dataset-metadata.json, kernel/ written")
