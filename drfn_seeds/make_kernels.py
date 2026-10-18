"""drfn-plain-seeds: no-label baseline (plain_bpe) at seeds 22 and 33, then the
same beam5 + no-repeat-4gram decoding as drfn-eval2.  Reuses two existing
datasets (no upload): drfn-clean-pack (training code + cleaned data) and
drfn-eval2-pack (beam decoding code).

    python3 make_kernels.py KAGGLE_USERNAME
"""
import json, os, sys
if len(sys.argv) != 2:
    sys.exit(__doc__)
OWNER = sys.argv[1]
HERE = os.path.dirname(os.path.abspath(__file__))
def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.splitlines(True)}
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {},
                     "outputs": [], "source": s.splitlines(True)}
INTRO = """# drfn-plain-seeds: the no-label baseline at two more seeds

"Labels help translation" (+1.58 chrF with beam decoding) is our largest
translation result, but its no-label baseline `plain_bpe` had one seed (11).
This trains seeds 22 and 33 in parallel (one per GPU), same cleaned data,
tokenizer and settings, then decodes with beam 5 + no-repeat 4-gram."""
SETUP = r'''import glob, os, shutil, subprocess
def find(pat):
    hits = sorted(glob.glob('/kaggle/input/**/' + pat, recursive=True))
    if not hits: raise RuntimeError('not found under /kaggle/input: ' + pat)
    return hits[0]
PACK = '/kaggle/working/pack'
if not os.path.exists(PACK): shutil.copytree(os.path.dirname(find('run_clean.sh')), PACK)
ev = os.path.dirname(find('run_eval2.sh'))
for f in ('beam.py', 'slot_tests.py', 'eval2.py'):
    shutil.copy(os.path.join(ev, f), PACK)
os.chdir(PACK); print(os.listdir(PACK))'''
TRAIN = r'''import torch
n = torch.cuda.device_count(); print('GPUs', n)
procs = []
for k, seed in enumerate((22, 33)):
    env = dict(os.environ, SEED=str(seed), LABELS='geezrpa', VARIANTS='plain_bpe',
               CUDA_VISIBLE_DEVICES=str(k % max(n, 1)))
    log = open(f'/kaggle/working/train_s{seed}.txt', 'w')
    procs.append(subprocess.Popen(['bash', 'run_clean.sh'], env=env, stdout=log, stderr=subprocess.STDOUT))
    if n < 2: procs[-1].wait()
print('exit codes', [p.wait() for p in procs])
for seed in (22, 33):
    print(seed, [l for l in open(f'/kaggle/working/train_s{seed}.txt') if 'final eval' in l or 'Error' in l][-3:])'''
DECODE = r'''print('exit', subprocess.call(['python', '-u', 'eval2.py', '--ckpt-root', 'runs',
                               '--out', '/kaggle/working/eval2']))
import json
for p in sorted(glob.glob('/kaggle/working/eval2/*/eval2.json')):
    r = json.load(open(p)); d = r['decode']
    print(r['run'], 'beam5_nr4 chrF %.2f  BLEU %.2f  (best epoch %s)' % (
        d['beam5_nr4']['chrF_char'], d['beam5_nr4']['BLEU'], r['best_epoch']))'''
nb = {"cells": [md(INTRO), code("!pip install -q tokenizers sacrebleu"), code(SETUP), code(TRAIN), code(DECODE)],
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(HERE, "kernel", "drfn-plain-seeds.ipynb"), "w"), indent=1)
json.dump({"id": f"{OWNER}/drfn-plain-seeds", "title": "drfn-plain-seeds",
           "code_file": "drfn-plain-seeds.ipynb", "language": "python", "kernel_type": "notebook",
           "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": True,
           "keywords": ["gpu"],
           "dataset_sources": [f"{OWNER}/drfn-clean-pack", f"{OWNER}/drfn-eval2-pack"],
           "kernel_sources": [], "competition_sources": [], "model_sources": [],
           "machine_shape": "NvidiaTeslaT4"},
          open(os.path.join(HERE, "kernel", "kernel-metadata.json"), "w"), indent=2)
print(f"owner {OWNER}: kernel/ written")
