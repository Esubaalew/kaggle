"""Small model, fix code, cleaned 256-slot analyser labels: two-seed paired bootstrap (char chrF, beam5_nr4)."""
import json, re, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'run_fix_clean/pack/data/age_dev.tsv', encoding='utf-8')]
m = CHRF(word_order=0); S = {}
for v in ('bpe_aux', 'cmm_spec', 'gmm'):
    for s in (11, 22):
        h = [strip(x) for x in json.load(open(f'{K}run_fix_clean/pack/runs/{v}_analyser_s{s}/dev_hyps_beam5_nr4.json'))]
        assert len(h) == len(refs); S[f'{v}_s{s}'] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda r, i: m._compute_score_from_stats(list(S[r][i].sum(0))).score
idx = np.random.default_rng(0).integers(0, len(refs), size=(2000, len(refs)))
mean2 = lambda v, i: np.mean([sc(f'{v}_s{s}', i) for s in (11, 22)])
def contrast(a, b):
    f = lambda i: mean2(a, i) - mean2(b, i); bs = np.array([f(i) for i in idx])
    return [round(f(np.arange(len(refs))), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)]
out = {'scores': {r: round(sc(r, np.arange(len(refs))), 3) for r in S},
       'contrasts': {f'{a} - {b}': contrast(a, b) for a, b in [('gmm','bpe_aux'), ('cmm_spec','bpe_aux'), ('gmm','cmm_spec')]}}
json.dump(out, open(K+'analysis/fix_clean_compare.json', 'w'), indent=1)
print(json.dumps(out['scores'])); [print(k, v) for k, v in out['contrasts'].items()]
