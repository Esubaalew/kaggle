"""NLLB fix runs: two-seed paired bootstrap (char chrF, beam5+nr4, invisible chars removed)."""
import json, re, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'drfn_nllb_fix/pack/data/age_dev.tsv', encoding='utf-8')]
P = {}
for s in (11, 22):
    P[f'ft_s{s}'] = f'run_nllb/nllb/ft_s{s}'
    P[f'gmm_free_s{s}'] = f'run_nllb_fix/nllb_fix/gmm_free_s{s}'
    P[f'gmm_s{s}'] = f'run_nllb_fix2/nllb_fix/gmm_s{s}'
    P[f'gmm_wide_s{s}'] = f'run_nllb_fix2/nllb_fix/gmm_wide_s{s}'
m = CHRF(word_order=0); S = {}
for r, d in P.items():
    h = [strip(x) for x in json.load(open(K+d+'/dev_hyps.json'))]; assert len(h) == len(refs), r
    S[r] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda r, i: m._compute_score_from_stats(list(S[r][i].sum(0))).score
B = 2000; idx = np.random.default_rng(0).integers(0, len(refs), size=(B, len(refs)))
def mean2(sys_, i): return np.mean([sc(f'{sys_}_s{s}', i) for s in (11, 22)])
def contrast(a, b):
    f = lambda i: mean2(a, i) - mean2(b, i)
    bs = np.array([f(i) for i in idx]); pt = f(np.arange(len(refs)))
    return [round(pt, 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)]
out = {'scores': {r: round(sc(r, np.arange(len(refs))), 3) for r in P},
       'contrasts': {f'{a} - {b}': contrast(a, b) for a, b in
                     [('gmm', 'gmm_free'), ('gmm_wide', 'gmm_free'), ('gmm_wide', 'gmm'),
                      ('gmm', 'ft'), ('gmm_wide', 'ft'), ('gmm_free', 'ft')]}}
for r in P:
    if not r.startswith('ft'):
        e = json.load(open(K+P[r]+'/eval.json')); out.setdefault('eval', {})[r] = {
            'dev_loss': round(e['best_dev_loss'], 4), 'BLEU': round(e['dev_beam5_nr4']['BLEU'], 2),
            'probe': e['probe'], 'slot_swap': e.get('slot_swap')}
json.dump(out, open(K+'analysis/nllb_fix_compare.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps(out['scores'])); [print(k, v) for k, v in out['contrasts'].items()]
