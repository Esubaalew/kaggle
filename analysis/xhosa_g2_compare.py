"""isiXhosa->English G2 (small model, 1,024-slot stem labels, v2 pseudo-stems). Paired bootstrap over dev
sentences of the seed-mean char chrF (beam5_nr4, invisible chars removed), B=2000; also a seed-level t-test."""
import json, re, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/run_xhosa_g2/pack/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'data/age_dev.tsv', encoding='utf-8')]
SY, SE = ('plain_bpe', 'bpe_aux', 'gmm'), (11, 22, 33)
m = CHRF(word_order=0); S = {}
for s in SY:
    for e in SE:
        h = [strip(x) for x in json.load(open(f'{K}runs/{s}_G2_wide_s{e}/dev_hyps_beam5_nr4.json'))]
        assert len(h) == len(refs); S[(s, e)] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda k, i: m._compute_score_from_stats(list(S[k][i].sum(0))).score
full = np.arange(len(refs)); idx = np.random.default_rng(0).integers(0, len(refs), size=(2000, len(refs)))
def contrast(a, b):
    f = lambda i: np.mean([sc((a, e), i) - sc((b, e), i) for e in SE]); bs = np.array([f(i) for i in idx])
    gaps = [sc((a, e), full) - sc((b, e), full) for e in SE]
    tt = np.mean(gaps) / (np.std(gaps, ddof=1) / np.sqrt(len(gaps))); pv = 1 - abs(tt) / np.sqrt(tt * tt + 2)  # exact two-sided p for df=2
    return {'diff': round(f(full), 3), 'ci95': [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
            'p_le0': round(float((bs <= 0).mean()), 4), 'per_seed_gap': [round(g, 3) for g in gaps],
            'seed_t_df2': round(float(tt), 2), 'seed_p_two_sided': round(float(pv), 4)}
out = {'n_dev': len(refs), 'chrF_char': {f'{s}_s{e}': round(sc((s, e), full), 2) for s in SY for e in SE},
       'mean_chrF_char': {s: round(np.mean([sc((s, e), full) for e in SE]), 2) for s in SY},
       'contrasts': {'gmm - bpe_aux': contrast('gmm', 'bpe_aux'), 'gmm - plain_bpe': contrast('gmm', 'plain_bpe'),
                     'bpe_aux - plain_bpe': contrast('bpe_aux', 'plain_bpe')}}
for e in SE:
    out.setdefault('probe_root_gmm', {})[e] = json.load(open(f'{K}runs/gmm_G2_wide_s{e}/final_eval.json')).get('probe_root')
json.dump(out, open('xhosa_g2_compare.json', 'w'), indent=1, default=str)
print(out['mean_chrF_char']); [print(k, v) for k, v in out['contrasts'].items()]
