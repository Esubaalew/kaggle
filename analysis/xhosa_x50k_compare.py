"""isiXhosa->English X50k (50,000 training pairs; G2 code, config and dev set) vs G2 (13,653 pairs).
Char chrF of beam5_nr4 dev hypotheses (invisible chars removed); paired bootstrap over dev sentences (B=2000)
of the seed-mean gap; seed-level t-test (X50k: seeds 11/22, df=1). G2 is re-scored on the same seeds 11/22."""
import json, re, math, numpy as np
from sacrebleu.metrics import CHRF, BLEU
K = '/Users/esubalew/is/researcher/kaggle/'
X, G = K + 'run_x50k/pack/', K + 'run_xhosa_g2/pack/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
rd = lambda P: [strip(l.rstrip('\n').split('\t')[1]) for l in open(P + 'data/age_dev.tsv', encoding='utf-8')]
refs = rd(X); assert refs == rd(G), 'dev sets differ'
SY, SE = ('plain_bpe', 'bpe_aux', 'gmm'), (11, 22)
m = CHRF(word_order=0); S = {}; H = {}
for tag, P, run in (('X50k', X, '{s}_X50k_wide_s{e}'), ('G2', G, '{s}_G2_wide_s{e}')):
    for s in SY:
        for e in SE:
            h = [strip(x) for x in json.load(open(P + 'runs/' + run.format(s=s, e=e) + '/dev_hyps_beam5_nr4.json'))]
            assert len(h) == len(refs); H[(tag, s, e)] = h
            S[(tag, s, e)] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda k, i: m._compute_score_from_stats(list(S[k][i].sum(0))).score
full = np.arange(len(refs)); idx = np.random.default_rng(0).integers(0, len(refs), size=(2000, len(refs)))
def contrast(a, b):
    f = lambda i: np.mean([sc(a + (e,), i) - sc(b + (e,), i) for e in SE]); bs = np.array([f(i) for i in idx])
    gaps = [sc(a + (e,), full) - sc(b + (e,), full) for e in SE]
    sd = np.std(gaps, ddof=1); tt = np.mean(gaps) / (sd / math.sqrt(2)) if sd > 0 else float('inf')
    return {'diff': round(f(full), 3), 'ci95': [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
            'p_le0': round(float((bs <= 0).mean()), 4), 'per_seed_gap': [round(g, 3) for g in gaps],
            'seed_t_df1': round(float(tt), 2), 'seed_p_two_sided': round(1 - 2 / math.pi * math.atan(abs(tt)), 4)}
bleu = BLEU(); chrpp = CHRF(word_order=2)
out = {'n_dev': len(refs), 'seeds': SE,
       'chrF_char': {f'{t}_{s}_s{e}': round(sc((t, s, e), full), 2) for t in ('G2', 'X50k') for s in SY for e in SE},
       'mean_chrF_char': {f'{t}_{s}': round(np.mean([sc((t, s, e), full) for e in SE]), 2) for t in ('G2', 'X50k') for s in SY},
       'mean_chrF++': {f'{t}_{s}': round(np.mean([chrpp.corpus_score(H[(t, s, e)], [refs]).score for e in SE]), 2) for t in ('G2', 'X50k') for s in SY},
       'mean_BLEU': {f'{t}_{s}': round(np.mean([bleu.corpus_score(H[(t, s, e)], [refs]).score for e in SE]), 2) for t in ('G2', 'X50k') for s in SY},
       'contrasts': {'X50k gmm - bpe_aux': contrast(('X50k', 'gmm'), ('X50k', 'bpe_aux')),
                     'X50k gmm - plain_bpe': contrast(('X50k', 'gmm'), ('X50k', 'plain_bpe')),
                     'X50k bpe_aux - plain_bpe': contrast(('X50k', 'bpe_aux'), ('X50k', 'plain_bpe')),
                     'G2 gmm - bpe_aux (seeds 11/22)': contrast(('G2', 'gmm'), ('G2', 'bpe_aux')),
                     'data X50k - G2, gmm': contrast(('X50k', 'gmm'), ('G2', 'gmm')),
                     'data X50k - G2, bpe_aux': contrast(('X50k', 'bpe_aux'), ('G2', 'bpe_aux')),
                     'data X50k - G2, plain_bpe': contrast(('X50k', 'plain_bpe'), ('G2', 'plain_bpe'))}}
for s in SY:
    for e in SE:
        fe = json.load(open(f'{X}runs/{s}_X50k_wide_s{e}/final_eval.json'))
        out.setdefault('best_epoch', {})[f'{s}_s{e}'] = fe.get('best_epoch')
        if s == 'gmm': out.setdefault('probe_root_gmm', {})[e] = fe.get('probe_root')
try: out['routing_ablation_gmm_s11'] = json.load(open(X + 'runs/gmm_X50k_wide_s11/routing_ablation.json'))
except FileNotFoundError: pass
json.dump(out, open('xhosa_x50k_compare.json', 'w'), indent=1, default=str)
