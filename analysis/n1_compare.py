"""NLLB-200-distilled-600M N1 (repaired AGE train 13,024, feature-bundle labels analyser_wide_feat). Arms: ft (plain
fine-tune), aux (labels as side heads), gmm gate-any, gmm gate-any with NLLB frozen for epoch 1 (frz1); seeds 11/22/33.
Char chrF of the beam-5 no-repeat-4 dev hypotheses (dev_hyps.json; invisible chars removed) on the full dev (1,741)
and the aligned dev (1,233). Paired bootstrap (B=1000) of the seed-mean gap; seed-level paired t-test (df=2)."""
import json, re, numpy as np
from scipy import stats
from sacrebleu.metrics import CHRF, BLEU
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K + 'run_n1/pack/data/age_dev.tsv', encoding='utf-8')]
AL = np.array(json.load(open(K + 'run_n1/pack/data/dev_aligned_index.json')))
SYS = {'ft': 'ft_wide_feat_s{}', 'aux': 'aux_wide_feat_s{}', 'gmm': 'gmm_wide_feat_gateany_s{}', 'gmm_frz1': 'gmm_wide_feat_gateany_frz1_s{}'}
SE = (11, 22, 33); m = CHRF(word_order=0); S, H, E = {}, {}, {}
for k, p in SYS.items():
    for e in SE:
        d = K + 'run_n1/nllb_n1/' + p.format(e)
        h = [strip(x) for x in json.load(open(d + '/dev_hyps.json'))]; assert len(h) == len(refs)
        H[(k, e)] = h; S[(k, e)] = np.array(m._extract_corpus_statistics(h, [refs]), float); E[(k, e)] = json.load(open(d + '/eval.json'))
sc = lambda k, i: m._compute_score_from_stats(list(S[k][i].sum(0))).score
FULL = np.arange(len(refs)); rng = np.random.default_rng(0)
BS = {'full': [rng.integers(0, len(FULL), len(FULL)) for _ in range(1000)], 'aligned': [AL[rng.integers(0, len(AL), len(AL))] for _ in range(1000)]}
IX = {'full': FULL, 'aligned': AL}
def contrast(a, b, sub):
    f = lambda i: np.mean([sc((a, e), i) - sc((b, e), i) for e in SE]); bs = np.array([f(i) for i in BS[sub]])
    g = [sc((a, e), IX[sub]) - sc((b, e), IX[sub]) for e in SE]; t = stats.ttest_1samp(g, 0.0)
    return {'diff': round(f(IX[sub]), 2), 'ci95': [round(np.percentile(bs, 2.5), 2), round(np.percentile(bs, 97.5), 2)],
            'p_le0': round(float((bs <= 0).mean()), 3), 'per_seed': [round(x, 2) for x in g], 'seed_p': round(float(t.pvalue), 4)}
out = {'mean_chrF_char': {sub: {k: round(np.mean([sc((k, e), IX[sub]) for e in SE]), 2) for k in SYS} for sub in IX},
       'per_seed_full': {k: [round(sc((k, e), FULL), 2) for e in SE] for k in SYS},
       'reported_eval_chrF_char': {k: [round(E[(k, e)]['dev_beam5_nr4']['chrF_char'], 2) for e in SE] for k in SYS},
       'mean_chrF++_full': {k: round(np.mean([CHRF(word_order=2).corpus_score(H[(k, e)], [refs]).score for e in SE]), 2) for k in SYS},
       'mean_BLEU_full': {k: round(np.mean([BLEU().corpus_score(H[(k, e)], [refs]).score for e in SE]), 2) for k in SYS},
       'best_epoch': {k: [E[(k, e)].get('best_epoch') for e in SE] for k in SYS}, 'contrasts': {}}
for a, b in [('gmm', 'aux'), ('gmm', 'ft'), ('gmm_frz1', 'aux'), ('gmm_frz1', 'gmm'), ('aux', 'ft'), ('gmm_frz1', 'ft')]:
    for sub in IX: out['contrasts'][f'{a} - {b} [{sub}]'] = contrast(a, b, sub)
for k in ('gmm', 'gmm_frz1'):
    for e in SE:
        pr, sw = E[(k, e)].get('probe', {}), E[(k, e)].get('slot_swap', {})
        out.setdefault('mechanism', {})[f'{k}_s{e}'] = {x: (round(v, 3) if isinstance(v, float) else v) for x, v in {**pr, **{'swap_' + y: z for y, z in sw.items()}}.items()}
json.dump(out, open('n1_compare.json', 'w'), indent=1)
