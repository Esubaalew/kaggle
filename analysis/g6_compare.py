"""G6 Ge'ez->Amharic small model: G4 feature-bundle labels + gate trained open for any labelled word, on the repaired
AGE pairs of G5 (13,024). Char chrF (beam5_nr4, invisible chars removed) on the full dev (1,741) and the
LaBSE-confirmed aligned dev (1,233). Paired bootstrap (B=1000) of the seed-mean difference; seed-level paired
t-test (exact p, df = seeds - 1). Within-pack: seeds 11-55. Against G4/G5 (same code): seeds 11/22/33."""
import json, re, numpy as np
from scipy import stats
from sacrebleu.metrics import CHRF, BLEU
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
rd = lambda g: [strip(l.rstrip('\n').split('\t')[1]) for l in open(K + g + '/pack/data/age_dev.tsv', encoding='utf-8')]
refs = rd('run_g6'); assert all(rd(g) == refs for g in ('run_g3', 'run_g4', 'run_g5'))
AL = np.array(json.load(open('/Users/esubalew/is/researcher/CMM/research_20261017/translation/age_realign/dev_aligned_index.json')))
SYS = {'G6 bpe_aux': ('run_g6/pack/runs/bpe_aux_G6_wide_feat_s{}', (11, 22, 33, 44, 55)),
       'G6 gmm gate-any': ('run_g6/pack/runs/gmm_gateany_G6_wide_feat_s{}', (11, 22, 33, 44, 55)),
       'G4 bpe_aux feat': ('run_g4/pack/runs/bpe_aux_G4_wide_feat_s{}', (11, 22, 33)),
       'G4 gmm feat gate-any': ('run_g4/pack/runs/gmm_gateany_G4_wide_feat_s{}', (11, 22, 33)),
       'G5 bpe_aux': ('run_g5/pack/runs/bpe_aux_G5_wide_s{}', (11, 22, 33)),
       'G5 gmm-1024': ('run_g5/pack/runs/gmm_G5_wide_s{}', (11, 22, 33))}
m = CHRF(word_order=0); S, H = {}, {}
for k, (p, seeds) in SYS.items():
    for e in seeds:
        h = [strip(x) for x in json.load(open(K + p.format(e) + '/dev_hyps_beam5_nr4.json'))]; assert len(h) == len(refs)
        H[(k, e)] = h; S[(k, e)] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda k, i: m._compute_score_from_stats(list(S[k][i].sum(0))).score
FULL = np.arange(len(refs)); rng = np.random.default_rng(0)
BS = {'full': [rng.integers(0, len(FULL), len(FULL)) for _ in range(1000)], 'aligned': [AL[rng.integers(0, len(AL), len(AL))] for _ in range(1000)]}
IX = {'full': FULL, 'aligned': AL}
def contrast(a, b, sub, seeds):
    f = lambda i: np.mean([sc((a, e), i) - sc((b, e), i) for e in seeds]); bs = np.array([f(i) for i in BS[sub]])
    g = [sc((a, e), IX[sub]) - sc((b, e), IX[sub]) for e in seeds]; t = stats.ttest_1samp(g, 0.0)
    return {'seeds': list(seeds), 'diff': round(f(IX[sub]), 2), 'ci95': [round(np.percentile(bs, 2.5), 2), round(np.percentile(bs, 97.5), 2)],
            'p_le0': round(float((bs <= 0).mean()), 3), 'per_seed': [round(x, 2) for x in g], 'seed_t': round(float(t.statistic), 2), 'seed_p': round(float(t.pvalue), 4)}
S5, S3 = (11, 22, 33, 44, 55), (11, 22, 33)
out = {'mean_chrF_char': {sub: {k: round(np.mean([sc((k, e), IX[sub]) for e in SYS[k][1]]), 2) for k in SYS} for sub in IX},
       'mean_chrF_char_seeds_11_22_33': {sub: {k: round(np.mean([sc((k, e), IX[sub]) for e in S3]), 2) for k in SYS} for sub in IX},
       'per_seed_full': {k: {e: round(sc((k, e), FULL), 2) for e in SYS[k][1]} for k in SYS},
       'mean_chrF++_full': {k: round(np.mean([CHRF(word_order=2).corpus_score(H[(k, e)], [refs]).score for e in SYS[k][1]]), 2) for k in SYS},
       'mean_BLEU_full': {k: round(np.mean([BLEU().corpus_score(H[(k, e)], [refs]).score for e in SYS[k][1]]), 2) for k in SYS},
       'contrasts': {}}
C = [('G6 gmm gate-any', 'G6 bpe_aux', S5), ('G6 gmm gate-any', 'G6 bpe_aux', S3),
     ('G6 gmm gate-any', 'G5 gmm-1024', S3), ('G6 gmm gate-any', 'G4 gmm feat gate-any', S3),
     ('G6 bpe_aux', 'G4 bpe_aux feat', S3), ('G6 bpe_aux', 'G5 bpe_aux', S3)]
for a, b, seeds in C:
    for sub in IX: out['contrasts'][f'{a} - {b} [{sub}, {len(seeds)} seeds]'] = contrast(a, b, sub, seeds)
for e in S5:
    fe = json.load(open(f'{K}run_g6/pack/runs/gmm_gateany_G6_wide_feat_s{e}/final_eval.json'))
    pr = fe.get('probe_root') or {}
    out.setdefault('mechanism_gmm', {})[e] = {'root_consistency': (pr.get('all_types') or {}).get('consistency'),
        'permutation': (pr.get('all_types') or {}).get('permutation'), 'unseen_with_siblings': (pr.get('unseen_with_siblings') or {}).get('consistency'),
        'gate': {k: round(v, 3) for k, v in (pr.get('gate_probe') or {}).items() if k.startswith('mean')}, 'best_epoch': fe.get('best_epoch')}
ab = json.load(open(f'{K}run_g6/pack/runs/gmm_gateany_G6_wide_feat_s11/routing_ablation.json'))['conditions']
out['routing_ablation_s11'] = {k: round(v['chrF_char'], 2) for k, v in ab.items()}
json.dump(out, open('g6_compare.json', 'w'), indent=1)
