"""G3/G4/G5 Ge'ez->Amharic small-model comparison. Char chrF (beam5_nr4, invisible chars removed) on the full
dev (1,741) and the LaBSE-confirmed aligned dev (1,233). Paired bootstrap (B=1000) of the seed-mean difference;
seed-level t-test (df=2, exact p)."""
import json, re, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'run_g3/pack/data/age_dev.tsv', encoding='utf-8')]
for g in ('run_g4', 'run_g5'):
    assert [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+g+'/pack/data/age_dev.tsv', encoding='utf-8')] == refs
AL = np.array(json.load(open('/Users/esubalew/is/researcher/CMM/research_20261017/translation/age_realign/dev_aligned_index.json')))
SYS = {'G3 plain': 'run_g3/pack/runs/plain_bpe_G3_wide_s{}', 'G3 bpe_aux': 'run_g3/pack/runs/bpe_aux_G3_wide_s{}',
       'G3 gmm-1024': 'run_g3/pack/runs/gmm_G3_wide_s{}', 'G3 gmm-2048': 'run_g3/pack/runs/gmm_G3_full_s{}',
       'G4 bpe_aux feat': 'run_g4/pack/runs/bpe_aux_G4_wide_feat_s{}', 'G4 gmm feat': 'run_g4/pack/runs/gmm_G4_wide_feat_s{}',
       'G4 gmm feat gate-any': 'run_g4/pack/runs/gmm_gateany_G4_wide_feat_s{}',
       'G5 bpe_aux': 'run_g5/pack/runs/bpe_aux_G5_wide_s{}', 'G5 gmm-1024': 'run_g5/pack/runs/gmm_G5_wide_s{}'}
SE = (11, 22, 33); m = CHRF(word_order=0); S = {}
for k, p in SYS.items():
    for e in SE:
        h = [strip(x) for x in json.load(open(K + p.format(e) + '/dev_hyps_beam5_nr4.json'))]; assert len(h) == len(refs)
        S[(k, e)] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda k, i: m._compute_score_from_stats(list(S[k][i].sum(0))).score
FULL = np.arange(len(refs)); rng = np.random.default_rng(0)
BS = {'full': [rng.integers(0, len(FULL), len(FULL)) for _ in range(1000)], 'aligned': [AL[rng.integers(0, len(AL), len(AL))] for _ in range(1000)]}
IX = {'full': FULL, 'aligned': AL}
def contrast(a, b, sub):
    f = lambda i: np.mean([sc((a, e), i) - sc((b, e), i) for e in SE]); bs = np.array([f(i) for i in BS[sub]])
    g = [sc((a, e), IX[sub]) - sc((b, e), IX[sub]) for e in SE]
    t = np.mean(g) / (np.std(g, ddof=1) / np.sqrt(3) + 1e-12); p = 1 - abs(t) / np.sqrt(t * t + 2)
    return {'diff': round(f(IX[sub]), 2), 'ci95': [round(np.percentile(bs, 2.5), 2), round(np.percentile(bs, 97.5), 2)],
            'per_seed': [round(x, 2) for x in g], 'seed_p': round(float(p), 3)}
out = {'mean': {sub: {k: round(np.mean([sc((k, e), IX[sub]) for e in SE]), 2) for k in SYS} for sub in IX},
       'per_seed_full': {k: [round(sc((k, e), FULL), 2) for e in SE] for k in SYS}, 'contrasts': {}}
C = [('G3 gmm-1024', 'G3 bpe_aux'), ('G3 gmm-2048', 'G3 bpe_aux'), ('G3 gmm-2048', 'G3 gmm-1024'), ('G3 bpe_aux', 'G3 plain'),
     ('G4 gmm feat', 'G4 bpe_aux feat'), ('G4 gmm feat gate-any', 'G4 bpe_aux feat'), ('G4 gmm feat gate-any', 'G4 gmm feat'),
     ('G4 gmm feat', 'G3 gmm-1024'), ('G4 bpe_aux feat', 'G3 bpe_aux'), ('G4 gmm feat gate-any', 'G3 bpe_aux'),
     ('G5 gmm-1024', 'G5 bpe_aux'), ('G5 gmm-1024', 'G3 gmm-1024'), ('G5 bpe_aux', 'G3 bpe_aux')]
for a, b in C:
    for sub in IX: out['contrasts'][f'{a} - {b} [{sub}]'] = contrast(a, b, sub)
json.dump(out, open('g345_compare.json', 'w'), indent=1)
for sub in IX: print(sub, out['mean'][sub])
for k, v in out['contrasts'].items(): print(f'{k:55s} {v}')
