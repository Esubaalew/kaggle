"""Wide (1024-slot) vs 256-slot small-model arms, fix code. Paired bootstrap over dev sentences of the
seed-mean char chrF (beam5_nr4, invisible chars removed). Colab and Kaggle runs share identical configs."""
import json, re, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'run_fix_clean/pack/data/age_dev.tsv', encoding='utf-8')]
C, W, F = 'run_fix_wide_colab/runs/', 'run_fix_wide_s22/pack/runs/', 'run_fix_clean/pack/runs/'
P = {'gmm_wide_s11': C+'gmm_analyser_wide_s11', 'gmm_wide_s22': C+'gmm_analyser_wide_s22', 'gmm_wide_s33': W+'gmm_analyser_wide_s33',
     'bpe_aux_wide_s11': C+'bpe_aux_analyser_wide_s11', 'bpe_aux_wide_s22': W+'bpe_aux_analyser_wide_s22', 'bpe_aux_wide_s33': W+'bpe_aux_analyser_wide_s33',
     'gmm_256_s11': F+'gmm_analyser_s11', 'gmm_256_s22': F+'gmm_analyser_s22',
     'bpe_aux_256_s11': F+'bpe_aux_analyser_s11', 'bpe_aux_256_s22': F+'bpe_aux_analyser_s22'}
m = CHRF(word_order=0); S = {}
for r, d in P.items():
    h = [strip(x) for x in json.load(open(K+d+'/dev_hyps_beam5_nr4.json'))]; assert len(h) == len(refs), r
    S[r] = np.array(m._extract_corpus_statistics(h, [refs]), float)
sc = lambda r, i: m._compute_score_from_stats(list(S[r][i].sum(0))).score
idx = np.random.default_rng(0).integers(0, len(refs), size=(2000, len(refs)))
mean = lambda sysn, seeds, i: np.mean([sc(f'{sysn}_s{s}', i) for s in seeds])
def contrast(a, b, seeds):
    f = lambda i: mean(a, seeds, i) - mean(b, seeds, i); bs = np.array([f(i) for i in idx])
    return {'diff': round(f(np.arange(len(refs))), 3), 'ci95': [round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)],
            'p_le0': round(float((bs <= 0).mean()), 4), 'seeds': list(seeds)}
full = np.arange(len(refs))
out = {'scores': {r: round(sc(r, full), 3) for r in P},
       'per_seed_gap_wide': {s: round(sc(f'gmm_wide_s{s}', full) - sc(f'bpe_aux_wide_s{s}', full), 3) for s in (11, 22, 33)},
       'contrasts': {'gmm_wide - bpe_aux_wide (3 seeds)': contrast('gmm_wide', 'bpe_aux_wide', (11, 22, 33)),
                     'gmm_wide - bpe_aux_wide (seed 33 only, same session)': contrast('gmm_wide', 'bpe_aux_wide', (33,)),
                     'gmm_wide - bpe_aux_wide (seeds 11,22)': contrast('gmm_wide', 'bpe_aux_wide', (11, 22)),
                     'gmm_256 - bpe_aux_256 (seeds 11,22)': contrast('gmm_256', 'bpe_aux_256', (11, 22)),
                     'gmm_wide - gmm_256 (seeds 11,22)': contrast('gmm_wide', 'gmm_256', (11, 22)),
                     'bpe_aux_wide - bpe_aux_256 (seeds 11,22)': contrast('bpe_aux_wide', 'bpe_aux_256', (11, 22))},
       'note': 'seed 11 both Colab; seed 22 gmm Colab, bpe_aux Kaggle; seed 33 both Kaggle. Configs identical except torch build (cu130 vs cu128).'}
json.dump(out, open(K+'analysis/wide_compare.json', 'w'), indent=1)
print(out['per_seed_gap_wide']); [print(k, v) for k, v in out['contrasts'].items()]
