"""Labels vs no labels with 3 seeds of plain_bpe (beam5 + no-repeat-4gram)."""
import json, re, os, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]')
strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'run_overnight/pack/data/age_dev.tsv', encoding='utf-8')]
H = {}
for d in ('run_eval2', 'run_seeds'):
    for r in os.listdir(K+d+'/eval2'):
        p = K+d+'/eval2/'+r+'/hyps_beam5_nr4.json'
        if os.path.exists(p): H[r] = [strip(x) for x in json.load(open(p))]
m = CHRF(word_order=0)
S = {r: np.array(m._extract_corpus_statistics(h, [refs]), float) for r, h in H.items()}
sc = lambda r, i: m._compute_score_from_stats(list(S[r][i].sum(0))).score
rng = np.random.default_rng(0); idx = rng.integers(0, len(refs), (2000, len(refs)))
def C(plus, minus):
    f = lambda i: np.mean([sc(r, i) for r in plus]) - np.mean([sc(r, i) for r in minus])
    a = np.arange(len(refs)); b = np.array([f(i) for i in idx])
    return [round(f(a), 3), round(np.percentile(b, 2.5), 3), round(np.percentile(b, 97.5), 3)]
plain = ['plain_bpe_geezrpa_s11', 'plain_bpe_geezrpa_s22', 'plain_bpe_geezrpa_s33']
out = {'scores': {r: round(sc(r, np.arange(len(refs))), 3) for r in sorted(H) if 'plain' in r},
       'plain_seed_sd': None, 'contrasts': {}}
v = [out['scores'][r] for r in plain]; out['plain_seed_sd'] = round(float(np.std(v, ddof=1)), 3)
out['contrasts']['bpe_aux_geezrpa(2 seeds) - plain(3 seeds)'] = C(['bpe_aux_geezrpa_s11', 'bpe_aux_geezrpa_s22'], plain)
out['contrasts']['bpe_aux_analyser(2) - plain(3)'] = C(['bpe_aux_analyser_s11', 'bpe_aux_analyser_s22'], plain)
out['contrasts']['gmm_geezrpa(2) - plain(3)'] = C(['gmm_geezrpa_s11', 'gmm_geezrpa_s22'], plain)
out['contrasts']['gmm_analyser(2) - plain(3)'] = C(['gmm_analyser_s11', 'gmm_analyser_s22'], plain)
out['contrasts']['cmm_spec_analyser(2) - plain(3)'] = C(['cmm_spec_analyser_s11', 'cmm_spec_analyser_s22'], plain)
# worst case across seeds: lowest labelled run minus highest plain run
lab = ['bpe_aux_geezrpa_s11', 'bpe_aux_geezrpa_s22', 'bpe_aux_analyser_s11', 'bpe_aux_analyser_s22']
out['min_labelled_minus_max_plain'] = round(min(sc(r, np.arange(len(refs))) for r in lab) - max(v), 3)
json.dump(out, open(K+'analysis/labels_effect.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
