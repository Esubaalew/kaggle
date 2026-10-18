"""Two-seed paired-bootstrap comparison on full dev (char chrF, invisible chars removed)."""
import json, re, sys, numpy as np
from sacrebleu.metrics import CHRF
K = '/Users/esubalew/is/researcher/kaggle/'
CF = re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]')
strip = lambda s: CF.sub('', s)
refs = [strip(l.rstrip('\n').split('\t')[1]) for l in open(K+'run_overnight/pack/data/age_dev.tsv', encoding='utf-8')]
RUNS = {}
for d in ['run_clean_geezrpa', 'run_clean_analyser', 'run_overnight', 'run_dec']:
    import os
    for r in os.listdir(K+d+'/pack/runs'):
        p = K+d+'/pack/runs/'+r+'/dev_hyps.json'
        if os.path.exists(p): RUNS[r] = p
m = CHRF(word_order=0)
stats = {}
for r, p in sorted(RUNS.items()):
    h = [strip(x) for x in json.load(open(p))]
    assert len(h) == len(refs)
    stats[r] = np.array(m._extract_corpus_statistics(h, [refs]), dtype=float)
score = lambda S: m._compute_score_from_stats(list(S.sum(0))).score
B = 2000; rng = np.random.default_rng(0)
idx = rng.integers(0, len(refs), size=(B, len(refs)))
def boot(f):
    # f maps an index array -> scalar; returns point, lo, hi
    pt = f(np.arange(len(refs))); bs = np.array([f(i) for i in idx])
    return pt, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
sc = lambda r, i: score(stats[r][i])
out = {'scores': {r: round(score(s), 3) for r, s in stats.items()}, 'contrasts': {}}
def contrast(name, plus, minus):
    """plus/minus: lists of (run) averaged with equal weight; difference of means."""
    if not all(r in stats for r in plus + minus): return
    f = lambda i: np.mean([sc(r, i) for r in plus]) - np.mean([sc(r, i) for r in minus])
    pt, lo, hi = boot(f); out['contrasts'][name] = [round(pt, 3), round(lo, 3), round(hi, 3)]
C = contrast
for s in ['s11', 's22']:
    for lab in ['geezrpa', 'analyser']:
        C(f'gmm-bpe_aux|{lab}|{s}', [f'gmm_{lab}_{s}'], [f'bpe_aux_{lab}_{s}'])
        C(f'cmm_spec-bpe_aux|{lab}|{s}', [f'cmm_spec_{lab}_{s}'], [f'bpe_aux_{lab}_{s}'])
        C(f'gmm-cmm_spec|{lab}|{s}', [f'gmm_{lab}_{s}'], [f'cmm_spec_{lab}_{s}'])
    for v in ['bpe_aux', 'gmm', 'cmm_spec']:
        C(f'{v}:analyser-geezrpa|{s}', [f'{v}_analyser_{s}'], [f'{v}_geezrpa_{s}'])
    # interaction (memory gain from better labels minus head gain)
    C(f'interaction_gmm_vs_heads|{s}', [f'gmm_analyser_{s}', f'bpe_aux_geezrpa_{s}'], [f'gmm_geezrpa_{s}', f'bpe_aux_analyser_{s}'])
# seed-to-seed differences (same system)
for v in ['bpe_aux_geezrpa', 'bpe_aux_analyser', 'gmm_geezrpa', 'gmm_analyser', 'cmm_spec_analyser']:
    C(f'seed22-seed11|{v}', [f'{v}_s22'], [f'{v}_s11'])
# two-seed means
for lab in ['geezrpa', 'analyser']:
    C(f'2seed gmm-bpe_aux|{lab}', [f'gmm_{lab}_s11', f'gmm_{lab}_s22'], [f'bpe_aux_{lab}_s11', f'bpe_aux_{lab}_s22'])
C('2seed cmm_spec-bpe_aux|analyser', ['cmm_spec_analyser_s11', 'cmm_spec_analyser_s22'], ['bpe_aux_analyser_s11', 'bpe_aux_analyser_s22'])
C('2seed gmm-cmm_spec|analyser', ['gmm_analyser_s11', 'gmm_analyser_s22'], ['cmm_spec_analyser_s11', 'cmm_spec_analyser_s22'])
for v in ['bpe_aux', 'gmm']:
    C(f'2seed {v}:analyser-geezrpa', [f'{v}_analyser_s11', f'{v}_analyser_s22'], [f'{v}_geezrpa_s11', f'{v}_geezrpa_s22'])
C('2seed interaction_gmm_vs_heads', ['gmm_analyser_s11', 'gmm_analyser_s22', 'bpe_aux_geezrpa_s11', 'bpe_aux_geezrpa_s22'],
  ['gmm_geezrpa_s11', 'gmm_geezrpa_s22', 'bpe_aux_analyser_s11', 'bpe_aux_analyser_s22'])
C('2seed labels: bpe_aux_geezrpa - plain_bpe(s11 only)', ['bpe_aux_geezrpa_s11', 'bpe_aux_geezrpa_s22'], ['plain_bpe_geezrpa_s11'])
# tokenizer diagnostic
C('wstok - metaspace|bpe_aux_geezrpa_s11', ['bpe_aux_geezrpa_s11_wstok'], ['bpe_aux_geezrpa_s11'])
C('wstok - 2seed metaspace mean', ['bpe_aux_geezrpa_s11_wstok'], ['bpe_aux_geezrpa_s11', 'bpe_aux_geezrpa_s22'])
# decoder memory
for v in ['gmm_dec', 'gmm_dec_free', 'gmm_dec_sep']:
    C(f'{v}-gmm|analyser|s11', [f'{v}_analyser_s11'], ['gmm_analyser_s11'])
    C(f'{v}-2seed gmm|analyser', [f'{v}_analyser_s11'], ['gmm_analyser_s11', 'gmm_analyser_s22'])
C('gmm_dec-gmm_dec_free|s11', ['gmm_dec_analyser_s11'], ['gmm_dec_free_analyser_s11'])
json.dump(out, open(K+'analysis/seed_compare.json', 'w'), indent=1, ensure_ascii=False)
for k, v in out['scores'].items(): print(f'{k:32s} {v:6.2f}')
for k, v in out['contrasts'].items(): print(f'{k:52s} {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]')
