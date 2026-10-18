"""Collect every fix-code run (small 256-slot, small 1024-slot wide, NLLB) into one tidy CSV."""
import json, glob, os, csv
K = '/Users/esubalew/is/researcher/kaggle/'
rows = []
def small(path, platform):
    for f in sorted(glob.glob(K + path + '/*/final_eval.json')):
        run = f.split('/')[-2]; e = json.load(open(f))
        if 'beam5_nr4' not in e: continue
        v, rest = run.split('_analyser')
        labels = 'analyser_wide (1024)' if rest.startswith('_wide') else 'analyser (256)'
        r = dict(model='small 20.3M', system=v, labels=labels, seed=int(run.rsplit('_s', 1)[1]), platform=platform,
                 chrF=e['beam5_nr4']['chrF_char'], chrFpp=e['beam5_nr4']['chrF++'], BLEU=e['beam5_nr4']['BLEU'],
                 greedy_chrF=e['chrF_char_full_dev'], best_epoch=e.get('best_epoch'))
        p = e.get('probe_root')
        if p:
            r.update(root_consistency=p['all_types']['consistency'], root_perm=p['all_types']['permutation'],
                     unseen_sib_consistency=p['unseen_with_siblings']['consistency'],
                     unseen_own_slot=p.get('unseen_slot_accuracy'),
                     gate_rooted=p['gate_probe']['mean_gate_rooted'], gate_rootless=p['gate_probe']['mean_gate_rootless'])
        w = e.get('slot_swap')
        if w:
            r.update(word_lost_swap=1 - w['keep_r_swap'], word_lost_kill=1 - w['keep_r_kill'], swap_success=w['success_swap'],
                     dNLL_root_corrupt=w['tf_delta_nll_root_tokens'], dNLL_other_corrupt=w['tf_delta_nll_other_tokens'])
        a = e.get('ablation')
        if a:
            r.update(memoff_dNLL_unlinked=a['unlinked']['delta'], memoff_dNLL_unseen=a['unseen']['delta'],
                     memoff_dNLL_50plus=a['50+']['delta'])
        rows.append(r)
small('run_fix_clean/pack/runs', 'Kaggle')
small('run_fix_wide_colab/runs', 'Colab')
small('run_fix_wide_s22/pack/runs', 'Kaggle')
for d in ['run_nllb_fix/nllb_fix/gmm_free_s11', 'run_nllb_fix/nllb_fix/gmm_free_s22'] + sorted(glob.glob(K + 'run_nllb_fix2/nllb_fix/*_s*')):
    d = d if d.startswith('/') else K + d; e = json.load(open(d + '/eval.json'))
    p, w = e['probe'], e.get('slot_swap') or {}
    r = dict(model='NLLB-600M', system=e['system'], labels={'analyser': 'analyser (256)', 'analyser_wide': 'analyser_wide (1024)'}[e['labels']],
             seed=e['seed'], platform='Kaggle', chrF=e['dev_beam5_nr4']['chrF_char'], chrFpp=e['dev_beam5_nr4']['chrF++'],
             BLEU=e['dev_beam5_nr4']['BLEU'], dev_loss=e['best_dev_loss'], best_epoch=e['best_epoch'],
             unseen_own_slot=p['own_slot_unseen'], seen_own_slot=p['own_slot_seen'],
             gate_rooted=p['mean_gate_rooted'], gate_rootless=p['mean_gate_rootless'])
    if w: r.update(word_lost_swap=1 - w['keep_r_swap'], word_lost_kill=1 - w['keep_r_kill'], swap_success=w['success_swap'])
    rows.append(r)
cols = []
for r in rows:
    for k in r:
        if k not in cols: cols.append(k)
with open(K + 'analysis/fix_results_all.csv', 'w', newline='') as fh:
    wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
    for r in rows: wr.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(len(rows), 'runs')
