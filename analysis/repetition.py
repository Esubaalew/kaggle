import json,re,os,sys,numpy as np
from sacrebleu.metrics import CHRF
W=sys.argv[1]
CF=re.compile('[\u200b-\u200f\u2060\ufeff\u00ad]'); ns=lambda s: re.sub(r'\s+','',CF.sub('',s))
refs=[CF.sub('',l.rstrip('\n').split('\t')[1]) for l in open('drfn_clean/pack/data/age_dev.tsv',encoding='utf-8')]
R=sum(len(ns(r)) for r in refs)
def has_loop(s,n=3):
    w=s.split(); g=[tuple(w[i:i+n]) for i in range(len(w)-n+1)]; return len(g)!=len(set(g))
def dedup(s,n=3):
    # drop any word n-gram repeat: keep first occurrence, skip tokens that would recreate a seen n-gram
    w=s.split(); out=[]; seen=set()
    for t in w:
        if len(out)>=n-1 and tuple(out[-(n-1):]+[t]) in seen: continue
        out.append(t)
        if len(out)>=n: seen.add(tuple(out[-n:]))
    return ' '.join(out)
refloop=np.mean([has_loop(r) for r in refs])
sets={}
d=json.load(open(W+'/dev_hyps.json'))
for k,v in d.items(): sets['OLD '+k]=v
for dd in ['run_clean_geezrpa','run_clean_analyser','run_overnight','run_dec']:
    for r in sorted(os.listdir(dd+'/pack/runs')):
        p=f'{dd}/pack/runs/{r}/dev_hyps.json'
        if os.path.exists(p): sets[r]=json.load(open(p))
m=CHRF(word_order=0); rows=[]
for k,h in sets.items():
    h=[CF.sub('',x) for x in h]; hd=[dedup(x) for x in h]
    rows.append(dict(run=k, len_ratio=sum(len(ns(x)) for x in h)/R, loop_rate=float(np.mean([has_loop(x) for x in h])),
        chrF=m.corpus_score(h,[refs]).score, chrF_dedup=m.corpus_score(hd,[refs]).score,
        len_ratio_dedup=sum(len(ns(x)) for x in hd)/R))
print('ref loop rate %.3f'%refloop)
for r in rows: print(f"{r['run']:30s} len {r['len_ratio']:.3f} loops {r['loop_rate']:.3f} chrF {r['chrF']:.2f} -> dedup {r['chrF_dedup']:.2f} (len {r['len_ratio_dedup']:.3f})")
c=[r for r in rows if not r['run'].startswith('OLD')]
print('corr(len_ratio, chrF) clean runs: %.2f'%np.corrcoef([r['len_ratio'] for r in c],[r['chrF'] for r in c])[0,1])
print('sd chrF raw %.3f  dedup %.3f (clean runs)'%(np.std([r['chrF'] for r in c]),np.std([r['chrF_dedup'] for r in c])))
json.dump(dict(ref_loop_rate=refloop,rows=rows),open('analysis/repetition.json','w'),indent=1)
