"""Root-routing intervention relative to cached labels, not independent gold detection.
Memory-off is reported separately from zero-correct routing. No retraining.
"""
import argparse, hashlib, json, random
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
import data as D
from model import build_model
from beam import beam_decode
from final_eval import score_corpus


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--run-dir",type=Path,required=True); ap.add_argument("--data-dir",type=Path,required=True)
    ap.add_argument("--limit",type=int,default=512);args=ap.parse_args()
    torch.set_num_threads(2)
    ck=torch.load(args.run_dir / "best.pt",map_location="cpu",weights_only=False);cfg=ck["cfg"]
    labels=json.loads((args.data_dir / "labels_analyser_wide.json").read_text()); entries=labels["entries"]
    model=build_model({k:v for k,v in cfg.items() if k != "target_params"},len(labels["root_classes"]),len(labels["pattern_classes"]),target_params=None)
    model.load_state_dict(ck["model"]);del ck
    device="cuda" if torch.cuda.is_available() else "cpu";model.to(device).eval()
    tok=Tokenizer.from_file(str(args.run_dir / "tokenizer.json"))
    pairs,_=D.load_pairs(args.data_dir / "age_dev.tsv")
    rng=random.Random(20261017);ids=sorted(rng.sample(range(len(pairs)),min(args.limit,len(pairs))));pairs=[pairs[i] for i in ids]
    mems=[x for x in model.modules() if hasattr(x,"override")]
    report={"interpretation":"controlled agreement with cached labels; not independent gold accuracy", "sample_indices":ids,"conditions":{}}
    for condition in ["natural","off",0.0,.4,.8,.95,1.0]:
        hyps=[];n_forced=0;n_correct=0
        for start in range(0,len(pairs),32):
            batch=pairs[start:start+32];enc=[];wids=[];words=[]
            for src,_ in batch:
                ii,ss=D.encode_words(tok,src,cfg["bos_id"],cfg["eos_id"],cfg["max_len"])
                enc.append(torch.tensor(ii));wids.append(D.word_ids_from_starts(ss,len(ii)));words.append(src.split())
            src=pad_sequence(enc,batch_first=True,padding_value=cfg["pad_id"]).to(device)
            wid=pad_sequence(wids,batch_first=True,padding_value=-1).to(device)
            if condition=="natural":override=None
            elif condition=="off":override={"off":True}
            else:
                forced=torch.full((len(batch),max(map(len,words))),-1,dtype=torch.long,device=device)
                for bi,ww in enumerate(words):
                    for wi,w in enumerate(ww):
                        root=entries.get(w,{}).get("root",-2)
                        if root<1:continue
                        seed=int.from_bytes(hashlib.sha256(f"20261017/{start+bi}/{wi}".encode()).digest()[:8],"big")
                        r=random.Random(seed);correct=r.random()<condition
                        wrong=r.randrange(1,len(labels["root_classes"])-1)
                        if wrong>=root:wrong+=1
                        forced[bi,wi]=root if correct else wrong
                        n_forced+=1;n_correct+=correct
                override={"root":forced}
            for m in mems:m.override=override
            output=beam_decode(model,src,wid,cfg["bos_id"],cfg["eos_id"],cfg["pad_id"],max_len=cfg["max_len"])
            hyps += [tok.decode(seq) for seq in output]
        score=score_corpus(hyps,[t for _,t in pairs]);score.update(n_forced=n_forced,agreement=n_correct/max(n_forced,1))
        report["conditions"][str(condition)]=score; print(condition,score,flush=True)
    for m in mems:m.override=None
    (args.run_dir / "routing_ablation.json").write_text(json.dumps(report,indent=2))


if __name__=="__main__":main()
