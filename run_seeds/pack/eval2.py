"""Step 1 + step 2 on existing checkpoints (no training).

For every run directory containing best.pt:
  decode full dev with  beam5_nr4 (primary, fixed in advance), beam5 (no
  blocking) and beam1_nr4 (greedy + blocking, used by the slot tests);
  write hyps + chrF/chrF++/BLEU.
  gmm*:      slot swap + memory ablation      (slot_tests.py)
  cmm_spec:  memory ablation (ungrounded contrast)

    python eval2.py --ckpt-root /kaggle/input --out /kaggle/working/eval2 \
                    --shard 0 --n-shards 2
"""
import argparse
import glob
import json
import os
import time
import sys
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import data as D
from beam import decode_pairs
from model import ThesisTransformer
import slot_tests as ST

CONFIGS = {"beam5_nr4": dict(beam=5, alpha=1.0, no_repeat=4),
           "beam5": dict(beam=5, alpha=1.0, no_repeat=0),
           "beam1_nr4": dict(beam=1, alpha=1.0, no_repeat=4)}


def strip_cf(s):
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


def score(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    h = [strip_cf(x) for x in hyps]
    r = [[strip_cf(x) for x in refs]]
    return {"chrF_char": CHRF(word_order=0).corpus_score(h, r).score,
            "chrF++": CHRF(word_order=2).corpus_score(h, r).score,
            "BLEU": BLEU().corpus_score(h, r).score}


def find_runs(root):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "**", "best.pt"), recursive=True)):
        d = os.path.dirname(p)
        name = os.path.basename(d)
        if name in runs:
            continue
        if os.path.exists(os.path.join(d, "tokenizer.json")):
            runs[name] = d
    return runs


def label_source(name):
    return "analyser" if "_analyser_" in name else "geezrpa" if "_geezrpa_" in name else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="/kaggle/input")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="/kaggle/working/eval2")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-dev", type=int, default=0, help="smoke tests only")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dev_pairs, _ = D.load_pairs(os.path.join(args.data_dir, "age_dev.tsv"))
    train_pairs, _ = D.load_pairs(os.path.join(args.data_dir, "age_train.tsv"))
    if args.max_dev:
        dev_pairs = dev_pairs[:args.max_dev]
    refs = [t for _, t in dev_pairs]
    print("dev", len(dev_pairs), "train", len(train_pairs), flush=True)
    runs = find_runs(args.ckpt_root)
    names = sorted(runs)
    if args.only:
        names = [n for n in names if n in args.only.split(",")]
    names = [n for k, n in enumerate(names) if k % args.n_shards == args.shard]
    print("runs on this shard:", names, flush=True)
    for name in names:
        t0 = time.time()
        rd = runs[name]
        od = os.path.join(args.out, name)
        os.makedirs(od, exist_ok=True)
        ck = torch.load(os.path.join(rd, "best.pt"), map_location=device, weights_only=False)
        cfg = ck["cfg"]
        model = ThesisTransformer(cfg, cfg["n_root_classes"], cfg["n_pattern_classes"])
        model.load_state_dict(ck["model"])
        model.to(device).eval()
        tok = D.load_tokenizer(os.path.join(rd, "tokenizer.json"))
        res = {"run": name, "variant": cfg["variant"], "best_epoch": ck.get("epoch"),
               "source_dir": rd, "decode": {}}
        hyps_all = {}
        for cname, kw in CONFIGS.items():
            t1 = time.time()
            hyps = decode_pairs(model, dev_pairs, tok, tok, cfg, device, **kw)
            hyps_all[cname] = hyps
            json.dump(hyps, open(os.path.join(od, f"hyps_{cname}.json"), "w"), ensure_ascii=False)
            res["decode"][cname] = {**score(hyps, refs), "seconds": round(time.time() - t1)}
            print(f"{name:30s} {cname:10s} chrF {res['decode'][cname]['chrF_char']:.2f} "
                  f"BLEU {res['decode'][cname]['BLEU']:.2f} ({res['decode'][cname]['seconds']}s)",
                  flush=True)
        lab = label_source(name)
        has_mem = len(ST.memories(model)) > 0
        if has_mem and lab:
            cache = D.LabelCache(os.path.join(args.data_dir, f"labels_{lab}.json"))
            tmap = json.load(open(os.path.join(args.data_dir, f"labels_tgt_{lab}.json")))["map"]
            tmap = {w: int(k) for w, k in tmap.items()}
            res["ablation"] = ST.ablation(model, dev_pairs, train_pairs, tok, cfg, device,
                                          cache.entries, tmap)
            print("  ablation", {k: round(v["delta"], 4) for k, v in res["ablation"].items()},
                  flush=True)
            grounded = any(m.grounded and m.word_level for m in ST.memories(model))
            if grounded:
                def dec_fn(pairs):
                    return decode_pairs(model, pairs, tok, tok, cfg, device, batch=48, sort=False,
                                        **CONFIGS["beam1_nr4"])
                sw, ex = ST.slot_swap(model, dev_pairs, hyps_all["beam1_nr4"], tok, cfg,
                                      device, cache.entries, tmap, dec_fn)
                rows = sw.pop("rows")
                res["slot_swap"] = sw
                json.dump(rows, open(os.path.join(od, "slot_swap_rows.json"), "w"),
                          ensure_ascii=False)
                json.dump(ex, open(os.path.join(od, "slot_swap_examples.json"), "w"),
                          ensure_ascii=False, indent=1)
                print("  swap", {k: (round(v, 4) if isinstance(v, float) else v)
                                 for k, v in sw.items()}, flush=True)
        res["seconds"] = round(time.time() - t0)
        json.dump(res, open(os.path.join(od, "eval2.json"), "w"), indent=1, ensure_ascii=False)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
