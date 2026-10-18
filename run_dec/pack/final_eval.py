"""End-of-training evaluation, run on the GPU right after the last epoch.

Writes into the run directory, so `fetch_small.py` (no weights) brings back
everything needed for scoring on the Mac:

  dev_hyps.json      greedy hypotheses for the FULL dev set, in dev order
  final_eval.json    char chrF (invisible chars removed) on full dev, chrF++,
                     and the surface-form routing probe at root AND pattern
                     level (models with a routing memory only)

Uses best.pt (selected by dev translation loss), not the last epoch.
"""
import json
import os
import time
import unicodedata

import torch
from torch.nn.utils.rnn import pad_sequence


def strip_cf(s):
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


@torch.no_grad()
def decode_all(model, pairs, src_tok, tgt_tok, cfg, device, batch=64):
    import data as D
    pad, bos, eos = cfg["pad_id"], cfg["bos_id"], cfg["eos_id"]
    enc = []
    for src, _ in pairs:
        ids, starts = D.encode_words(src_tok, src, bos, eos, cfg["max_len"])
        enc.append((torch.tensor(ids), D.word_ids_from_starts(starts, len(ids))))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0]))
    hyps = [None] * len(enc)
    model.eval()
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        src = pad_sequence([enc[i][0] for i in idx], batch_first=True,
                           padding_value=pad).to(device)
        wid = pad_sequence([enc[i][1] for i in idx], batch_first=True,
                           padding_value=-1).to(device)
        ys = model.greedy_decode(src, cfg["max_len"], bos, eos, word_ids=wid)
        for row, i in zip(ys.tolist(), idx):
            toks = []
            for t in row[1:]:
                if t == eos:
                    break
                if t not in (pad, bos):
                    toks.append(t)
            hyps[i] = tgt_tok.decode(toks)
    return hyps


def run_final_eval(model, run_dir, dev_pairs, train_pairs, src_tok, tgt_tok,
                   cfg, device, label_entries=None):
    from sacrebleu.metrics import CHRF
    t0 = time.time()
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    hyps = decode_all(model, dev_pairs, src_tok, tgt_tok, cfg, device)
    refs = [t for _, t in dev_pairs]
    json.dump(hyps, open(os.path.join(run_dir, "dev_hyps.json"), "w"),
              ensure_ascii=False)
    res = {
        "best_epoch": ck.get("epoch"),
        "chrF_char_full_dev": CHRF(word_order=0).corpus_score(
            [strip_cf(h) for h in hyps], [[strip_cf(r) for r in refs]]).score,
        "chrF++_full_dev": CHRF(word_order=2).corpus_score(
            [strip_cf(h) for h in hyps], [[strip_cf(r) for r in refs]]).score,
        "n_dev": len(refs),
    }

    has_memory = any(getattr(l, "memory", None) is not None
                     for l in getattr(model, "enc_layers", []))
    if has_memory and label_entries is not None:
        from grounded import routing_consistency_types
        train_types = {w for s, _ in train_pairs for w in s.split()}
        for level in ("root", "pattern"):
            try:
                res[f"probe_{level}"] = routing_consistency_types(
                    model, dev_pairs, src_tok, label_entries, cfg, device,
                    train_types=train_types, level=level)
            except Exception as exc:          # never fail a run on a probe
                res[f"probe_{level}"] = {"error": str(exc)}
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(run_dir, "final_eval.json"), "w"), indent=1)
    print(f"final eval: chrF(char, full dev) {res['chrF_char_full_dev']:.2f}  "
          f"chrF++ {res['chrF++_full_dev']:.2f}  ({res['seconds']}s)", flush=True)
    for level in ("root", "pattern"):
        p = res.get(f"probe_{level}", {}).get("all_types")
        if p and p.get("consistency") is not None:
            print(f"  probe {level}: {p['consistency']:.3f} vs chance "
                  f"{p['permutation']:.3f}; unseen own-slot "
                  f"{res[f'probe_{level}'].get('unseen_slot_accuracy')}", flush=True)
    return res
