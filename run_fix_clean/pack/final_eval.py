"""End-of-training evaluation for drfn_fix (runs on GPU right after the last epoch).

Computes and saves in a single pass from `best.pt`:
  - `dev_hyps_beam5_nr4.json` & `beam5_nr4` metrics (chrF, chrF++, BLEU)
  - `dev_hyps.json` & greedy metrics (for continuity with earlier tables)
  - `probe_root` & `probe_pattern` (surface-form slot consistency + unseen-word
    own-slot accuracy + §4.6 Probe 2 gate statistics on rooted vs rootless words)
  - `ablation` & `slot_swap` when a target root map (`labels_tgt_*.json`) is present
"""
import json
import os
import time
import unicodedata

import torch
from torch.nn.utils.rnn import pad_sequence

from beam import decode_pairs
import slot_tests as ST


def strip_cf(s):
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


def score_corpus(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    h = [strip_cf(x) for x in hyps]
    r = [[strip_cf(x) for x in refs]]
    return {
        "chrF_char": CHRF(word_order=0).corpus_score(h, r).score,
        "chrF++": CHRF(word_order=2).corpus_score(h, r).score,
        "BLEU": BLEU().corpus_score(h, r).score,
    }


@torch.no_grad()
def decode_greedy(model, pairs, src_tok, tgt_tok, cfg, device, batch=64):
    import data as D
    pad, bos, eos = cfg["pad_id"], cfg["bos_id"], cfg["max_len"] and cfg["eos_id"]
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
                   cfg, device, label_entries=None, tgt_label_path=None,
                   smoke=False):
    t0 = time.time()
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device,
                    weights_only=False)
    model.load_state_dict(ck["model"])
    model.to(device).eval()

    refs = [t for _, t in dev_pairs]
    # 1. Greedy decode (continuity)
    hyps_greedy = decode_greedy(model, dev_pairs, src_tok, tgt_tok, cfg, device)
    json.dump(hyps_greedy, open(os.path.join(run_dir, "dev_hyps.json"), "w"),
              ensure_ascii=False)
    sc_greedy = score_corpus(hyps_greedy, refs)

    # 2. Primary decode: Beam 5 + no-repeat 4-gram
    hyps_beam5 = decode_pairs(model, dev_pairs, src_tok, tgt_tok, cfg, device,
                              beam=5, alpha=1.0, no_repeat=4)
    json.dump(hyps_beam5, open(os.path.join(run_dir, "dev_hyps_beam5_nr4.json"), "w"),
              ensure_ascii=False)
    sc_beam5 = score_corpus(hyps_beam5, refs)

    res = {
        "best_epoch": ck.get("epoch"),
        "chrF_char_full_dev": sc_greedy["chrF_char"],
        "chrF++_full_dev": sc_greedy["chrF++"],
        "BLEU_greedy_full_dev": sc_greedy["BLEU"],
        "beam5_nr4": sc_beam5,
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
                    train_types=train_types, level=level,
                    n_permutations=20 if smoke else 200)
            except Exception as exc:
                res[f"probe_{level}"] = {"error": str(exc)}

        if tgt_label_path and os.path.exists(tgt_label_path):
            try:
                tmap = {w: int(k) for w, k in json.load(
                    open(tgt_label_path, encoding="utf-8"))["map"].items()}
                res["ablation"] = ST.ablation(
                    model, dev_pairs, train_pairs, src_tok, cfg, device,
                    label_entries, tmap)
                grounded = any(m.grounded and m.word_level for m in ST.memories(model))
                if grounded:
                    hyps_b1 = decode_pairs(model, dev_pairs, src_tok, tgt_tok,
                                           cfg, device, beam=1, alpha=1.0, no_repeat=4)
                    def dec_fn(pairs):
                        return decode_pairs(model, pairs, src_tok, tgt_tok, cfg,
                                            device, batch=48, sort=False,
                                            beam=1, alpha=1.0, no_repeat=4)
                    sw, ex = ST.slot_swap(
                        model, dev_pairs, hyps_b1, src_tok, cfg, device,
                        label_entries, tmap, dec_fn,
                        max_items=40 if smoke else 600)
                    rows = sw.pop("rows", [])
                    res["slot_swap"] = sw
                    json.dump(rows, open(os.path.join(run_dir, "slot_swap_rows.json"), "w"),
                              ensure_ascii=False)
                    json.dump(ex, open(os.path.join(run_dir, "slot_swap_examples.json"), "w"),
                              ensure_ascii=False, indent=1)
            except Exception as exc:
                res["causal_error"] = str(exc)

    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(run_dir, "final_eval.json"), "w"), indent=1)
    print(f"final eval: beam5_nr4 chrF {sc_beam5['chrF_char']:.2f} "
          f"chrF++ {sc_beam5['chrF++']:.2f} BLEU {sc_beam5['BLEU']:.2f} "
          f"(greedy chrF {sc_greedy['chrF_char']:.2f}) ({res['seconds']}s)",
          flush=True)
    for level in ("root", "pattern"):
        p = res.get(f"probe_{level}", {}).get("all_types")
        if p and p.get("consistency") is not None:
            print(f"  probe {level}: {p['consistency']:.3f} vs chance "
                  f"{p['permutation']:.3f}; unseen own-slot "
                  f"{res[f'probe_{level}'].get('unseen_slot_accuracy')}", flush=True)
    gp = res.get("probe_root", {}).get("gate_probe")
    if gp:
        print(f"  gate probe: rooted={gp['mean_gate_rooted']:.3f} "
              f"(n={gp['n_rooted_tokens']}) vs rootless={gp['mean_gate_rootless']:.3f} "
              f"(n={gp['n_rootless_tokens']})", flush=True)
    return res
