"""Causal tests of the grounded root memory (step 2).

The probe shows that the memory's slot k collects the forms of root k.  These
tests ask whether the TRANSLATOR uses what the slot holds.

1. Slot swap (encoder).  Pick a dev sentence and a Ge'ez word w whose root r
   the model already translates (the greedy+blocking output contains an
   Amharic word of class r).  Force w's root routing to a one-hot on another
   root r' (pattern routing left exactly as it was) and decode again.
     success  = output now contains an Amharic word of class r'
     base     = same check when w is forced to its OWN argmax slot (control
                for one-hot forcing itself), for the same r'
     keep_r   = output still contains a class-r word
   If slots are used, success >> base and keep_r drops.  Amharic word classes
   come from labels_tgt_*.json (learned from TRAINING pairs only).
   Teacher-forced version: change in reference log-likelihood (swap minus
   self) on the reference tokens of class-r words vs all other tokens.

2. Memory ablation.  Remove the memory's contribution (c = 0) for every word
   and measure teacher-forced dev NLL per reference token, split by the
   training frequency of the Ge'ez word that carries the same root class in
   the same sentence (unseen form / 1-4 / 5-49 / 50+ / no linked word).
   Prediction: removing grounded memory hurts root-linked tokens, most of all
   for rare and unseen source forms.  cmm_spec (ungrounded memory) is the
   contrast.

Interventions monkeypatch GroundedMorphMemory.forward through a per-module
`override` dict; with override=None the original forward runs unchanged.
"""
import collections
import json
import random

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

import data as D
import grounded as G

_orig_forward = G.GroundedMorphMemory.forward


def _patched_forward(self, h, key_padding_mask=None, word_ids=None):
    ov = getattr(self, "override", None)
    if ov is None:
        return _orig_forward(self, h, key_padding_mask, word_ids)
    b, t, d = h.shape
    if self.word_level:
        n_words = max(int(word_ids.max().item()) + 1, 1)
        x, flat_index, valid = G.pool_words(h, word_ids, n_words)
    else:
        x = h.reshape(b * t, d)
    root_scores, r = self._route(x, "root")
    pat_scores, a = self._route(x + r, "pattern")      # pattern from ORIGINAL r
    if self.word_level and ov.get("root") is not None:
        force = ov["root"]                              # (B, W') long, -1 = none
        W2 = force.size(1)
        if W2 < n_words:
            force = F.pad(force, (0, n_words - W2), value=-1)
        force = force[:, :n_words].reshape(-1)
        m = force >= 0
        if m.any():
            onehot = F.one_hot(force.clamp_min(0), root_scores.size(-1)).to(root_scores.dtype)
            root_scores = torch.where(m.unsqueeze(-1), onehot, root_scores)
            r = root_scores @ self.mv_root
    comp = torch.cat([r, a, r * a], -1) if self.with_binding else torch.cat([r, a], -1)
    c = self.compose(comp)
    if self.word_level and ov.get("kill") is not None:
        kill = ov["kill"]
        W2 = kill.size(1)
        if W2 < n_words:
            kill = F.pad(kill, (0, n_words - W2), value=False)
        kill = kill[:, :n_words].reshape(-1)
        c = c.masked_fill(kill.unsqueeze(-1), 0.0)
    if self.word_level:
        c = c[flat_index].reshape(b, t, d) * valid.unsqueeze(-1)
        root_scores_tok = root_scores[flat_index].reshape(b, t, -1)
        pat_scores_tok = pat_scores[flat_index].reshape(b, t, -1)
    else:
        c = c.reshape(b, t, d)
        root_scores_tok = root_scores.reshape(b, t, -1)
        pat_scores_tok = pat_scores.reshape(b, t, -1)
    if ov.get("off"):
        c = torch.zeros_like(c)
    lam = torch.sigmoid(self.gate(torch.cat([h, c], -1))) if self.scalar_gate \
        else torch.sigmoid(self.gate_h(h) + self.gate_c(c))
    out = self.norm(h + self.dropout(lam * c))
    if key_padding_mask is not None:
        out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
    return out, {"root_scores": root_scores_tok, "pattern_scores": pat_scores_tok,
                 "gate": lam.detach(), "grounded": self.grounded}


G.GroundedMorphMemory.forward = _patched_forward


def memories(model):
    return [l.memory for l in model.enc_layers
            if isinstance(getattr(l, "memory", None), G.GroundedMorphMemory)]


def set_override(model, ov):
    for m in memories(model):
        m.override = ov


def _encode(pairs, tok, cfg):
    out = []
    for src, tgt in pairs:
        ids, starts = D.encode_words(tok, src, cfg["bos_id"], cfg["eos_id"], cfg["max_len"])
        tids, _ = D.encode_words(tok, tgt, cfg["bos_id"], cfg["eos_id"], cfg["max_len"])
        out.append((ids, starts, tids))
    return out


@torch.no_grad()
def routed_slots(model, enc_items, cfg, device, batch=64):
    """argmax root slot per word (last memory layer), list of lists."""
    res = [None] * len(enc_items)
    for b0 in range(0, len(enc_items), batch):
        chunk = enc_items[b0:b0 + batch]
        src = pad_sequence([torch.tensor(c[0]) for c in chunk], batch_first=True,
                           padding_value=cfg["pad_id"]).to(device)
        wid = pad_sequence([D.word_ids_from_starts(c[1], len(c[0])) for c in chunk],
                           batch_first=True, padding_value=-1).to(device)
        _, routings = model.encode(src, src == cfg["pad_id"], wid)
        rs = routings[-1]["root_scores"].argmax(-1).cpu()          # (B, T)
        gate = routings[-1]["gate"].squeeze(-1).cpu()
        for j, c in enumerate(chunk):
            res[b0 + j] = [(int(rs[j, p]), float(gate[j, p])) if p < len(c[0]) else (0, 0.0)
                           for p in c[1]]
    return res


@torch.no_grad()
def token_nll(model, enc_items, cfg, device, ovs=None, batch=48):
    """Per-reference-token NLL (list of lists, aligned with tids[1:]).
    ovs: optional list of per-item override builders: fn(B_index)->None; we pass
    full-batch override tensors built from per-item specs instead."""
    pad = cfg["pad_id"]
    res = [None] * len(enc_items)
    for b0 in range(0, len(enc_items), batch):
        chunk = enc_items[b0:b0 + batch]
        src = pad_sequence([torch.tensor(c[0]) for c in chunk], batch_first=True,
                           padding_value=pad).to(device)
        wid = pad_sequence([D.word_ids_from_starts(c[1], len(c[0])) for c in chunk],
                           batch_first=True, padding_value=-1).to(device)
        tgt = pad_sequence([torch.tensor(c[2]) for c in chunk], batch_first=True,
                           padding_value=pad).to(device)
        if ovs is not None:
            set_override(model, ovs(b0, len(chunk), device))
        out = model(src, tgt[:, :-1], word_ids=wid)
        set_override(model, None)
        lp = F.log_softmax(out["logits"].float(), -1)
        gold = tgt[:, 1:]
        nll = -lp.gather(-1, gold.unsqueeze(-1)).squeeze(-1)
        for j, c in enumerate(chunk):
            res[b0 + j] = nll[j, :len(c[2]) - 1].cpu().tolist()
    return res


def _tgt_word_spans(tok, text):
    """[(word, start, end)] over ids WITHOUT bos, i.e. aligned to tids[1:]."""
    spans, pos = [], 0
    for w in text.split():
        n = len(tok.encode(w).ids)
        spans.append((w, pos, pos + n))
        pos += n
    return spans


def _classes_in(text, tmap):
    return {tmap[w] for w in text.split() if w in tmap}


@torch.no_grad()
def slot_swap(model, dev_pairs, base_hyps, tok, cfg, device, entries, tmap,
              decode_fn, n_swaps=2, seed=0, max_items=700):
    rng = random.Random(seed)
    T = collections.defaultdict(set)
    for w, k in tmap.items():
        T[int(k)].add(w)
    pool = sorted(k for k, ws in T.items() if len(ws) >= 3)
    enc = _encode(dev_pairs, tok, cfg)
    routed = routed_slots(model, enc, cfg, device)
    items = []
    for i, ((src, ref), hyp) in enumerate(zip(dev_pairs, base_hyps)):
        hyp_c, ref_c = _classes_in(hyp, tmap), _classes_in(ref, tmap)
        cands = []
        for wi, w in enumerate(src.split()):
            e = entries.get(w)
            if not e:
                continue
            r = int(e.get("root", 0))
            if r <= 0 or r not in T or wi >= len(routed[i]):
                continue
            if r in ref_c and r in hyp_c:
                cands.append((wi, w, r))
        if not cands:
            continue
        wi, w, r = rng.choice(cands)
        excl = hyp_c | ref_c | {r}
        alts = [k for k in pool if k not in excl]
        for rp in rng.sample(alts, min(n_swaps, len(alts))):
            items.append({"i": i, "wi": wi, "word": w, "r": r, "rp": rp,
                          "routed": routed[i][wi][0], "gate": routed[i][wi][1]})
    rng.shuffle(items)
    items = items[:max_items]

    def run(cond):
        """Decode every item's sentence under a condition."""
        hyps = []
        for b0 in range(0, len(items), 48):
            chunk = items[b0:b0 + 48]
            Wmax = max(len(dev_pairs[it["i"]][0].split()) for it in chunk)
            force = torch.full((len(chunk), Wmax), -1, dtype=torch.long)
            kill = torch.zeros((len(chunk), Wmax), dtype=torch.bool)
            for j, it in enumerate(chunk):
                if cond == "swap":
                    force[j, it["wi"]] = it["rp"]
                elif cond == "self":
                    force[j, it["wi"]] = it["routed"]
                elif cond == "kill":
                    kill[j, it["wi"]] = True
            set_override(model, {"root": force.to(device), "kill": kill.to(device)})
            hyps += decode_fn([dev_pairs[it["i"]] for it in chunk])
            set_override(model, None)
        return hyps

    H = {c: run(c) for c in ("self", "swap", "kill")}

    def has(text, k):
        return any(tmap.get(w) == k for w in text.split())

    n = len(items)
    rows = []
    for j, it in enumerate(items):
        rows.append({**it,
                     "succ_swap": has(H["swap"][j], it["rp"]),
                     "succ_self": has(H["self"][j], it["rp"]),
                     "keep_swap": has(H["swap"][j], it["r"]),
                     "keep_self": has(H["self"][j], it["r"]),
                     "keep_kill": has(H["kill"][j], it["r"])})
    mean = lambda key: sum(r[key] for r in rows) / max(n, 1)

    # teacher-forced: reference log-likelihood change, class-r tokens vs others
    sub = [dev_pairs[it["i"]] for it in items]
    enc_items = _encode(sub, tok, cfg)

    def ov_builder(cond):
        def f(b0, nb, device):
            chunk = items[b0:b0 + nb]
            Wmax = max(len(dev_pairs[it["i"]][0].split()) for it in chunk)
            force = torch.full((nb, Wmax), -1, dtype=torch.long)
            for j, it in enumerate(chunk):
                force[j, it["wi"]] = it["rp"] if cond == "swap" else it["routed"]
            return {"root": force.to(device)}
        return f
    nll_self = token_nll(model, enc_items, cfg, device, ov_builder("self"))
    nll_swap = token_nll(model, enc_items, cfg, device, ov_builder("swap"))
    d_root, d_other = [], []
    for j, it in enumerate(items):
        ref = dev_pairs[it["i"]][1]
        for w, s, e in _tgt_word_spans(tok, ref):
            for p in range(s, min(e, len(nll_self[j]))):
                (d_root if tmap.get(w) == it["r"] else d_other).append(
                    nll_swap[j][p] - nll_self[j][p])
    examples = []
    for j, it in enumerate(items[:40]):
        src, ref = dev_pairs[it["i"]]
        examples.append({"src": src, "word": it["word"], "r": it["r"], "rp": it["rp"],
                         "ref": ref, "hyp_self": H["self"][j], "hyp_swap": H["swap"][j],
                         "hyp_kill": H["kill"][j]})
    avg = lambda v: sum(v) / len(v) if v else None
    return {
        "n_items": n,
        "routed_to_own_root": mean_bool([it["routed"] == it["r"] for it in items]),
        "mean_gate": avg([it["gate"] for it in items]),
        "success_swap": mean("succ_swap"), "success_self": mean("succ_self"),
        "keep_r_self": mean("keep_self"), "keep_r_swap": mean("keep_swap"),
        "keep_r_kill": mean("keep_kill"),
        "tf_delta_nll_root_tokens": avg(d_root), "n_root_tokens": len(d_root),
        "tf_delta_nll_other_tokens": avg(d_other), "n_other_tokens": len(d_other),
        "rows": rows,
    }, examples


def mean_bool(v):
    return sum(bool(x) for x in v) / max(len(v), 1)


def _off_builder(b0, nb, device):
    return {"off": True}


@torch.no_grad()
def ablation(model, dev_pairs, train_pairs, tok, cfg, device, entries, tmap):
    """Teacher-forced NLL with memory on vs off, bucketed by source frequency."""
    freq = collections.Counter(w for s, _ in train_pairs for w in s.split())
    enc = _encode(dev_pairs, tok, cfg)
    on = token_nll(model, enc, cfg, device)
    off = token_nll(model, enc, cfg, device, _off_builder)

    def bucket(f):
        return "unseen" if f == 0 else "1-4" if f < 5 else "5-49" if f < 50 else "50+"
    acc = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for j, (src, ref) in enumerate(dev_pairs):
        root_freq = {}
        for w in src.split():
            e = entries.get(w)
            r = int(e.get("root", 0)) if e else 0
            if r > 0:
                root_freq[r] = min(root_freq.get(r, 10 ** 9), freq.get(w, 0))
        for w, s, e in _tgt_word_spans(tok, ref):
            k = tmap.get(w)
            b = bucket(root_freq[k]) if k in root_freq else "unlinked"
            for p in range(s, min(e, len(on[j]))):
                a = acc[b]
                a[0] += on[j][p]
                a[1] += off[j][p]
                a[2] += 1
    out = {}
    for b, (s_on, s_off, n) in acc.items():
        out[b] = {"n_tokens": n, "nll_on": s_on / n, "nll_off": s_off / n,
                  "delta": (s_off - s_on) / n}
    tot_on = sum(v[0] for v in acc.values())
    tot_off = sum(v[1] for v in acc.values())
    ntot = sum(v[2] for v in acc.values())
    out["all"] = {"n_tokens": ntot, "nll_on": tot_on / ntot, "nll_off": tot_off / ntot,
                  "delta": (tot_off - tot_on) / ntot}
    return out
