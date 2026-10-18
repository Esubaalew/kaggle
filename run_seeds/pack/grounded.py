"""Grounded Morphological Memory (GMM).

A drop-in replacement for cmm.CompositionalMorphMemory that changes two
structural things and fixes three places where the implemented CMM departs
from the written protocol (mmm_positioning.pdf, Eqs. 4-12).

WHAT IS NEW (the architectural contribution)
--------------------------------------------
1. Routing grounding.  The root/pattern labels supervise the memory's OWN
   routing distributions A^r, A^a -- not an external head whose output is
   discarded.  Slot k of M_V^r therefore *is* root class k.  The retrieved
   vector R^r = A^r M_V^r is on the computational path to the decoder, so
   the supervision installs persistent structure in weights that survive to
   inference, instead of only shaping the encoder state by gradient
   pressure.  Labels remain training-only and absent at inference (C3).

2. Word-level routing.  A Semitic root is a property of a word, not of a
   BPE piece.  Routing once per word (mean-pool the word's subword span,
   route, broadcast the retrieval back) makes the assignment well-posed.
   The current code instead routes per subword and patches the mismatch on
   the label side by copying a word's label onto every piece.

Both are orthogonal, giving the 2x2 ablation
    {subword, word} routing  x  {free, grounded}
in which the published CMM is the (subword, free) cell.

WHAT IS RESTORED (protocol fidelity)
------------------------------------
a. Untied keys and values.  cmm.py uses one `root_bank` as both key and
   value with no query projection:  softmax(h B^T) B.  The protocol Eqs.
   (4)-(5) specify Q = H W_Q and separate M_K (K x d_k), M_V (K x d).  With
   tied K/V the retrieved vector is pulled toward whichever slot is already
   closest to h in embedding space, which biases slots toward surface
   similarity rather than abstract roots.
b. Scalar gate.  Protocol Eq. (9) is a per-token scalar
   lambda_t = sigma(w^T [h_t ; r_t] + b).  cmm.py uses a d-dimensional
   elementwise gate, which makes the §4.6 "gate statistics by token class"
   probe ill-defined.  `scalar_gate=True` restores Eq. (9).
c. Explicit entropy sign.  §3.6 says "Minimizing L_ent sharpens each
   token's routing", and Eq. (12) adds +alpha * L_ent.  train.py computes
   `total = total - entropy_reg * ent`, i.e. it MAXIMIZES entropy and
   drives routing toward uniform -- the paper's own failure mode (i).
   `entropy_sign` makes the choice explicit and named; see grounding_loss.

Set `grounded=False, word_level=False, scalar_gate=False, tied_kv=True`
to recover the published CMM exactly.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def pool_words(h, word_ids, n_words):
    """Mean-pool token states into word slots.

    h         : (B, T, d)
    word_ids  : (B, T) long, word index within the sentence, -1 where the
                position is not part of a word (BOS/EOS/PAD).
    n_words   : int, max words per sentence in this batch (W).

    Returns (pooled, flat_index, valid) where pooled is (B*W, d).
    """
    b, t, d = h.shape
    valid = word_ids >= 0
    offs = torch.arange(b, device=h.device).unsqueeze(1) * n_words
    flat_index = (word_ids.clamp_min(0) + offs).reshape(-1)
    hf = h.reshape(b * t, d)
    vf = valid.reshape(-1)

    pooled = h.new_zeros(b * n_words, d)
    counts = h.new_zeros(b * n_words)
    pooled.index_add_(0, flat_index[vf], hf[vf])
    counts.index_add_(0, flat_index[vf], torch.ones_like(counts[:1]).expand(int(vf.sum())))
    pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
    return pooled, flat_index, valid


class GroundedMorphMemory(nn.Module):
    """Factorized persistent memory with optionally grounded routing.

    n_root_slots / n_pattern_slots: when `grounded=True` these MUST equal
    the label inventory sizes, because slot k is identified with class k.
    """

    def __init__(self, d_model, n_root_slots=256, n_pattern_slots=17,
                 d_key=None, with_binding=True, word_level=True,
                 grounded=True, scalar_gate=True, tied_kv=False,
                 dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_key = d_key or max(1, d_model // 4)
        self.n_root_slots = n_root_slots
        self.n_pattern_slots = n_pattern_slots
        self.with_binding = with_binding
        self.word_level = word_level
        self.grounded = grounded
        self.scalar_gate = scalar_gate
        self.tied_kv = tied_kv

        if tied_kv:
            self.root_bank = nn.Parameter(
                torch.randn(n_root_slots, d_model) / math.sqrt(d_model))
            self.pattern_bank = nn.Parameter(
                torch.randn(n_pattern_slots, d_model) / math.sqrt(d_model))
        else:
            self.wq_root = nn.Linear(d_model, self.d_key)
            self.wq_pat = nn.Linear(d_model, self.d_key)
            self.mk_root = nn.Parameter(torch.empty(n_root_slots, self.d_key))
            self.mv_root = nn.Parameter(torch.empty(n_root_slots, d_model))
            self.mk_pat = nn.Parameter(torch.empty(n_pattern_slots, self.d_key))
            self.mv_pat = nn.Parameter(torch.empty(n_pattern_slots, d_model))
            for p in (self.mk_root, self.mv_root, self.mk_pat, self.mv_pat):
                nn.init.xavier_uniform_(p)

        comp_dim = 3 * d_model if with_binding else 2 * d_model
        self.compose = nn.Linear(comp_dim, d_model)
        if scalar_gate:
            self.gate = nn.Linear(2 * d_model, 1)          # Eq. (9)
        else:
            self.gate_h = nn.Linear(d_model, d_model)
            self.gate_c = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def _route(self, x, which):
        """x: (N, d) -> (scores (N, K), retrieved (N, d))."""
        if self.tied_kv:
            bank = self.root_bank if which == "root" else self.pattern_bank
            logits = x @ bank.t() / math.sqrt(self.d_model)
            scores = F.softmax(logits, dim=-1)
            return scores, scores @ bank
        wq = self.wq_root if which == "root" else self.wq_pat
        mk = self.mk_root if which == "root" else self.mk_pat
        mv = self.mv_root if which == "root" else self.mv_pat
        logits = wq(x) @ mk.t() / math.sqrt(self.d_key)
        scores = F.softmax(logits, dim=-1)
        return scores, scores @ mv

    def forward(self, h, key_padding_mask=None, word_ids=None):
        b, t, d = h.shape

        if self.word_level:
            if word_ids is None:
                raise ValueError("word_level=True requires word_ids")
            n_words = max(int(word_ids.max().item()) + 1, 1)
            x, flat_index, valid = pool_words(h, word_ids, n_words)
        else:
            x = h.reshape(b * t, d)

        root_scores, r = self._route(x, "root")            # Eqs. (4)-(5)
        pat_scores, a = self._route(x + r, "pattern")      # Eqs. (6)-(7)

        comp = torch.cat([r, a, r * a], -1) if self.with_binding \
            else torch.cat([r, a], -1)
        c = self.compose(comp)                             # Eq. (8)

        if self.word_level:
            # broadcast word-level retrievals and routings back to subwords
            c = c[flat_index].reshape(b, t, d) * valid.unsqueeze(-1)
            root_scores_tok = root_scores[flat_index].reshape(b, t, -1)
            pat_scores_tok = pat_scores[flat_index].reshape(b, t, -1)
        else:
            c = c.reshape(b, t, d)
            root_scores_tok = root_scores.reshape(b, t, -1)
            pat_scores_tok = pat_scores.reshape(b, t, -1)

        if self.scalar_gate:
            lam = torch.sigmoid(self.gate(torch.cat([h, c], -1)))   # (b,t,1)
        else:
            lam = torch.sigmoid(self.gate_h(h) + self.gate_c(c))
        out = self.norm(h + self.dropout(lam * c))                  # Eq. (9)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        info = {"root_scores": root_scores_tok,
                "pattern_scores": pat_scores_tok,
                "gate": lam.detach(),
                "grounded": self.grounded}
        return out, info


def grounding_loss(routings, root_lab, pat_lab, root_weight=0.5,
                   pattern_weight=0.5, entropy_weight=0.0,
                   entropy_sign=+1.0, balance_weight=0.0,
                   ignore_index=-100, ignore_pattern_ids=()):
    """Cross-entropy on the memory's own routing distributions.

    routings : list of info dicts returned by GroundedMorphMemory.forward
    root_lab, pat_lab : (B, T) long, ignore_index where unlabelled.

    entropy_sign = +1 minimizes routing entropy (sharpening, as §3.6
    specifies); -1 reproduces train.py's current anti-collapse bonus, which
    maximizes entropy.  Pass the sign explicitly so the choice is recorded
    in the run config rather than implied by a minus in the trainer.

    Returns (total, terms).
    """
    terms = {}
    total = None
    if not routings:
        return None, terms

    if ignore_pattern_ids:
        pat_lab = pat_lab.clone()
        for pid in ignore_pattern_ids:
            pat_lab[pat_lab == pid] = ignore_index

    ce_r, ce_p, ents, bals = [], [], [], []
    for info in routings:
        if not info.get("grounded"):
            continue
        rs = info["root_scores"].clamp_min(EPS)
        ps = info["pattern_scores"].clamp_min(EPS)
        ce_r.append(F.nll_loss(rs.log().reshape(-1, rs.size(-1)),
                               root_lab.reshape(-1),
                               ignore_index=ignore_index))
        ce_p.append(F.nll_loss(ps.log().reshape(-1, ps.size(-1)),
                               pat_lab.reshape(-1),
                               ignore_index=ignore_index))
        for s in (rs, ps):
            ents.append(-(s * s.log()).sum(-1).mean())
            mean_assign = s.mean(dim=(0, 1))
            uni = torch.full_like(mean_assign, 1.0 / mean_assign.size(-1))
            bals.append((mean_assign * (mean_assign.log() - uni.log())).sum())

    if not ce_r:
        return None, terms
    l_root = torch.stack(ce_r).mean()
    l_pat = torch.stack(ce_p).mean()
    total = root_weight * l_root + pattern_weight * l_pat
    terms["route_root"] = l_root.detach()
    terms["route_pattern"] = l_pat.detach()
    if entropy_weight:
        ent = torch.stack(ents).mean()
        total = total + entropy_sign * entropy_weight * ent
        terms["entropy"] = ent.detach()
    if balance_weight:
        bal = torch.stack(bals).mean()
        total = total + balance_weight * bal
        terms["balance"] = bal.detach()
    return total, terms


@torch.no_grad()
def routing_consistency_types(model, pairs, tokenizer, label_entries, cfg,
                              device, layer=0, train_types=None,
                              min_types=2, n_permutations=200, seed=0,
                              level="root"):
    """Section 4.6 probe 1, measured over SURFACE FORMS, not tokens.

    The token-level `routing_consistency` below is inflated by repetition:
    every occurrence of a frequent word routes the same way, so a root whose
    tokens are mostly one surface form scores high with no abstraction at all,
    and the token-shuffling baseline does not control for it. Section 4.6
    defines the probe over "attested surface forms", which this implements:

      1. word slot  = argmax of the routing distribution averaged over the
                      word's BPE pieces (identical pieces for word-level
                      routing; a real average for subword routing);
      2. type slot  = majority slot over that word type's occurrences;
      3. root score = fraction of the root's types sharing its majority slot,
                      over roots with >= min_types distinct types;
      4. baseline   = same score with root labels permuted ACROSS TYPES,
                      preserving the type-level root marginal.

    If `train_types` is given, also reports the probe restricted to types
    UNSEEN in training whose root has a seen sibling, which is the
    mechanism-level version of the held-out paradigm test. For grounded
    models (slot k = class k) it additionally reports slot accuracy on those
    unseen types: does a new inflection route to its root's own slot?

    Returns a dict; values are None when there is not enough data.
    """
    import collections
    import random as _rnd
    from data import encode_words, word_ids_from_starts

    model.eval()
    occ = collections.defaultdict(list)          # word type -> [slot, ...]
    for src, tgt in pairs:
        ids, starts = encode_words(tokenizer, src, cfg["bos_id"],
                                   cfg["eos_id"], cfg["max_len"])
        n = len(ids)
        wid = word_ids_from_starts(starts, n)
        s = torch.tensor([ids], device=device)
        t_ = torch.tensor([[cfg["bos_id"], cfg["eos_id"]]], device=device)
        out = model(s, t_, wid.unsqueeze(0).to(device))
        key = f"{level}_scores"
        rs = [r for r in out.get("routings", []) if key in r]
        if not rs:
            return {"error": "model has no routing memory"}
        A = rs[layer][key][0]                   # (n, K)
        words = src.split()
        for k in range(len(starts)):
            m = (wid == k)
            if not m.any():
                continue
            occ[words[k]].append(int(A[m.to(A.device)].mean(0).argmax()))

    type_slot = {w: collections.Counter(v).most_common(1)[0][0]
                 for w, v in occ.items()}
    type_root = {w: int(label_entries.get(w, {}).get(level, -100))
                 for w in type_slot}
    type_root = {w: r for w, r in type_root.items() if r >= 1}

    def score(assign, types):
        by = collections.defaultdict(list)
        for w in types:
            by[assign[w]].append(type_slot[w])
        vals = [collections.Counter(v).most_common(1)[0][1] / len(v)
                for v in by.values() if len(v) >= min_types]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    def with_baseline(types):
        types = sorted(types)
        obs, n_roots = score(type_root, types)
        if obs is None:
            return {"consistency": None, "permutation": None, "n_roots": 0,
                    "n_types": len(types)}
        rng = _rnd.Random(seed)
        labs = [type_root[w] for w in types]
        perms = []
        for _ in range(n_permutations):
            rng.shuffle(labs)
            v, _ = score(dict(zip(types, labs)), types)
            if v is not None:
                perms.append(v)
        mean = sum(perms) / len(perms) if perms else None
        sd = ((sum((p - mean) ** 2 for p in perms) / (len(perms) - 1)) ** 0.5
              if len(perms) > 1 else None)
        # one-sided permutation p-value, with the +1 correction
        p = ((1 + sum(1 for q in perms if q >= obs)) / (1 + len(perms))
             if perms else None)
        slots = collections.Counter(type_slot[w] for w in types)
        top10 = sum(c for _, c in slots.most_common(10)) / max(1, len(types))
        return {"consistency": obs, "permutation": mean, "perm_sd": sd,
                "p_value": p, "n_roots": n_roots, "n_types": len(types),
                "slots_used": len(slots), "top10_slot_share": top10}

    res = {"all_types": with_baseline(type_root.keys())}

    if train_types is not None:
        seen_roots = {r for w, r in type_root.items() if w in train_types}
        unseen = [w for w, r in type_root.items()
                  if w not in train_types and r in seen_roots]
        # consistency of unseen forms WITH their seen siblings
        fam = [w for w, r in type_root.items()
               if r in {type_root[u] for u in unseen}]
        res["unseen_with_siblings"] = with_baseline(fam)
        res["n_unseen_types"] = len(unseen)
        if unseen:
            acc = sum(type_slot[w] == type_root[w] for w in unseen) / len(unseen)
            res["unseen_slot_accuracy"] = acc   # meaningful only if grounded
        else:
            res["unseen_slot_accuracy"] = None
    return res


@torch.no_grad()
def routing_consistency(model, loader, device, layer=0, level="root",
                        min_forms=2, top_n=200, n_permutations=100, seed=0):
    """§4.6 probe 1: do surface forms of one root share a routing slot?

    For each label class with >= min_forms distinct surface positions,
    computes the fraction of its tokens whose routing argmax equals the
    class's majority slot.  The permutation baseline reshuffles the
    class labels across tokens, preserving both marginals.

    Returns {"consistency": float, "permutation": float, "n_classes": int}.
    """
    import collections
    model.eval()
    key = "root_scores" if level == "root" else "pattern_scores"
    by_class = collections.defaultdict(list)

    for src, tgt, root_lab, pat_lab, word_ids in loader:
        src = src.to(device)
        lab = (root_lab if level == "root" else pat_lab).to(device)
        out = model(src, tgt[:, :-1].to(device), word_ids.to(device))
        routings = [r for r in out.get("routings", []) if key in r]
        if not routings:
            return {"consistency": float("nan"), "permutation": float("nan"),
                    "n_classes": 0}
        arg = routings[layer][key].argmax(-1)
        m = lab >= 1                      # skip <none> (0) and ignore (-100)
        for cls, slot in zip(lab[m].tolist(), arg[m].tolist()):
            by_class[cls].append(slot)

    classes = sorted(by_class, key=lambda c: -len(by_class[c]))[:top_n]
    classes = [c for c in classes if len(set(by_class[c])) >= 1
               and len(by_class[c]) >= min_forms]
    if not classes:
        return {"consistency": float("nan"), "permutation": float("nan"),
                "n_classes": 0}

    def score(mapping):
        vals = []
        for c in classes:
            slots = mapping[c]
            maj = collections.Counter(slots).most_common(1)[0][1]
            vals.append(maj / len(slots))
        return sum(vals) / len(vals)

    obs = score(by_class)

    import random as _rnd
    rng = _rnd.Random(seed)
    pool = [s for c in classes for s in by_class[c]]
    perms = []
    for _ in range(n_permutations):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        mapping, i = {}, 0
        for c in classes:
            n = len(by_class[c])
            mapping[c] = shuffled[i:i + n]
            i += n
        perms.append(score(mapping))
    return {"consistency": obs, "permutation": sum(perms) / len(perms),
            "n_classes": len(classes)}
