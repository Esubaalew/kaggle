"""Grounded Morphological Memory (GMM) — Fixed (2026-10-14).

Fixes three architectural bottlenecks in the original GroundedMorphMemory:
1. Null-Safe Gate Grounding (Eq. 9 + §4.6 Probe 2):
   - Masks Slot 0 (`<none>`) out of the root routing softmax (`logits[:, 0] = -1e9`)
     so `root_scores` distributes probability purely over real in-bank roots `1..K`.
   - Supervises the scalar gate `lambda_t = sigma(w_g^T [h_t ; c_t] + b_g)` with
     Binary Cross-Entropy (`BCE(lambda_t, root_lab >= 1)`) on all word positions
     (`root_lab >= 0`), so `lambda_t -> 1` on in-bank rooted words and
     `lambda_t -> 0` on rootless / out-of-inventory words (`ውስተ`, `ከመ`, `እግዚአብሔር`,
     punctuation, rare unindexed roots). This eliminates the 48% false-root collision
     rate and directly realizes §4.6 Probe 2 (gate separation by token class).
   - Masks entropy sharpening (`+0.01 * H`) strictly to positions with `root_lab >= 1`.

2. Value & Composition Grounding (`L_val`, zero extra parameters):
   - Previously, `L_ground = CE(alpha^r, y^root)` only differentiated into the
     routing keys `M_K` and `W_Q`, leaving `d L_ground / d M_V = 0` and
     `d L_ground / d W_compose = 0`.
   - We add a parameter-free value alignment loss on the composed word vector `c`:
     `val_logits_r = (c @ mv_root.t()) / sqrt(d)` and `val_logits_p = (c @ mv_pat.t()) / sqrt(d)`,
     supervised by `CE(val_logits_r, y^root)` and `CE(val_logits_p, y^pat)` on `y >= 1`.
     This directly trains `M_V^r`, `M_V^p`, and `W_compose` so that the vector
     injected into the encoder residual stream encodes the word's root and pattern.

3. Pre-norm Residual Stream (`h + dropout(lam * c_norm)`) with Exact Bypass:
   - LayerNorm (`self.norm`) is applied to `c_word = self.compose(comp)` BEFORE
     applying `valid` (BOS/EOS/PAD), `kill`, or `off` masks, and both `c_norm`
     and `lam` are masked by `active_mask`. Thus, at non-word positions (BOS/EOS/PAD)
     and during `kill` or `off` interventions, `lam * c_norm` is identically zero
     (`out == h` bit-for-bit, with no LayerNorm bias leakage).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


def pool_words(h, word_ids, n_words):
    """Mean-pool token states into word slots."""
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
    """Factorized persistent memory with grounded routing, value alignment, and gate supervision."""

    def __init__(self, d_model, n_root_slots=256, n_pattern_slots=17,
                 d_key=None, with_binding=True, word_level=True,
                 grounded=True, scalar_gate=True, tied_kv=False,
                 dropout=0.1, pre_norm_residual=None):
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
        self.pre_norm_residual = word_level if pre_norm_residual is None else pre_norm_residual

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
            with torch.no_grad():
                self.mv_root[0].zero_()
                self.mv_pat[0].zero_()

        comp_dim = 3 * d_model if with_binding else 2 * d_model
        self.compose = nn.Linear(comp_dim, d_model)
        if self.pre_norm_residual:
            nn.init.xavier_uniform_(self.compose.weight, gain=0.1)
            nn.init.zeros_(self.compose.bias)
        if scalar_gate:
            self.gate = nn.Linear(2 * d_model, 1)          # Eq. (9)
            nn.init.zeros_(self.gate.bias)
        else:
            self.gate_h = nn.Linear(d_model, d_model)
            self.gate_c = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.override = None

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
        if self.grounded and logits.size(-1) > 1:
            logits = logits.clone()
            logits[:, 0] = -1e9
        scores = F.softmax(logits, dim=-1)
        return scores, scores @ mv

    def forward(self, h, key_padding_mask=None, word_ids=None):
        b, t, d = h.shape
        ov = self.override

        if self.word_level:
            if word_ids is None:
                raise ValueError("word_level=True requires word_ids")
            n_words = max(int(word_ids.max().item()) + 1, 1)
            x, flat_index, valid = pool_words(h, word_ids, n_words)
        else:
            x = h.reshape(b * t, d)
            valid = torch.ones((b, t), dtype=torch.bool, device=h.device)
            if key_padding_mask is not None:
                valid = valid & (~key_padding_mask)

        root_scores, r = self._route(x, "root")            # Eqs. (4)-(5)
        pat_scores, a = self._route(x + r, "pattern")      # Eqs. (6)-(7)

        if self.word_level and ov is not None and ov.get("root") is not None:
            force = ov["root"]
            W2 = force.size(1)
            if W2 < n_words:
                force = F.pad(force, (0, n_words - W2), value=-1)
            force = force[:, :n_words].reshape(-1)
            m = force >= 0
            if m.any():
                onehot = F.one_hot(force.clamp_min(0), root_scores.size(-1)).to(root_scores.dtype)
                root_scores = torch.where(m.unsqueeze(-1), onehot, root_scores)
                r = root_scores @ self.mv_root

        comp = torch.cat([r, a, r * a], -1) if self.with_binding \
            else torch.cat([r, a], -1)
        c_word = self.compose(comp)                        # Eq. (8)

        # Normalize BEFORE applying kill/off/valid masks so LayerNorm.bias
        # never leaks into killed words, `off` ablations, or BOS/EOS/PAD.
        if self.pre_norm_residual:
            c_word = self.norm(c_word)

        if self.word_level and ov is not None and ov.get("kill") is not None:
            kill = ov["kill"]
            W2 = kill.size(1)
            if W2 < n_words:
                kill = F.pad(kill, (0, n_words - W2), value=False)
            kill_flat = kill[:, :n_words].reshape(-1)
            c_word = c_word.masked_fill(kill_flat.unsqueeze(-1), 0.0)
            active_tok = valid & (~kill_flat[flat_index].reshape(b, t))
        else:
            active_tok = valid

        if ov is not None and ov.get("off"):
            active_tok = torch.zeros_like(active_tok)

        if self.word_level:
            c_norm = c_word[flat_index].reshape(b, t, d) * active_tok.unsqueeze(-1)
            root_scores_tok = root_scores[flat_index].reshape(b, t, -1)
            pat_scores_tok = pat_scores[flat_index].reshape(b, t, -1)
        else:
            c_norm = c_word.reshape(b, t, d) * active_tok.unsqueeze(-1)
            root_scores_tok = root_scores.reshape(b, t, -1)
            pat_scores_tok = pat_scores.reshape(b, t, -1)

        if self.scalar_gate:
            lam_raw = torch.sigmoid(self.gate(torch.cat([h, c_norm], -1)))   # (b,t,1)
        else:
            lam_raw = torch.sigmoid(self.gate_h(h) + self.gate_c(c_norm))
        lam = lam_raw * active_tok.unsqueeze(-1).to(lam_raw.dtype)

        if self.pre_norm_residual:
            out = h + self.dropout(lam * c_norm)
        else:
            out = self.norm(h + self.dropout(lam * c_norm))
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        info = {
            "root_scores": root_scores_tok,
            "pattern_scores": pat_scores_tok,
            "c_norm": c_norm,
            "mv_root": None if self.tied_kv else self.mv_root,
            "mv_pat": None if self.tied_kv else self.mv_pat,
            "gate": lam.detach(),
            "gate_live": lam_raw.squeeze(-1) if self.scalar_gate else None,
            "grounded": self.grounded,
        }
        return out, info


def grounding_loss(routings, root_lab, pat_lab, root_weight=0.25,
                   pattern_weight=0.25, val_weight=0.25, gate_weight=0.2,
                   entropy_weight=0.0, entropy_sign=+1.0, balance_weight=0.0,
                   ignore_index=-100, ignore_pattern_ids=()):
    """Three-part Grounded Morphological Memory loss."""
    terms = {}
    if not routings:
        return None, terms

    if ignore_pattern_ids:
        pat_lab = pat_lab.clone()
        for pid in ignore_pattern_ids:
            pat_lab[pat_lab == pid] = ignore_index

    root_pos = root_lab.masked_fill(root_lab <= 0, ignore_index)
    pat_pos = pat_lab.masked_fill(pat_lab <= 0, ignore_index)
    valid_word = root_lab >= 0
    has_root_target = (root_lab >= 1).float()

    ce_r, ce_p, val_r, val_p, gate_bces, ents, bals = [], [], [], [], [], [], []
    for info in routings:
        if not info.get("grounded"):
            continue
        rs = info["root_scores"].clamp_min(EPS)
        ps = info["pattern_scores"].clamp_min(EPS)
        ce_r.append(F.nll_loss(rs.log().reshape(-1, rs.size(-1)),
                               root_pos.reshape(-1),
                               ignore_index=ignore_index))
        ce_p.append(F.nll_loss(ps.log().reshape(-1, ps.size(-1)),
                               pat_pos.reshape(-1),
                               ignore_index=ignore_index))

        cn = info.get("c_norm")
        mvr = info.get("mv_root")
        mvp = info.get("mv_pat")
        if cn is not None and mvr is not None and val_weight > 0:
            d_scale = math.sqrt(cn.size(-1))
            vr_logits = (cn @ mvr.t()) / d_scale
            vp_logits = (cn @ mvp.t()) / d_scale
            if vr_logits.size(-1) > 1:
                vr_logits = vr_logits.clone()
                vr_logits[..., 0] = -1e9
            if vp_logits.size(-1) > 1:
                vp_logits = vp_logits.clone()
                vp_logits[..., 0] = -1e9
            val_r.append(F.cross_entropy(vr_logits.reshape(-1, vr_logits.size(-1)),
                                         root_pos.reshape(-1),
                                         ignore_index=ignore_index))
            val_p.append(F.cross_entropy(vp_logits.reshape(-1, vp_logits.size(-1)),
                                         pat_pos.reshape(-1),
                                         ignore_index=ignore_index))

        gl = info.get("gate_live")
        if gl is not None and gate_weight > 0 and valid_word.any():
            gate_bces.append(F.binary_cross_entropy(
                gl[valid_word].clamp(1e-6, 1.0 - 1e-6),
                has_root_target[valid_word]))

        rm = root_pos != ignore_index
        pm = pat_pos != ignore_index
        if rm.any():
            s_r = rs[rm]
            ents.append(-(s_r * s_r.log()).sum(-1).mean())
        if pm.any():
            s_p = ps[pm]
            ents.append(-(s_p * s_p.log()).sum(-1).mean())
        for s in (rs, ps):
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
    if val_r:
        lv_r = torch.stack(val_r).mean()
        lv_p = torch.stack(val_p).mean()
        total = total + val_weight * (lv_r + lv_p)
        terms["val_root"] = lv_r.detach()
        terms["val_pat"] = lv_p.detach()
    if gate_bces:
        lg = torch.stack(gate_bces).mean()
        total = total + gate_weight * lg
        terms["gate_bce"] = lg.detach()
    if entropy_weight and ents:
        ent = torch.stack(ents).mean()
        total = total + entropy_sign * entropy_weight * ent
        terms["entropy"] = ent.detach()
    if balance_weight and bals:
        bal = torch.stack(bals).mean()
        total = total + balance_weight * bal
        terms["balance"] = bal.detach()
    return total, terms


@torch.no_grad()
def routing_consistency_types(model, pairs, tokenizer, label_entries, cfg,
                              device, layer=0, train_types=None,
                              min_types=2, n_permutations=200, seed=0,
                              level="root"):
    """Section 4.6 probe 1 (surface-form slot consistency) + Probe 2 (gate by token class)."""
    import collections
    import random as _rnd
    from data import encode_words, word_ids_from_starts

    model.eval()
    occ = collections.defaultdict(list)
    gate_rooted, gate_rootless = [], []
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
        A = rs[layer][key][0]
        G_t = rs[layer]["gate"][0].squeeze(-1)
        words = src.split()
        for k in range(len(starts)):
            m = (wid == k)
            if not m.any():
                continue
            w_str = words[k]
            occ[w_str].append(int(A[m.to(A.device)].mean(0).argmax()))
            if level == "root" and G_t.dim() == 1:
                g_val = float(G_t[m.to(G_t.device)].mean().item())
                r_id = int(label_entries.get(w_str, {}).get("root", 0))
                if r_id >= 1:
                    gate_rooted.append(g_val)
                else:
                    gate_rootless.append(g_val)

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
        p = ((1 + sum(1 for q in perms if q >= obs)) / (1 + len(perms))
             if perms else None)
        slots = collections.Counter(type_slot[w] for w in types)
        top10 = sum(c for _, c in slots.most_common(10)) / max(1, len(types))
        return {"consistency": obs, "permutation": mean, "perm_sd": sd,
                "p_value": p, "n_roots": n_roots, "n_types": len(types),
                "slots_used": len(slots), "top10_slot_share": top10}

    res = {"all_types": with_baseline(type_root.keys())}
    if gate_rooted or gate_rootless:
        res["gate_probe"] = {
            "mean_gate_rooted": sum(gate_rooted) / max(len(gate_rooted), 1),
            "n_rooted_tokens": len(gate_rooted),
            "mean_gate_rootless": sum(gate_rootless) / max(len(gate_rootless), 1),
            "n_rootless_tokens": len(gate_rootless),
        }

    if train_types is not None:
        seen_roots = {r for w, r in type_root.items() if w in train_types}
        unseen = [w for w, r in type_root.items()
                  if w not in train_types and r in seen_roots]
        fam = [w for w, r in type_root.items()
               if r in {type_root[u] for u in unseen}]
        res["unseen_with_siblings"] = with_baseline(fam)
        res["n_unseen_types"] = len(unseen)
        if unseen:
            acc = sum(type_slot[w] == type_root[w] for w in unseen) / len(unseen)
            res["unseen_slot_accuracy"] = acc
        else:
            res["unseen_slot_accuracy"] = None
    return res


@torch.no_grad()
def routing_consistency(model, loader, device, layer=0, level="root",
                        min_forms=2, top_n=200, n_permutations=100, seed=0):
    """Token-level probe (kept for backward compatibility)."""
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
        m = lab >= 1
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
