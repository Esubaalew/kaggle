"""Decoder-side root memory: the decoder reads the encoder's grounded root slots.

The encoder's grounded memory (grounded.GroundedMorphMemory) binds slot k to
root class k: forms of a Ge'ez root, including forms never seen in training,
route to the same slot and read the same stored vector. The decoder, however,
generates Amharic without ever consulting those slots.

DecoderRootMemory lets each decoder position query the root bank and add the
retrieved vector to its state through a scalar gate, as the encoder memory
does (protocol Eqs. 4, 5, 9; no pattern bank and no binding term, because
there are no Amharic pattern labels).

  shared=<encoder memory>   reads the SAME M_K / M_V as the encoder, so slot k
                            means "Ge'ez root k and its Amharic rendering":
                            one bilingual root memory
  shared=None               owns a separate bank of the same size (control)
  grounded=True             routing supervised with target-side root labels
  grounded=False            routing free (control); same regularisers

Target-side labels live in the SAME class space as the source labels: an
Amharic word gets class k when it carries the Amharic consonant skeleton that
Ge'ez root class k is most strongly associated with in training pairs
(build_target_labels.py). They are training-only, like the source labels, so
inference needs no analyser on either side.

Alignment. Decoder input position j holds tgt[j] and predicts tgt[j+1]. Its
routing label is the root of the word containing tgt[j+1], i.e. "which root
am I producing now": labels = tgt_root[:, 1:].
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-9


class DecoderRootMemory(nn.Module):
    def __init__(self, d_model, n_slots, d_key, shared=None, grounded=True,
                 dropout=0.1):
        super().__init__()
        self.d_model, self.n_slots, self.d_key = d_model, n_slots, d_key
        self.grounded = grounded
        self.wq = nn.Linear(d_model, d_key)
        if shared is None:
            self.mk = nn.Parameter(torch.empty(n_slots, d_key))
            self.mv = nn.Parameter(torch.empty(n_slots, d_model))
            nn.init.xavier_uniform_(self.mk)
            nn.init.xavier_uniform_(self.mv)
            object.__setattr__(self, "_shared", None)
        else:
            if getattr(shared, "tied_kv", True):
                raise ValueError("shared bank must have untied M_K / M_V")
            if shared.n_root_slots != n_slots or shared.d_key != d_key:
                raise ValueError("shared bank shape mismatch: "
                                 f"{shared.n_root_slots}x{shared.d_key} vs "
                                 f"{n_slots}x{d_key}")
            # Deliberately NOT registered as a submodule: the parameters are
            # owned by the encoder memory, counted once, saved once, and
            # receive gradient from both sides.
            object.__setattr__(self, "_shared", shared)
        self.gate = nn.Linear(2 * d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    @property
    def is_shared(self):
        return self._shared is not None

    def bank(self):
        if self._shared is not None:
            return self._shared.mk_root, self._shared.mv_root
        return self.mk, self.mv

    def forward(self, h, key_padding_mask=None):
        mk, mv = self.bank()
        scores = F.softmax(self.wq(h) @ mk.t() / math.sqrt(self.d_key), dim=-1)
        r = scores @ mv
        lam = torch.sigmoid(self.gate(torch.cat([h, r], dim=-1)))
        out = self.norm(h + self.dropout(lam * r))
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out, {"root_scores": scores, "gate": lam.detach(),
                     "grounded": self.grounded, "shared": self.is_shared}


def decoder_grounding_loss(dec_routings, tgt_root, tgt_in, pad_id,
                           weight=0.5, entropy_weight=0.0, entropy_sign=1.0,
                           ignore_index=-100):
    """CE of decoder routing against shifted target root labels.

    dec_routings : list of info dicts from DecoderRootMemory
    tgt_root     : (B, T) target root classes aligned with tgt (-100 = none)
    tgt_in       : (B, T-1) decoder input ids, used for the padding mask
    The entropy term is applied to grounded AND free decoder memories alike,
    so grounded vs free differs only in the CE term.
    """
    terms = {}
    if not dec_routings:
        return None, terms
    labels = tgt_root[:, 1:]                       # predicts tgt[j+1]
    valid = (tgt_in != pad_id)
    ce, ents = [], []
    for info in dec_routings:
        s = info["root_scores"].clamp_min(EPS)     # (B, T-1, K)
        h = -(s * s.log()).sum(-1)
        ents.append((h * valid).sum() / valid.sum().clamp_min(1))
        if info["grounded"] and (labels != ignore_index).any():
            ce.append(F.nll_loss(s.log().reshape(-1, s.size(-1)),
                                 labels.reshape(-1), ignore_index=ignore_index))
    total = None
    if ce:
        l = torch.stack(ce).mean()
        total = weight * l
        terms["dec_route_root"] = l.detach()
    if entropy_weight:
        e = torch.stack(ents).mean()
        total = (0 if total is None else total) + entropy_sign * entropy_weight * e
        terms["dec_entropy"] = e.detach()
    return total, terms


@torch.no_grad()
def decoder_routing_accuracy(model, loader, device):
    """Teacher-forced dev check: does the decoder route to the right root?

    For each decoder memory layer, the share of labelled target positions
    whose routing argmax equals the label, against the rate of always
    guessing the most frequent label (a baseline any classifier must beat).
    Only meaningful for shared or grounded decoder memories, where slot k is
    class k.
    """
    import collections
    model.eval()
    hits = collections.Counter()
    n, freq = 0, collections.Counter()
    for batch in loader:
        src, tgt, _, _, wid, tgt_root = [x.to(device) for x in batch]
        out = model(src, tgt[:, :-1], wid)
        labels = tgt_root[:, 1:]
        m = labels >= 1
        if not m.any():
            continue
        n += int(m.sum())
        freq.update(labels[m].tolist())
        for li, info in enumerate(out.get("dec_routings", [])):
            hits[li] += int((info["root_scores"].argmax(-1)[m] == labels[m]).sum())
    if n == 0:
        return {"error": "no labelled target positions"}
    return {"n_labelled_positions": n,
            "accuracy_by_layer": {str(k): v / n for k, v in sorted(hits.items())},
            "majority_class_rate": freq.most_common(1)[0][1] / n}
