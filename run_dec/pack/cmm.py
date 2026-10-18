"""Compositional Morphological Memory (CMM) modules.

Rebuilt from the 2026-08-01 advisor report. The original training code was
deleted; this is a clean-room reimplementation of the documented protocol.

CMM (full):
    Placed in encoder layers 1 and 2, after self-attention, before the FFN.
    - Root bank:   R_bank, 256 slots x d_model. Routing over H gives R.
    - Pattern bank: P_bank, 64 slots x d_model. Routing over (H + R) gives A.
    - Composition: C = [R ; A ; R*A] (binding term), projected to d_model,
      combined with H through a learned gate, plus a residual connection.
    - geezRPA root/pattern labels are training-only supervision targets for
      the auxiliary heads in model.py. They are NEVER fed into the model and
      are absent at inference.

cmm_nobind: same, but the composition drops the binding term: C = [R ; A].
flat: a single non-compositional memory bank (320 slots), one routing read,
      gated residual. No root/pattern factorisation.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositionalMorphMemory(nn.Module):
    def __init__(self, d_model, n_root_slots=256, n_pattern_slots=64,
                 with_binding=True, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_root_slots = n_root_slots
        self.n_pattern_slots = n_pattern_slots
        self.with_binding = with_binding

        self.root_bank = nn.Parameter(torch.randn(n_root_slots, d_model) / math.sqrt(d_model))
        self.pattern_bank = nn.Parameter(torch.randn(n_pattern_slots, d_model) / math.sqrt(d_model))

        comp_dim = 3 * d_model if with_binding else 2 * d_model
        self.compose = nn.Linear(comp_dim, d_model)
        self.gate_h = nn.Linear(d_model, d_model)
        self.gate_c = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, key_padding_mask=None):
        """h: (batch, seq, d). Returns (out, info) with routing distributions."""
        scale = 1.0 / math.sqrt(self.d_model)

        root_logits = torch.matmul(h, self.root_bank.t()) * scale
        root_scores = F.softmax(root_logits, dim=-1)
        r = torch.matmul(root_scores, self.root_bank)

        cond = h + r
        pat_logits = torch.matmul(cond, self.pattern_bank.t()) * scale
        pat_scores = F.softmax(pat_logits, dim=-1)
        a = torch.matmul(pat_scores, self.pattern_bank)

        if self.with_binding:
            comp = torch.cat([r, a, r * a], dim=-1)
        else:
            comp = torch.cat([r, a], dim=-1)
        c = self.compose(comp)
        gate = torch.sigmoid(self.gate_h(h) + self.gate_c(c))
        out = h + self.dropout(gate * c)
        out = self.norm(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        info = {"root_scores": root_scores, "pattern_scores": pat_scores}
        return out, info


class FlatMemory(nn.Module):
    """Non-compositional control: one flat bank, single routing read."""

    def __init__(self, d_model, n_slots=320, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_slots = n_slots
        self.bank = nn.Parameter(torch.randn(n_slots, d_model) / math.sqrt(d_model))
        self.gate_h = nn.Linear(d_model, d_model)
        self.gate_m = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, key_padding_mask=None):
        scale = 1.0 / math.sqrt(self.d_model)
        logits = torch.matmul(h, self.bank.t()) * scale
        scores = F.softmax(logits, dim=-1)
        m = torch.matmul(scores, self.bank)
        gate = torch.sigmoid(self.gate_h(h) + self.gate_m(m))
        out = h + self.dropout(gate * m)
        out = self.norm(out)
        if key_padding_mask is not None:
            out = out.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
        return out, {"scores": scores}
