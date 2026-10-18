"""Transformer encoder-decoder with CMM / auxiliary-head variants.

Variants (parameter-matched to ~20.34M by adjusting FFN width):
    plain_bpe   vanilla transformer, no auxiliary heads
    bpe_aux     + auxiliary root/pattern prediction heads (no memory)
    bpe_placebo + auxiliary heads predicting matched fake labels
    cmm         full CMM (root+pattern) + auxiliary heads on real labels
    cmm_nobind  CMM without the binding term + auxiliary heads
    cmm_placebo CMM + auxiliary heads on matched fake labels
    flat        flat memory + auxiliary heads on real labels

The auxiliary heads predict geezRPA-derived root/pattern labels from the
final encoder states. Their outputs are NEVER fed back into the translation
representation; labels are training-only targets, absent at inference.
"""

import copy
import math

import torch
import torch.nn as nn

from cmm import CompositionalMorphMemory, FlatMemory

VARIANTS = ["plain_bpe", "bpe_aux", "bpe_placebo", "cmm",
            "cmm_nobind", "cmm_placebo", "flat"]

# 0-based encoder layer indices holding the memory module.
# Report says "encoder layers 1 and 2"; original code is lost, so this is a
# documented assumption (0-based indices 1 and 2). Record any change.
DEFAULT_CMM_LAYERS = [1, 2]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout, memory=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.memory = memory  # CMM / flat, applied after self-attn, before FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_key_padding_mask=None):
        routing = None
        a, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask,
                              need_weights=False)
        x = self.norm1(x + self.dropout(a))
        if self.memory is not None:
            x, routing = self.memory(x, key_padding_mask=src_key_padding_mask)
        x = self.norm2(x + self.ffn(x))
        return x, routing


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ffn_dim, dropout):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                               batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                                batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None,
                tgt_key_padding_mask=None, src_key_padding_mask=None):
        a, _ = self.self_attn(x, x, x, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask,
                              need_weights=False)
        x = self.norm1(x + self.dropout(a))
        a, _ = self.cross_attn(x, enc_out, enc_out,
                               key_padding_mask=src_key_padding_mask,
                               need_weights=False)
        x = self.norm2(x + self.dropout(a))
        x = self.norm3(x + self.ffn(x))
        return x


class ThesisTransformer(nn.Module):
    def __init__(self, cfg, n_root_classes, n_pattern_classes):
        super().__init__()
        self.cfg = cfg
        self.variant = cfg["variant"]
        d = cfg["d_model"]
        self.d_model = d
        self.pad_id = cfg["pad_id"]

        self.has_aux = self.variant != "plain_bpe"
        self.use_cmm = self.variant in ("cmm", "cmm_nobind", "cmm_placebo")
        self.use_flat = self.variant == "flat"
        self.cmm_layers = list(cfg.get("cmm_layers", DEFAULT_CMM_LAYERS))

        # src/tgt vocab sizes may differ (task 2: morpheme src, BPE tgt);
        # both default to vocab_size so the shared-BPE behavior is unchanged.
        src_vocab = cfg.get("src_vocab_size", cfg["vocab_size"])
        tgt_vocab = cfg.get("tgt_vocab_size", cfg["vocab_size"])
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=cfg["pad_id"])
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=cfg["pad_id"])
        self.pos = PositionalEncoding(d, max_len=cfg["max_len"] + 8,
                                      dropout=cfg["dropout"])
        self.scale = math.sqrt(d)

        def make_memory():
            if self.use_cmm:
                return CompositionalMorphMemory(
                    d, n_root_slots=cfg["n_root_slots"],
                    n_pattern_slots=cfg["n_pattern_slots"],
                    with_binding=(self.variant != "cmm_nobind"),
                    dropout=cfg["dropout"])
            if self.use_flat:
                return FlatMemory(d, n_slots=cfg["flat_slots"],
                                  dropout=cfg["dropout"])
            return None

        self.enc_layers = nn.ModuleList([
            EncoderLayer(d, cfg["n_heads"], cfg["ffn_dim"], cfg["dropout"],
                         memory=make_memory() if i in self.cmm_layers and
                         (self.use_cmm or self.use_flat) else None)
            for i in range(cfg["n_encoder_layers"])
        ])
        self.dec_layers = nn.ModuleList([
            DecoderLayer(d, cfg["n_heads"], cfg["ffn_dim"], cfg["dropout"])
            for _ in range(cfg["n_decoder_layers"])
        ])
        self.out_proj = nn.Linear(d, cfg.get("tgt_vocab_size",
                                              cfg["vocab_size"]))

        self.n_root_classes = n_root_classes
        self.n_pattern_classes = n_pattern_classes
        if self.has_aux:
            self.root_head = nn.Linear(d, n_root_classes)
            self.pattern_head = nn.Linear(d, n_pattern_classes)
        else:
            self.root_head = None
            self.pattern_head = None

    def encode(self, src, src_key_padding_mask=None):
        x = self.pos(self.src_emb(src) * self.scale)
        routings = []
        for layer in self.enc_layers:
            x, routing = layer(x, src_key_padding_mask)
            if routing is not None:
                routings.append(routing)
        return x, routings

    def decode(self, tgt, enc_out, tgt_mask=None, tgt_key_padding_mask=None,
               src_key_padding_mask=None):
        x = self.pos(self.tgt_emb(tgt) * self.scale)
        for layer in self.dec_layers:
            x = layer(x, enc_out, tgt_mask, tgt_key_padding_mask,
                      src_key_padding_mask)
        return x

    def forward(self, src, tgt_in):
        src_pad = (src == self.pad_id)
        enc_out, routings = self.encode(src, src_pad)
        tgt_pad = (tgt_in == self.pad_id)
        seq = tgt_in.size(1)
        causal = torch.triu(torch.ones(seq, seq, device=src.device,
                                       dtype=torch.bool), diagonal=1)
        dec_out = self.decode(tgt_in, enc_out, tgt_mask=causal,
                              tgt_key_padding_mask=tgt_pad,
                              src_key_padding_mask=src_pad)
        logits = self.out_proj(dec_out)
        out = {"logits": logits, "routings": routings}
        if self.has_aux:
            out["root_logits"] = self.root_head(enc_out)
            out["pattern_logits"] = self.pattern_head(enc_out)
        return out

    @torch.no_grad()
    def greedy_decode(self, src, max_len, bos_id, eos_id):
        self.eval()
        src_pad = (src == self.pad_id)
        enc_out, _ = self.encode(src, src_pad)
        bsz = src.size(0)
        ys = torch.full((bsz, 1), bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(bsz, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            seq = ys.size(1)
            causal = torch.triu(torch.ones(seq, seq, device=src.device,
                                           dtype=torch.bool), diagonal=1)
            dec = self.decode(ys, enc_out, tgt_mask=causal,
                              src_key_padding_mask=src_pad)
            nxt = self.out_proj(dec[:, -1]).argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, self.pad_id), nxt)
            ys = torch.cat([ys, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == eos_id)
            if done.all():
                break
        return ys


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(cfg, n_root_classes, n_pattern_classes,
                target_params=None, tol=1e-4):
    """Build a variant, adjusting FFN width to match target_params.

    The report's matched grid kept every system within 0.01% of CMM's
    20,343,170 parameters by adjusting FFN width. We reproduce that here.
    """
    cfg = dict(cfg)
    if target_params is None:
        target_params = cfg.get("target_params")
    if target_params is None:
        return ThesisTransformer(cfg, n_root_classes, n_pattern_classes)

    lo, hi = 128, 4096
    best = None
    for _ in range(14):  # binary search on ffn_dim
        mid = (lo + hi) // 2
        cfg["ffn_dim"] = mid
        m = ThesisTransformer(cfg, n_root_classes, n_pattern_classes)
        n = count_params(m)
        if best is None or abs(n - target_params) < abs(best[0] - target_params):
            best = (n, mid)
        if n < target_params:
            lo = mid + 1
        else:
            hi = mid - 1
    n, ffn = best
    rel = abs(n - target_params) / target_params
    # One ffn_dim step moves ~6.2k params (~3e-4 relative), so 1e-4 is not
    # reachable by FFN width alone; 5e-4 is the practical floor here.
    tol = max(tol, 5e-4)
    assert rel <= tol, f"param match failed: {n} vs {target_params} (rel {rel:.5f})"
    cfg["ffn_dim"] = ffn
    model = ThesisTransformer(cfg, n_root_classes, n_pattern_classes)
    model.matched_params = n
    return model
