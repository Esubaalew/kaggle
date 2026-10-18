"""Step 3: does the grounded root memory work inside a strong pretrained model?

Host model: NLLB-200-distilled-600M. Ge'ez is not one of NLLB's languages
(its tag gez_Ethi maps to <unk>), so the source side uses tir_Ethi (Tigrinya,
the closest NLLB language: North Ethio-Semitic, same script). Target amh_Ethi.

Systems (everything else identical: data, tokenizer, schedule, decoding):
  zeroshot  NLLB as released, no training
  ft        full fine-tune
  aux       ft + root/pattern side heads on the final encoder state (bpe_aux analogue)
  gmm       ft + grounded morphological memory inserted as a RESIDUAL module
            after encoder layers 4 and 8 (of 12): slot k of the root bank is
            root class k, the routing is supervised by the labels, and the
            retrieved composition c is added as h + lambda * c.  The compose
            projection starts at zero, so at step 0 the model is exactly
            NLLB.  No LayerNorm is added, so the pretrained residual stream
            is untouched at initialisation.

Settings fixed before any run: AdamW lr 5e-5 (pretrained) / 1e-3 (new
modules), warmup 300, linear decay over 8 epochs, batch 16 x 2 accumulation,
label smoothing 0.1, fp16 autocast, early stop on dev loss (patience 2).
Memory (T4, 15 GB): the 256k-word embedding matrix (shared with the output
layer; 262M of 615M parameters) is FROZEN for every trained system, output
scores are computed only at real target positions, and batches are capped
at 1,024 padded tokens (at most 16 sentences).
Decoding: beam 5, no-repeat 4-gram, length penalty 1.0 - the same as eval2.
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL = os.environ.get("NLLB_MODEL", "facebook/nllb-200-distilled-600M")   # env override: smoke tests only
SRC_LANG, TGT_LANG = "tir_Ethi", "amh_Ethi"
MEM_LAYERS = (3, 7)          # after the 4th and 8th encoder layer
MAX_SRC, MAX_TGT = 160, 160
IGN = -100


def pool_words(h, word_ids, n_words):
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
    return pooled / counts.clamp_min(1.0).unsqueeze(-1), flat_index, valid


def _fit(x, n, val):
    if x.size(1) < n:
        x = F.pad(x, (0, n - x.size(1)), value=val)
    return x[:, :n].reshape(-1)


class ResidualGMM(nn.Module):
    """Grounded memory (Eqs. 4-9 of the proposal) as a residual adapter."""

    def __init__(self, d, n_root, n_pat, d_key=256, dropout=0.1):
        super().__init__()
        self.dk = d_key
        self.wq_root = nn.Linear(d, d_key)
        self.wq_pat = nn.Linear(d, d_key)
        self.mk_root = nn.Parameter(torch.empty(n_root, d_key))
        self.mv_root = nn.Parameter(torch.empty(n_root, d))
        self.mk_pat = nn.Parameter(torch.empty(n_pat, d_key))
        self.mv_pat = nn.Parameter(torch.empty(n_pat, d))
        for p in (self.mk_root, self.mv_root, self.mk_pat, self.mv_pat):
            nn.init.xavier_uniform_(p)
        self.compose = nn.Linear(3 * d, d)
        nn.init.zeros_(self.compose.weight)
        nn.init.zeros_(self.compose.bias)
        self.gate = nn.Linear(2 * d, 1)
        self.drop = nn.Dropout(dropout)
        self.word_ids = None
        self.override = None
        self.last = None

    def forward(self, h):
        wid = self.word_ids
        b, t, d = h.shape
        assert wid is not None and tuple(wid.shape) == (b, t), \
            (None if wid is None else tuple(wid.shape), tuple(h.shape))
        n_words = max(int(wid.max().item()) + 1, 1)
        with torch.autocast(h.device.type, enabled=False):
            hf = h.float()
            x, flat_index, valid = pool_words(hf, wid, n_words)
            rs = F.softmax(self.wq_root(x) @ self.mk_root.t() / math.sqrt(self.dk), -1)
            r = rs @ self.mv_root
            ps = F.softmax(self.wq_pat(x + r) @ self.mk_pat.t() / math.sqrt(self.dk), -1)
            a = ps @ self.mv_pat                       # pattern from the ORIGINAL r
            ov = self.override or {}
            if ov.get("root") is not None:
                force = _fit(ov["root"], n_words, -1)
                m = force >= 0
                if m.any():
                    oh = F.one_hot(force.clamp_min(0), rs.size(-1)).float()
                    r = torch.where(m.unsqueeze(-1), oh, rs) @ self.mv_root
            c = self.compose(torch.cat([r, a, r * a], -1))
            if ov.get("kill") is not None:
                c = c.masked_fill(_fit(ov["kill"], n_words, False).unsqueeze(-1), 0.0)
            if ov.get("off"):
                c = torch.zeros_like(c)
            c = c[flat_index].reshape(b, t, d) * valid.unsqueeze(-1)
            lam = torch.sigmoid(self.gate(torch.cat([hf, c], -1)))
            out = hf + self.drop(lam * c)
        self.last = {"root_scores": rs[flat_index].reshape(b, t, -1),
                     "pattern_scores": ps[flat_index].reshape(b, t, -1),
                     "gate": lam.detach()}
        return out.to(h.dtype)


class LayerWithMemory(nn.Module):
    def __init__(self, layer, mem):
        super().__init__()
        self.layer = layer
        self.mem = mem

    def forward(self, *args, **kwargs):
        out = self.layer(*args, **kwargs)
        if isinstance(out, tuple):            # transformers 4.x
            return (self.mem(out[0]),) + tuple(out[1:])
        return self.mem(out)                  # transformers 5.x


def grounding_loss(mems, root_lab, pat_lab, ent_w=0.01):
    """Same objective as grounded.grounding_loss for gmm: 0.5 CE(root) +
    0.5 CE(pattern) on the routing, plus 0.01 * entropy (sharpening)."""
    ce_r, ce_p, ents = [], [], []
    for m in mems:
        rs = m.last["root_scores"].clamp_min(1e-9)
        ps = m.last["pattern_scores"].clamp_min(1e-9)
        ce_r.append(F.nll_loss(rs.log().reshape(-1, rs.size(-1)), root_lab.reshape(-1), ignore_index=IGN))
        ce_p.append(F.nll_loss(ps.log().reshape(-1, ps.size(-1)), pat_lab.reshape(-1), ignore_index=IGN))
        for s in (rs, ps):
            ents.append(-(s * s.log()).sum(-1).mean())
    return 0.5 * torch.stack(ce_r).mean() + 0.5 * torch.stack(ce_p).mean() + \
        ent_w * torch.stack(ents).mean()


def strip_cf(s):
    return "".join(c for c in s if unicodedata.category(c) != "Cf")


def load_pairs(path):
    out = []
    for line in open(path, encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) == 2 and p[0].strip() and p[1].strip():
            out.append((p[0].strip(), p[1].strip()))
    return out


class Encoder:
    """Word-by-word SentencePiece encoding (identical to whole-sentence
    encoding for this data: 0 mismatches in 2,000 sentences), so a word's
    pieces, its label and its routing unit are the same positions."""

    def __init__(self, tok, entries):
        self.tok = tok
        self.entries = entries or {}
        self.cache = {}
        self.src_id = tok.convert_tokens_to_ids(SRC_LANG)
        self.tgt_id = tok.convert_tokens_to_ids(TGT_LANG)
        self.eos = tok.eos_token_id
        self.pad = tok.pad_token_id
        assert self.src_id != tok.unk_token_id and self.tgt_id != tok.unk_token_id

    def pieces(self, w):
        p = self.cache.get(w)
        if p is None:
            p = self.tok(w, add_special_tokens=False).input_ids
            self.cache[w] = p
        return p

    def src(self, text):
        ids, wid, root, pat = [self.src_id], [-1], [IGN], [IGN]
        for k, w in enumerate(text.split()):
            p = self.pieces(w)
            e = self.entries.get(w) or {}
            r, q = int(e.get("root", 0)), int(e.get("pattern", 0))
            ids += p
            wid += [k] * len(p)
            root += [r if r > 0 else IGN] * len(p)
            pat += [q if q > 0 else IGN] * len(p)
        n = MAX_SRC - 1
        return ids[:n] + [self.eos], wid[:n] + [-1], root[:n] + [IGN], pat[:n] + [IGN]

    def tgt(self, text):
        ids = [self.tgt_id]
        for w in text.split():
            ids += self.pieces(w)
        return ids[:MAX_TGT - 1] + [self.eos]


def pad_to(seqs, val):
    n = max(len(s) for s in seqs)
    return torch.tensor([s + [val] * (n - len(s)) for s in seqs], dtype=torch.long)


def make_batches(items, bs, rng=None, max_tok=1024):
    order = sorted(range(len(items)), key=lambda i: (len(items[i][0]), len(items[i][4])))
    batches, cur, longest = [], [], 0
    for i in order:
        L = max(len(items[i][0]), len(items[i][4]))
        if cur and (len(cur) >= bs or max(longest, L) * (len(cur) + 1) > max_tok):
            batches.append(cur)
            cur, longest = [], 0
        cur.append(i)
        longest = max(longest, L)
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def collate(items, idx, enc, device):
    s = [items[i] for i in idx]
    src = pad_to([x[0] for x in s], enc.pad).to(device)
    wid = pad_to([x[1] for x in s], -1).to(device)
    root = pad_to([x[2] for x in s], IGN).to(device)
    pat = pad_to([x[3] for x in s], IGN).to(device)
    lab = pad_to([x[4] for x in s], IGN).to(device)
    dec_in = torch.cat([torch.full((lab.size(0), 1), enc.eos, device=device),
                        lab[:, :-1].masked_fill(lab[:, :-1] == IGN, enc.pad)], 1)
    return src, wid, root, pat, lab, dec_in


class System:
    def __init__(self, kind, n_root, n_pat, device, pad, aux_weight=0.5):
        from transformers import AutoModelForSeq2SeqLM
        self.kind = kind
        self.pad = pad
        self.aux_weight = aux_weight
        self.model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
        d = self.model.config.d_model
        self.mems = []
        self.heads = None
        if kind in ("gmm", "gmm_free"):      # gmm_free: same module, routing NOT supervised
            layers = self.model.model.encoder.layers
            for i in MEM_LAYERS:
                mem = ResidualGMM(d, n_root, n_pat, dropout=self.model.config.dropout).to(device)
                layers[i] = LayerWithMemory(layers[i], mem)
                self.mems.append(mem)
        if kind == "aux":
            self.heads = nn.ModuleDict({"root": nn.Linear(d, n_root),
                                        "pat": nn.Linear(d, n_pat)}).to(device)
        # frozen for all systems (memory budget); lm_head is the same tensor
        self.model.get_input_embeddings().weight.requires_grad_(False)

    def new_params(self):
        ps = [p for m in self.mems for p in m.parameters()]
        if self.heads is not None:
            ps += list(self.heads.parameters())
        return ps

    def train(self, flag=True):
        self.model.train(flag)
        if self.heads is not None:
            self.heads.train(flag)

    def set_words(self, wid):
        for m in self.mems:
            m.word_ids = wid

    def encode_only(self, src, wid):
        self.set_words(wid)
        return self.model.model.encoder(input_ids=src, attention_mask=(src != self.pad).long())

    def forward_selected(self, src, wid, lab, dec_in):
        """Output scores only at real target positions (saves ~GBs on a
        256k vocabulary). Returns (logits (N, V) float, gold (N,), outputs)."""
        self.set_words(wid)
        out = self.model.model(input_ids=src, attention_mask=(src != self.pad).long(),
                               decoder_input_ids=dec_in)
        valid = lab != IGN
        logits = self.model.lm_head(out.last_hidden_state[valid]).float()
        return logits, lab[valid], out

    def loss(self, src, wid, root, pat, lab, dec_in, smoothing):
        logits, gold, out = self.forward_selected(src, wid, lab, dec_in)
        l_tr = F.cross_entropy(logits, gold, label_smoothing=smoothing)
        total = l_tr
        if self.mems and self.kind == "gmm":
            total = total + grounding_loss(self.mems, root, pat)
        if self.heads is not None:
            hs = out.encoder_last_hidden_state.float()
            lr_ = self.heads["root"](hs)
            lp_ = self.heads["pat"](hs)
            w = self.aux_weight
            total = total + w * F.cross_entropy(lr_.reshape(-1, lr_.size(-1)), root.reshape(-1), ignore_index=IGN) \
                + w * F.cross_entropy(lp_.reshape(-1, lp_.size(-1)), pat.reshape(-1), ignore_index=IGN)
        return total, l_tr


def amp(device):
    return torch.autocast(device, dtype=torch.float16, enabled=device == "cuda")


@torch.no_grad()
def dev_loss(sysm, items, enc, device, bs=16):
    sysm.train(False)
    tot, n = 0.0, 0
    for idx in make_batches(items, bs):
        src, wid, root, pat, lab, dec_in = collate(items, idx, enc, device)
        with amp(device):
            logits, gold, _ = sysm.forward_selected(src, wid, lab, dec_in)
        tot += F.cross_entropy(logits, gold, reduction="sum").item()
        n += gold.numel()
    return tot / n


@torch.no_grad()
def translate(sysm, items, enc, tok, device, beam=5, no_repeat=4, bs=24, override_fn=None,
              keep_order=False):
    sysm.train(False)
    hyps = [None] * len(items)
    order = list(range(len(items))) if keep_order else \
        sorted(range(len(items)), key=lambda i: len(items[i][0]))
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        src, wid, *_ = collate(items, idx, enc, device)
        sysm.set_words(wid)
        if override_fn is not None:
            for m in sysm.mems:
                m.override = override_fn(b0, len(idx), device)
        with amp(device):
            gen = sysm.model.generate(input_ids=src, attention_mask=(src != enc.pad).long(),
                                      forced_bos_token_id=enc.tgt_id, num_beams=beam,
                                      no_repeat_ngram_size=no_repeat, length_penalty=1.0,
                                      max_new_tokens=min(256, 2 * src.size(1) + 10),
                                      early_stopping=True)
        for m in sysm.mems:
            m.override = None
        for row, i in zip(tok.batch_decode(gen, skip_special_tokens=True), idx):
            hyps[i] = row.strip()
    return hyps


def score(hyps, refs):
    from sacrebleu.metrics import BLEU, CHRF
    h = [strip_cf(x) for x in hyps]
    r = [[strip_cf(x) for x in refs]]
    return {"chrF_char": CHRF(word_order=0).corpus_score(h, r).score,
            "chrF++": CHRF(word_order=2).corpus_score(h, r).score,
            "BLEU": BLEU().corpus_score(h, r).score}


@torch.no_grad()
def routed_words(sysm, items, enc, device, bs=32):
    """{(sentence, word): (argmax root slot, gate)} at the last memory layer."""
    sysm.train(False)
    out = {}
    for b0 in range(0, len(items), bs):
        idx = list(range(b0, min(b0 + bs, len(items))))
        src, wid, *_ = collate(items, idx, enc, device)
        with amp(device):
            sysm.encode_only(src, wid)
        am = sysm.mems[-1].last["root_scores"].argmax(-1).cpu()
        gt = sysm.mems[-1].last["gate"].squeeze(-1).float().cpu()
        w_cpu = wid.cpu()
        for j, i in enumerate(idx):
            for p in range(src.size(1)):
                w = int(w_cpu[j, p])
                if w >= 0 and (i, w) not in out:
                    out[(i, w)] = (int(am[j, p]), float(gt[j, p]))
    return out


def probe(routed, dev_pairs, entries, train_types):
    hit, cnt, cls = collections.Counter(), collections.Counter(), collections.Counter()
    for (i, w), (slot, _) in routed.items():
        word = dev_pairs[i][0].split()[w]
        r = int((entries.get(word) or {}).get("root", 0))
        if r <= 0:
            continue
        k = "seen" if word in train_types else "unseen"
        cnt[k] += 1
        cls[r] += 1
        hit[k] += int(slot == r)
    tot = sum(cnt.values())
    return {"own_slot_seen": hit["seen"] / max(cnt["seen"], 1), "n_seen": cnt["seen"],
            "own_slot_unseen": hit["unseen"] / max(cnt["unseen"], 1), "n_unseen": cnt["unseen"],
            "majority_class_rate": max(cls.values()) / max(tot, 1) if cls else None}


def slot_swap(sysm, dev_items, dev_pairs, base_hyps, routed, tmap, entries, enc, tok, device,
              n_swaps=2, seed=0, max_items=600):
    """Same design as slot_tests.slot_swap (small models), for the NLLB memory."""
    rng = random.Random(seed)
    T = collections.defaultdict(set)
    for w, k in tmap.items():
        T[int(k)].add(w)
    pool = sorted(k for k, ws in T.items() if len(ws) >= 3)
    cls = lambda text: {tmap[w] for w in text.split() if w in tmap}
    items = []
    for i, ((src, ref), hyp) in enumerate(zip(dev_pairs, base_hyps)):
        hc, rc = cls(hyp), cls(ref)
        cands = []
        for wi, w in enumerate(src.split()):
            r = int((entries.get(w) or {}).get("root", 0))
            if r > 0 and r in T and r in hc and r in rc and (i, wi) in routed:
                cands.append((wi, w, r))
        if not cands:
            continue
        wi, w, r = rng.choice(cands)
        alts = [k for k in pool if k not in hc | rc | {r}]
        for rp in rng.sample(alts, min(n_swaps, len(alts))):
            items.append({"i": i, "wi": wi, "word": w, "r": r, "rp": rp,
                          "routed": routed[(i, wi)][0], "gate": routed[(i, wi)][1]})
    rng.shuffle(items)
    items = items[:max_items]
    if not items:
        return {"n_items": 0}, [], []
    sub = [dev_items[it["i"]] for it in items]

    def builder(cond):
        def f(b0, nb, device):
            chunk = items[b0:b0 + nb]
            W = max(len(dev_pairs[it["i"]][0].split()) for it in chunk)
            force = torch.full((nb, W), -1, dtype=torch.long)
            kill = torch.zeros((nb, W), dtype=torch.bool)
            for j, it in enumerate(chunk):
                if cond == "swap":
                    force[j, it["wi"]] = it["rp"]
                elif cond == "self":
                    force[j, it["wi"]] = it["routed"]
                else:
                    kill[j, it["wi"]] = True
            return {"root": force.to(device), "kill": kill.to(device)}
        return f
    H = {c: translate(sysm, sub, enc, tok, device, beam=1, no_repeat=4, bs=24,
                      override_fn=builder(c), keep_order=True) for c in ("self", "swap", "kill")}
    has = lambda text, k: any(tmap.get(w) == k for w in text.split())
    rows = []
    for j, it in enumerate(items):
        rows.append({**it, "succ_swap": has(H["swap"][j], it["rp"]), "succ_self": has(H["self"][j], it["rp"]),
                     "keep_swap": has(H["swap"][j], it["r"]), "keep_self": has(H["self"][j], it["r"]),
                     "keep_kill": has(H["kill"][j], it["r"])})
    n = len(rows)
    mean = lambda k: sum(r[k] for r in rows) / n
    ex = [{"src": dev_pairs[it["i"]][0], "word": it["word"], "r": it["r"], "rp": it["rp"],
           "ref": dev_pairs[it["i"]][1], "hyp_self": H["self"][j], "hyp_swap": H["swap"][j],
           "hyp_kill": H["kill"][j]} for j, it in enumerate(items[:40])]
    return {"n_items": n, "routed_to_own_root": sum(it["routed"] == it["r"] for it in items) / n,
            "mean_gate": sum(it["gate"] for it in items) / n,
            "success_swap": mean("succ_swap"), "success_self": mean("succ_self"),
            "keep_r_self": mean("keep_self"), "keep_r_swap": mean("keep_swap"),
            "keep_r_kill": mean("keep_kill")}, rows, ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["zeroshot", "ft", "aux", "gmm", "gmm_free"])
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--labels", default="analyser")
    ap.add_argument("--data-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--out", default="/kaggle/working/nllb")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=0, help="smoke tests only")
    ap.add_argument("--max-dev", type=int, default=0, help="smoke tests only")
    ap.add_argument("--save-weights", type=int, default=1)
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight of EACH side-head loss (aux only)")
    args = ap.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    labels = json.load(open(os.path.join(args.data_dir, f"labels_{args.labels}.json")))
    entries = labels["entries"]
    n_root, n_pat = len(labels["root_classes"]), len(labels["pattern_classes"])
    enc = Encoder(tok, entries)
    train = load_pairs(os.path.join(args.data_dir, "age_train.tsv"))
    dev = load_pairs(os.path.join(args.data_dir, "age_dev.tsv"))
    if args.max_train:
        train = train[:args.max_train]
    if args.max_dev:
        dev = dev[:args.max_dev]
    mk = lambda pairs: [(*enc.src(s), enc.tgt(t)) for s, t in pairs]
    tr_items, dv_items = mk(train), mk(dev)
    refs = [t for _, t in dev]
    train_types = {w for s, _ in train for w in s.split()}
    tag = args.system if args.aux_weight == 0.5 else f"{args.system}_w{args.aux_weight:g}".replace(".", "")
    run = f"{tag}_s{args.seed}" if args.system != "zeroshot" else "zeroshot"
    od = os.path.join(args.out, run)
    os.makedirs(od, exist_ok=True)
    print(run, "train", len(tr_items), "dev", len(dv_items), "device", device, flush=True)
    sysm = System(args.system, n_root, n_pat, device, enc.pad, args.aux_weight)
    n_new = sum(p.numel() for p in sysm.new_params())
    res = {"run": run, "system": args.system, "seed": args.seed, "labels": args.labels,
           "model": MODEL, "src_lang": SRC_LANG, "tgt_lang": TGT_LANG,
           "mem_layers": list(MEM_LAYERS) if args.system in ("gmm", "gmm_free") else None,
           "aux_weight": args.aux_weight if args.system == "aux" else None,
           "new_params": n_new,
           "nllb_params": sum(p.numel() for p in sysm.model.parameters()) -
           sum(p.numel() for m in sysm.mems for p in m.parameters())}
    t0 = time.time()
    if args.system != "zeroshot":
        new_ids = {id(p) for p in sysm.new_params()}
        base = [p for p in sysm.model.parameters() if id(p) not in new_ids and p.requires_grad]
        res["trainable_params"] = sum(p.numel() for p in base) + n_new
        groups = [{"params": base, "lr": 5e-5}]
        if new_ids:
            groups.append({"params": sysm.new_params(), "lr": 1e-3})
        opt = torch.optim.AdamW(groups, weight_decay=0.0)
        base_lrs = [g["lr"] for g in groups]
        all_params = [p for g in groups for p in g["params"]]
        scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
        bs, accum, warm = 16, 2, 300
        steps_per_epoch = math.ceil(len(make_batches(tr_items, bs)) / accum)
        total_steps = steps_per_epoch * args.epochs
        rng = random.Random(args.seed)
        best, best_state, bad, step = float("inf"), None, 0, 0
        log = open(os.path.join(od, "train_log.jsonl"), "w")
        for ep in range(1, args.epochs + 1):
            sysm.train(True)
            batches = make_batches(tr_items, bs, rng)
            run_l, run_n = 0.0, 0
            for bi, idx in enumerate(batches):
                src, wid, root, pat, lab, dec_in = collate(tr_items, idx, enc, device)
                with amp(device):
                    total, l_tr = sysm.loss(src, wid, root, pat, lab, dec_in, 0.1)
                scaler.scale(total / accum).backward()
                run_l += l_tr.item()
                run_n += 1
                if (bi + 1) % accum == 0 or bi + 1 == len(batches):
                    f = (step + 1) / warm if step < warm else \
                        max(0.0, (total_steps - step) / max(total_steps - warm, 1))
                    for g, lr in zip(opt.param_groups, base_lrs):
                        g["lr"] = lr * f
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 100 == 0:
                        print(f"  ep {ep} step {step}/{total_steps} train_ce {run_l / run_n:.4f} "
                              f"({time.time() - t0:.0f}s)", flush=True)
            dl = dev_loss(sysm, dv_items, enc, device)
            log.write(json.dumps({"epoch": ep, "step": step, "train_ce": run_l / max(run_n, 1),
                                  "dev_loss": dl, "t": round(time.time() - t0)}) + "\n")
            log.flush()
            print(f"epoch {ep} dev_loss {dl:.4f} ({time.time() - t0:.0f}s)", flush=True)
            if dl < best - 1e-4:
                best, bad = dl, 0
                best_state = {k: v.detach().to("cpu", copy=True) for k, v in sysm.model.state_dict().items()}
                res["best_epoch"] = ep
            else:
                bad += 1
                if bad >= 2:
                    break
        res["best_dev_loss"] = best
        sysm.model.load_state_dict(best_state)
        del opt
        if args.save_weights:
            torch.save({k: (v.half() if v.is_floating_point() else v) for k, v in best_state.items()},
                       os.path.join(od, "best_fp16.pt"))
        del best_state
    else:
        res["dev_loss"] = dev_loss(sysm, dv_items, enc, device)
    res["train_seconds"] = round(time.time() - t0)
    t1 = time.time()
    hyps = translate(sysm, dv_items, enc, tok, device, beam=5, no_repeat=4)
    json.dump(hyps, open(os.path.join(od, "dev_hyps.json"), "w"), ensure_ascii=False)
    res["dev_beam5_nr4"] = score(hyps, refs)
    print(run, "dev", {k: round(v, 2) for k, v in res["dev_beam5_nr4"].items()},
          f"({time.time() - t1:.0f}s)", flush=True)
    if sysm.mems:
        routed = routed_words(sysm, dv_items, enc, device)
        res["probe"] = probe(routed, dev, entries, train_types)
        print("  probe", res["probe"], flush=True)
        tmap = {w: int(k) for w, k in json.load(open(os.path.join(
            args.data_dir, f"labels_tgt_{args.labels}.json")))["map"].items()}
    if sysm.mems and args.system == "gmm":     # slot identity only defined when grounded
        greedy = translate(sysm, dv_items, enc, tok, device, beam=1, no_repeat=4)
        json.dump(greedy, open(os.path.join(od, "dev_hyps_beam1_nr4.json"), "w"), ensure_ascii=False)
        sw, rows, ex = slot_swap(sysm, dv_items, dev, greedy, routed, tmap, entries, enc, tok, device)
        res["slot_swap"] = sw
        json.dump(rows, open(os.path.join(od, "slot_swap_rows.json"), "w"), ensure_ascii=False)
        json.dump(ex, open(os.path.join(od, "slot_swap_examples.json"), "w"), ensure_ascii=False, indent=1)
        print("  swap", {k: round(v, 3) if isinstance(v, float) else v for k, v in sw.items()}, flush=True)
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(os.path.join(od, "eval.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
