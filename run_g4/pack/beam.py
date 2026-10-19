"""Batched beam search with no-repeat-n-gram blocking for ThesisTransformer.

Decode settings were fixed BEFORE looking at any beam output:
  beam 5, length penalty alpha 1.0 (score / len**alpha), no-repeat 4-gram on
  target subword ids, length cap 2 * source length + 10 (<= max_len).
4-grams, not 3-grams: 4.9% of dev references repeat a subword 3-gram but only
1.8% repeat a 4-gram (the same rate at which references repeat a WORD
trigram), so 4-gram blocking almost never forbids a legitimate repeat.

beam=1, no_repeat=0, a large length cap  ==  model.greedy_decode (tested).
"""
import torch
import torch.nn.functional as F

NEG = -1e9


def _banned(gen, n):
    """Tokens that would complete an n-gram already present in `gen`."""
    if n <= 0 or len(gen) < n - 1:
        return ()
    prefix = tuple(gen[len(gen) - (n - 1):])
    out = set()
    for i in range(len(gen) - n + 1):
        if tuple(gen[i:i + n - 1]) == prefix:
            out.add(gen[i + n - 1])
    return out


@torch.no_grad()
def beam_decode(model, src, word_ids, bos, eos, pad, beam=5, alpha=1.0,
                no_repeat=4, max_len=128, len_a=2.0, len_b=10):
    """Returns a list (len B) of generated token-id lists (no BOS/EOS)."""
    model.eval()
    dev = src.device
    B, K = src.size(0), beam
    src_pad = src == pad
    enc, _ = model.encode(src, src_pad, word_ids)
    src_len = (~src_pad).sum(1)
    limits = torch.clamp((len_a * src_len.float() + len_b).long(), max=max_len).tolist()
    S = enc.size(1)
    enc = enc.unsqueeze(1).expand(B, K, S, enc.size(-1)).reshape(B * K, S, -1)
    spad = src_pad.unsqueeze(1).expand(B, K, S).reshape(B * K, S)
    ys = torch.full((B * K, 1), bos, dtype=torch.long, device=dev)
    scores = torch.full((B, K), NEG, device=dev)
    scores[:, 0] = 0.0
    finished = [[] for _ in range(B)]      # (normalised score, tokens)
    done = [False] * B
    for step in range(max(limits)):
        seq = ys.size(1)
        causal = torch.triu(torch.ones(seq, seq, device=dev, dtype=torch.bool), 1)
        dec = model.decode(ys, enc, tgt_mask=causal, src_key_padding_mask=spad)
        logp = F.log_softmax(model.out_proj(dec[:, -1]).float(), -1)
        V = logp.size(-1)
        logp[:, pad] = NEG
        logp[:, bos] = NEG
        if no_repeat:
            gens = ys[:, 1:].tolist()
            for r, g in enumerate(gens):
                ban = _banned(g, no_repeat)
                if ban:
                    logp[r, list(ban)] = NEG
        cand = (scores.reshape(B * K, 1) + logp).reshape(B, K * V)
        top_s, top_i = cand.topk(2 * K, dim=1)
        top_s, top_i = top_s.tolist(), top_i.tolist()
        new_rows, new_tok, new_sc = [], [], []
        ys_list = None
        for b in range(B):
            live = []
            if not done[b]:
                gen_len = seq                   # tokens incl. the new one
                at_limit = seq >= limits[b]
                for rank, (s, idx) in enumerate(zip(top_s[b], top_i[b])):
                    if s <= NEG / 2:
                        break
                    bi, tk = divmod(idx, V)
                    row = b * K + bi
                    if tk == eos or at_limit:
                        if rank < K:
                            if ys_list is None:
                                ys_list = ys[:, 1:].tolist()
                            toks = ys_list[row] + ([] if tk == eos else [tk])
                            finished[b].append((s / (gen_len ** alpha), toks))
                        continue
                    live.append((s, row, tk))
                    if len(live) == K:
                        break
                if len(finished[b]) >= K or at_limit or not live:
                    done[b] = True
            if done[b]:
                live = []
            while len(live) < K:
                live.append((NEG, b * K, eos))
            for s, row, tk in live:
                new_sc.append(s)
                new_rows.append(row)
                new_tok.append(tk)
        if all(done):
            break
        rows = torch.tensor(new_rows, device=dev)
        ys = torch.cat([ys[rows], torch.tensor(new_tok, device=dev).unsqueeze(1)], 1)
        scores = torch.tensor(new_sc, device=dev).reshape(B, K)
    out = []
    gens = ys[:, 1:].tolist()
    sc = scores.tolist()
    for b in range(B):
        if finished[b]:
            out.append(max(finished[b], key=lambda x: x[0])[1])
        else:                                   # nothing finished: best live
            k = max(range(K), key=lambda k: sc[b][k])
            out.append(gens[b * K + k])
    return out


@torch.no_grad()
def decode_pairs(model, pairs, src_tok, tgt_tok, cfg, device, batch=48, sort=True, **kw):
    """Full-dev decode in dev order. kw goes to beam_decode."""
    from torch.nn.utils.rnn import pad_sequence
    import data as D
    pad, bos, eos = cfg["pad_id"], cfg["bos_id"], cfg["eos_id"]
    enc = []
    for src, _ in pairs:
        ids, starts = D.encode_words(src_tok, src, bos, eos, cfg["max_len"])
        enc.append((torch.tensor(ids), D.word_ids_from_starts(starts, len(ids))))
    order = sorted(range(len(enc)), key=lambda i: len(enc[i][0])) if sort \
        else list(range(len(enc)))      # sort=False: row j of a batch == pairs[j]
    hyps = [None] * len(enc)
    for b0 in range(0, len(order), batch):
        idx = order[b0:b0 + batch]
        src = pad_sequence([enc[i][0] for i in idx], batch_first=True,
                           padding_value=pad).to(device)
        wid = pad_sequence([enc[i][1] for i in idx], batch_first=True,
                           padding_value=-1).to(device)
        outs = beam_decode(model, src, wid, bos, eos, pad,
                           max_len=cfg["max_len"], **kw)
        for toks, i in zip(outs, idx):
            hyps[i] = tgt_tok.decode([t for t in toks if t not in (pad, bos, eos)])
    return hyps
