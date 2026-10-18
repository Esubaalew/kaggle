"""G2 label semantics: >=1 in-bank root, -1 out-of-bank, -2 unknown,
0 explicit rootless; -100 only BOS/EOS/PAD. Negative classes are masked from
classification, while -1/-2 are also excluded from gate supervision.
"""

import hashlib
import json
import os
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

NONE = "<none>"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pairs(path):
    pairs, skipped = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                skipped += 1
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                skipped += 1
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if not src or not tgt:
                skipped += 1
                continue
            pairs.append((src, tgt))
    return pairs, skipped


def train_bpe(train_pairs, vocab_size, save_path, seed=11):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import Metaspace
    from tokenizers import decoders

    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = Metaspace(replacement="▁", prepend_scheme="always")
    tok.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    trainer = BpeTrainer(vocab_size=vocab_size,
                         special_tokens=["[PAD]", "[BOS]", "[EOS]", "[UNK]"])
    corpus = [s for p in train_pairs for s in p]
    tok.train_from_iterator(corpus, trainer=trainer)
    tok.save(save_path)
    return save_path


def load_tokenizer(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def encode_words(tokenizer, text, bos_id, eos_id, max_len):
    """Encode keeping word boundaries: returns (ids, word_starts)."""
    ids, word_starts = [bos_id], []
    for w in text.split():
        word_starts.append(len(ids))
        ids.extend(tokenizer.encode(w).ids)
    ids.append(eos_id)
    if len(ids) > max_len:
        ids = ids[:max_len]
    return ids, word_starts


def word_ids_from_starts(starts, n):
    """Word index per source position; -1 at BOS/EOS and past truncation."""
    wid = torch.full((n,), -1, dtype=torch.long)
    for k, pos in enumerate(starts):
        if pos >= n:
            continue
        end = starts[k + 1] if k + 1 < len(starts) else n - 1
        end = min(end, n - 1)                      # never include EOS
        wid[pos:end] = k
    return wid


class LabelCache:
    def __init__(self, path):
        with open(path, encoding="utf-8") as f:
            self.doc = json.load(f)
        if self.doc.get("provisional"):
            print("WARNING: provisional label cache - for smoke tests only, "
                  "NEVER for science.")
        self.root_classes = self.doc["root_classes"]
        self.pattern_classes = self.doc["pattern_classes"]
        self.entries = self.doc["entries"]
        self.n_root = len(self.root_classes)
        self.n_pattern = len(self.pattern_classes)

    def lookup(self, word):
        e = self.entries.get(word)
        if e is None:
            return 0, 0
        return int(e.get("root", 0)), int(e.get("pattern", 0))

    def placebo_entries(self, seed):
        """Permute (root, pattern) pairs across words, seeded. Joint
        marginals preserved, word->label mapping broken."""
        words = sorted(self.entries.keys())
        pairs = [(self.entries[w].get("root", 0), self.entries[w].get("pattern", 0))
                 for w in words]
        rng = random.Random(seed)
        perm = pairs[:]
        rng.shuffle(perm)
        return {w: {"root": r, "pattern": p}
                for w, (r, p) in zip(words, perm)}


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer, label_cache, cfg, split,
                 placebo=False, placebo_seed=0,
                 src_tokenizer=None, tgt_tokenizer=None):
        self.cfg = cfg
        self.src_tok = src_tokenizer or tokenizer
        self.tgt_tok = tgt_tokenizer or tokenizer
        if label_cache is None:
            self.entries = {}
        elif placebo:
            self.entries = label_cache.placebo_entries(placebo_seed)
        else:
            self.entries = label_cache.entries
        self.ignore = cfg["aux_ignore_index"]
        self.max_len = cfg["max_len"]
        self.items = []
        for src, tgt in pairs:
            src_ids, starts = encode_words(self.src_tok, src, cfg["bos_id"],
                                           cfg["eos_id"], self.max_len)
            tgt_ids, _ = encode_words(self.tgt_tok, tgt, cfg["bos_id"],
                                      cfg["eos_id"], self.max_len)
            if len(src_ids) < 3 or len(tgt_ids) < 3:
                continue
            self.items.append((src_ids, tgt_ids, src.split(), starts))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        src_ids, tgt_ids, words, starts = self.items[i]
        n = len(src_ids)
        root = torch.full((n,), self.ignore, dtype=torch.long)
        pat = torch.full((n,), self.ignore, dtype=torch.long)
        word_ids = word_ids_from_starts(starts, n)
        for wi, pos in enumerate(starts):
            if pos >= n:
                continue
            end = starts[wi + 1] if wi + 1 < len(starts) else n - 1
            end = min(end, n - 1)  # never label EOS
            e = self.entries.get(words[wi]) if wi < len(words) else None
            r, p = (int(e.get("root", 0)), int(e.get("pattern", 0))) if e else (0, 0)
            # FIX 2026-10-14: Map rootless/out-of-bank words (`r <= 0`) to 0 on
            # all real word spans (`pos:end`), leaving only BOS/EOS/PAD as -100.
            root[pos:end] = r  # preserves -1 (out-of-bank), -2 (unknown), 0 (rootless)
            pat[pos:end] = p if p >= 1 else 0
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long), root, pat, word_ids)


def collate(batch, pad_id):
    src, tgt, root, pat, wid = zip(*batch)
    src = pad_sequence(src, batch_first=True, padding_value=pad_id)
    tgt = pad_sequence(tgt, batch_first=True, padding_value=pad_id)
    root = pad_sequence(root, batch_first=True, padding_value=-100)
    pat = pad_sequence(pat, batch_first=True, padding_value=-100)
    wid = pad_sequence(wid, batch_first=True, padding_value=-1)
    return src, tgt, root, pat, wid


def make_loader(pairs, tokenizer, label_cache, cfg, split, shuffle,
                placebo=False, placebo_seed=0, num_workers=0,
                src_tokenizer=None, tgt_tokenizer=None):
    ds = PairDataset(pairs, tokenizer, label_cache, cfg, split,
                     placebo=placebo, placebo_seed=placebo_seed,
                     src_tokenizer=src_tokenizer, tgt_tokenizer=tgt_tokenizer)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=shuffle,
                      collate_fn=lambda b: collate(b, cfg["pad_id"]),
                      num_workers=num_workers,
                      generator=torch.Generator().manual_seed(cfg["seed"]))
