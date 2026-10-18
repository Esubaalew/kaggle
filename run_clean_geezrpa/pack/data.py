"""Data pipeline: TSV pairs, shared 16k BPE, label cache, placebo labels.

Data files (~/workspace/thesis-data/age/):
    age_train.tsv / age_dev.tsv / age_test.tsv, Ge'ez <TAB> Amharic.
    Lines without exactly 2 tab-separated fields are skipped (the 40
    monolingual Ge'ez lines); this yields the documented 13,856 / 1,732 /
    1,732 split. Hashes are recorded in run metadata.

Label cache schema (JSON), pointed to by --label-cache:
    {
      "version": 1,
      "analyzer": "geezRPA-repaired-2026-09-30",
      "provisional": false,            # true ONLY for smoke-test caches
      "root_classes": ["<none>", ...],  # id 0 = no label
      "pattern_classes": ["<none>", ...],
      "entries": {"word": {"root": rid, "pattern": pid}}
    }
Word-level labels are aligned to BPE subwords: the first subword of each
word carries the label, continuation subwords get aux_ignore_index (-100).

Placebo labels (bpe_placebo, cmm_placebo): the multiset of (root, pattern)
pairs from the cache is randomly permuted across cached words with a seeded
RNG, preserving the joint marginal distribution while breaking the
word -> label correspondence. This is the "matched fake labels" control.
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

    # FIX 2026-10-12: keep word boundaries. Each word is encoded with a
    # leading '▁', and the matching decoder turns it back into a space.
    # The previous Whitespace pre-tokenizer kept no boundary, so decoded
    # output split every BPE piece into its own "word".
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
    """Word index per source position; -1 at BOS/EOS and past truncation.

    PATCH 2026-10-12. Same span arithmetic as the label alignment in
    PairDataset.__getitem__, so a word's BPE pieces, its root label and its
    routing unit are the same set of positions by construction.
    """
    wid = torch.full((n,), -1, dtype=torch.long)
    for k, pos in enumerate(starts):
        if pos >= n:
            continue
        end = starts[k + 1] if k + 1 < len(starts) else n - 1
        end = min(end, n - 1)                      # never include EOS
        wid[pos:end] = k
    return wid


def segment_word(stemmer, word):
    """Split a Ge'ez word into surface morphemes: prefixes + stem + suffixes.

    Uses GeezStemmer.extract_root(word); the token format is the analyzer's
    own segmentation (analysis["prefixes"], analysis["stem"],
    analysis["suffixes"]). Words the analyzer cannot segment (non_lexical,
    root None, or an exception) fall back to the whole word as one token.
    Deterministic: same word always yields the same token list.
    """
    try:
        res = stemmer.extract_root(word)
        a = res.get("analysis", {})
        if res.get("root") is None or a.get("method") == "non_lexical":
            return [word]
        toks = list(a.get("prefixes") or []) + [a.get("stem") or word] + \
            list(a.get("suffixes") or [])
        toks = [t for t in toks if t]
        return toks or [word]
    except Exception:
        return [word]


class _Encoded:
    """Minimal stand-in for tokenizers.Encoding: only .ids is used."""
    def __init__(self, ids):
        self.ids = ids


class MorphemeTokenizer:
    """Source-side tokenizer that emits geezRPA morpheme ids.

    encode(word).ids -> list of vocab ids for the word's morphemes
    (prefixes + stem + suffixes, see segment_word). Morphemes absent from
    the vocab map to [UNK]. The analyzer is loaded lazily so the default
    BPE path never depends on the repair workspace.
    """

    def __init__(self, vocab_path, repair_dir=None):
        with open(vocab_path, encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.unk_id = self.vocab.get("[UNK]")
        if self.unk_id is None:
            raise ValueError(f"{vocab_path} has no [UNK] token")
        import sys as _sys
        repair = repair_dir or os.environ.get(
            "ETHIOMORPH_REPAIR",
            os.path.expanduser("~/workspace/ethiomorph-repair"))
        _sys.path.insert(0, repair)
        from src.stemmer import GeezStemmer
        self.stemmer = GeezStemmer()
        self._inv = {i: t for t, i in self.vocab.items()}

    def encode(self, word):
        ids = [self.vocab.get(m, self.unk_id)
               for m in segment_word(self.stemmer, word)]
        return _Encoded(ids)

    def decode(self, ids):
        return " ".join(self._inv.get(i, "[UNK]") for i in ids)

    def __len__(self):
        return len(self.vocab)


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
        # Separate src/tgt tokenizers (task 2: morpheme src, BPE tgt).
        # When not given, both fall back to `tokenizer`: old behavior.
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
        word_ids = word_ids_from_starts(starts, n)          # PATCH
        for wi, pos in enumerate(starts):
            if pos >= n:
                continue
            # Report 2026-08-01: the word-level label is copied to ALL BPE
            # pieces overlapping the word, not just the first piece.
            end = starts[wi + 1] if wi + 1 < len(starts) else n - 1
            end = min(end, n - 1)  # never label EOS
            e = self.entries.get(words[wi]) if wi < len(words) else None
            r, p = (int(e.get("root", 0)), int(e.get("pattern", 0))) if e else (0, 0)
            if r:
                root[pos:end] = r
            if p:
                pat[pos:end] = p
        return (torch.tensor(src_ids, dtype=torch.long),
                torch.tensor(tgt_ids, dtype=torch.long), root, pat, word_ids)


def collate(batch, pad_id):
    src, tgt, root, pat, wid = zip(*batch)                  # PATCH
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
