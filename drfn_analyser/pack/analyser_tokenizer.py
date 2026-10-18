"""Ge'ez root analyser (ByT5 morphology model) readings, used two ways in the drfn trainer.

1. Root and pattern LABELS for the auxiliary heads (build_label_cache): the same file format and the same size as the
   geezRPA k256 cache of advisor-update-2019 (top 255 roots + <none>, top 16 patterns + <none>), so the only change
   in a run is who made the labels.
2. Root as an extra INPUT token after each word's BPE pieces (BpeRootTokenizer, --src-tokenizer bpe_root). The word's
   spelling stays (names still copy through BPE); one token per word adds the analyser's root class.

The readings come from build_analyser_cache.py (the ByT5 model reads every distinct Ge'ez word once).
A reading is {"root": "ቀ-ተ-ለ" or None, "type": ..., "stem", "tense", "person", "gender", "number", "object"};
a word the analyser could not read at all has no reading (None).
"""
import json
from collections import Counter

VERB_TYPES = (None, "verb")


def pattern_of(reading):
    """Pattern class of one reading: <stem>_<tense> for a finite verb, the word class otherwise
    (noun, adjective, infinitive, particle, preposition, pronoun, ...); None when there is nothing to say."""
    if not reading:
        return None
    t = reading.get("type")
    if t in VERB_TYPES:
        if reading.get("stem") or reading.get("tense"):
            return f"{reading.get('stem') or 'nostem'}_{reading.get('tense') or 'notense'}"
        return None                      # root-only reply: no pattern
    return t


def build_label_cache(train_words, readings, k_root=255, k_pattern=16, model_note=""):
    """LabelCache JSON (data.LabelCache format). Ranks by TRAIN token frequency, like the k256 cache.
    Ids 1..k are classes; id 0 is <none>; a root or pattern outside the top k is -100 (ignored), not forced."""
    rc, pc = Counter(), Counter()
    for w in train_words:
        r = readings.get(w)
        if r and r.get("root"):
            rc[r["root"]] += 1
        p = pattern_of(r)
        if p:
            pc[p] += 1
    roots = [r for r, _ in rc.most_common(k_root)]
    pats = [p for p, _ in pc.most_common(k_pattern)]
    rid = {r: i + 1 for i, r in enumerate(roots)}
    pid = {p: i + 1 for i, p in enumerate(pats)}
    entries = {}
    for w, r in readings.items():
        if not r:
            continue
        e = {"root": rid.get(r.get("root"), -100) if r.get("root") else -100,
             "pattern": pid.get(pattern_of(r), -100) if pattern_of(r) else -100}
        if e["root"] != -100 or e["pattern"] != -100:
            entries[w] = e
    tok = len(train_words)
    sup_r = sum(1 for w in train_words if entries.get(w, {}).get("root", -100) > 0)
    sup_p = sum(1 for w in train_words if entries.get(w, {}).get("pattern", -100) > 0)
    return {"version": 1, "analyzer": "geez ByT5 root analyser " + model_note, "provisional": False,
            "root_classes": ["<none>"] + roots, "pattern_classes": ["<none>"] + pats, "entries": entries,
            "metadata": {"root_slots": k_root + 1, "pattern_slots": k_pattern + 1, "unmapped_label": -100,
                         "rank": "train Ge'ez tokens", "train_tokens": tok,
                         "tokens_root_supervised": sup_r, "tokens_pattern_supervised": sup_p,
                         "pattern_rule": "<stem>_<tense> for finite verbs, else the word class"}}


class _Encoded:
    def __init__(self, ids):
        self.ids = ids


class BpeRootTokenizer:
    """BPE pieces of the word, then one root token (id = base + root class) when the analyser gives the word one of
    the label cache's root classes. Same interface as the BPE tokenizer for data.encode_words: encode(w).ids."""

    def __init__(self, bpe, label_cache_path, base_size):
        self.bpe = bpe
        with open(label_cache_path, encoding="utf-8") as f:
            doc = json.load(f)
        self.base = base_size
        self.n_root = len(doc["root_classes"])
        self.root_of = {w: e["root"] for w, e in doc["entries"].items() if e.get("root", -100) > 0}

    def encode(self, word):
        ids = list(self.bpe.encode(word).ids)
        r = self.root_of.get(word)
        if r:
            ids.append(self.base + r)
        return _Encoded(ids)

    def __len__(self):
        return self.base + self.n_root


FEATURES = ("type", "stem", "tense", "person", "gender", "number", "object")


def reading_tokens(reading):
    """Two tokens per analysed word: the root, and one token for the whole morphology of that word.
    [root:ቀ-ተ-ለ] [m:stem=basic|tense=past|person=3|gender=m|number=s]   (fields the reading lacks are left out).
    One combined token keeps a sentence as short as BPE (a token per feature made 21% of training sentences longer
    than the 128-token limit)."""
    morph = "|".join(f"{f}={reading[f]}" for f in FEATURES if reading.get(f))
    return ["[root:%s]" % reading["root"], "[m:%s]" % (morph or "none")]


class AnalyserInputTokenizer:
    """The analyser as the Ge'ez-side tokenizer. A word with a reading whose root occurs at least `min_root_count` times
    in the training text enters as two tokens, root + morphology, and NOT as BPE pieces. Every other word (no reading,
    no root, a rare root) keeps its BPE pieces, so it can still be copied. BPE ids are 0..base-1; analyser tokens are
    base + index; a morphology token never seen in training maps to [UNK] (id 3).
    Same interface as the BPE tokenizer for data.encode_words: encode(w).ids."""

    def __init__(self, bpe, readings, train_words, base_size, min_root_count=2):
        self.bpe, self.readings, self.base = bpe, readings, base_size
        rc = Counter(readings[w]["root"] for w in train_words if readings.get(w) and readings[w].get("root"))
        self.roots = {r for r, n in rc.items() if n >= min_root_count}
        toks = Counter()
        for w in train_words:
            if self._analysed(w):
                toks.update(reading_tokens(readings[w]))
        self.vocab = {t: base_size + i for i, (t, _) in enumerate(sorted(toks.items(), key=lambda x: (-x[1], x[0])))}
        self.n_root_tokens = sum(1 for t in self.vocab if t.startswith("[root:"))
        self.n_morph_tokens = len(self.vocab) - self.n_root_tokens
        self.n_analysed = sum(1 for w in train_words if self._analysed(w))
        self.n_words = len(train_words)

    def _analysed(self, w):
        # Only words made of Ethiopic letters (U+1200-U+135A); punctuation and numerals always stay BPE.
        if not all("\u1200" <= ch <= "\u135a" for ch in w):
            return False
        r = self.readings.get(w)
        return bool(r and r.get("root") in self.roots)

    def tokens(self, word):
        """Token strings, for inspection."""
        if self._analysed(word):
            return reading_tokens(self.readings[word])
        return list(self.bpe.encode(word).tokens)

    def encode(self, word):
        if self._analysed(word):
            return _Encoded([self.vocab.get(t, 3) for t in reading_tokens(self.readings[word])])
        return _Encoded(list(self.bpe.encode(word).ids))

    def __len__(self):
        return self.base + len(self.vocab)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"base": self.base, "root_tokens": self.n_root_tokens, "morph_tokens": self.n_morph_tokens,
                       "analysed_train_tokens": self.n_analysed, "train_tokens": self.n_words,
                       "vocab": self.vocab}, f, ensure_ascii=False)
