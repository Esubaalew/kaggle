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
