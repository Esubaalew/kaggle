"""Replace the pattern classes of a G3 label cache with morphological feature bundles.

A bundle is (part of speech, stem, tense, person, gender, number) from the corrected run-A reading of a
word. Bundles are given to content words (verbs, nouns, adjectives, nominals, infinitives, numerals) whatever
their root status, so words whose root is out of the root bank or unknown still get a feature label.
Rootless words (function words, names) keep pattern 0. The K-1 most frequent bundles over AGE training
tokens get their own class; rarer bundles get 0.

Where geez-dataset lists a form only as a noun, adjective or nominal (rows not EXCLUDED, any other status)
and run-A reads it as a verb, the dataset class is used. This corrects a systematic run-A error: construct
and accusative nouns in -ä (ቃለ, ምድረ, ሕዝበ, መልአከ) read as 3ms perfects, and bare nouns (ሕዝብ, ብእሲ) read
as imperatives.

Root labels, root classes and the target-side map are copied unchanged.

Usage: python build_feature_labels.py DATA_DIR READINGS_JSON DATASET_ROWS_JSONL [K=128]
Writes DATA_DIR/labels_analyser_{wide,full}_feat.json and labels_tgt_analyser_{wide,full}_feat.json.
"""
import collections
import json
import shutil
import sys
from pathlib import Path

CONTENT = {"verb", "noun", "adjective", "nominal", "infinitive", "numeral", "gerund", "verbal_noun"}


def bundle(r):
    if not r:
        return None
    pos = r.get("type") or r.get("pos")
    if pos in (None, "verb") and (r.get("tense") or r.get("stem")):
        pos = "verb"
    if pos not in CONTENT:
        return None
    if pos == "verb":
        return "verb|" + "|".join(str(r.get(k) or "-") for k in ("stem", "tense", "person", "gender", "number"))
    return pos + "|" + "|".join(str(r.get(k) or "-") for k in ("gender", "number"))


def dataset_nominal_overrides(rows_path):
    classes, first = collections.defaultdict(set), {}
    for line in open(rows_path, encoding="utf-8"):
        r = json.loads(line)
        if r.get("status") == "EXCLUDED":
            continue
        classes[r["form"]].add(r.get("word_class"))
        first.setdefault(r["form"], r)
    return {w: first[w] for w, c in classes.items() if c and c <= {"noun", "adjective", "nominal"}}


def main(data_dir, readings_path, rows_path, k=128):
    data_dir = Path(data_dir)
    readings = json.load(open(readings_path, encoding="utf-8"))["readings"]
    nominal = dataset_nominal_overrides(rows_path)
    n_over = 0
    for w, row in nominal.items():
        r = readings.get(w)
        if r and (r.get("pos") == "verb" or (r.get("type") is None and r.get("tense"))):
            readings[w] = dict(r, pos=row["word_class"], type=row["word_class"], stem=None, tense=None,
                               person=None, gender=row.get("gender"), number=row.get("number"))
            n_over += 1
    print(f"dataset word class overrides run-A verb reading for {n_over} word types")
    train_tokens = collections.Counter()
    for line in open(data_dir / "age_train.tsv", encoding="utf-8"):
        train_tokens.update(line.split("\t")[0].split())
    for suffix in ("wide", "full"):
        cache = json.load(open(data_dir / f"labels_analyser_{suffix}.json", encoding="utf-8"))
        counts = collections.Counter()
        word_bundle = {}
        for w, e in cache["entries"].items():
            b = None if e.get("root_status") == "rootless" else bundle(readings.get(w))
            word_bundle[w] = b
            if b:
                counts[b] += train_tokens.get(w, 0)
        classes = ["<none>"] + [b for b, c in counts.most_common(k - 1) if c > 0]
        bid = {b: i for i, b in enumerate(classes)}
        n_lab = n_tok = 0
        for w, e in cache["entries"].items():
            e["pattern"] = bid.get(word_bundle[w], 0)
            if e["pattern"]:
                n_lab += 1
                n_tok += train_tokens.get(w, 0)
        cache["pattern_classes"] = classes
        cache["metadata"] = dict(cache.get("metadata", {}), pattern_kind="feature_bundle",
                                 feature_bundle_slots=len(classes), train_tokens_with_bundle=n_tok,
                                 distinct_train_bundles=len(counts))
        out = data_dir / f"labels_analyser_{suffix}_feat.json"
        out.write_text(json.dumps(cache, ensure_ascii=False))
        shutil.copy(data_dir / f"labels_tgt_analyser_{suffix}.json", data_dir / f"labels_tgt_analyser_{suffix}_feat.json")
        tot = sum(train_tokens.values())
        print(f"{suffix}: {len(classes)} bundle classes from {len(counts)} bundles; "
              f"{n_tok} of {tot} train tokens ({100 * n_tok / tot:.1f}%) carry a bundle label")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 128)
