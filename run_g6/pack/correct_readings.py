"""Correct run-A analyser readings of AGE word types before label building.

Three corrections, applied in this order (curated dataset rows still take
precedence later, inside build_labels.py):

1. Function words (FUNCTION_WORDS, drafted from standard Ge'ez grammar; needs
   native review) and their clitic-extended forms -> rootless.
2. Proper names detected automatically (names_auto.json): the clitic-stripped
   form reappears in the paired Amharic sentence in >= 50% of its occurrences,
   its run-A root is not a training-lexicon root, the form is not a dataset
   form, and the root has a small word family (< 4 distinct bases) or >= 4
   radicals -> rootless ("name").
3. Proclitics (ወ, ዘ, ለ, በ, እም and chains). Run-A often counts a proclitic as a
   root radical (ለሕዝብ -> ለ-ሐ-ዘ-በ). When the word starts with a proclitic whose
   consonant equals the first radical of the full-word root, the root is not a
   training-lexicon root, and the remainder's root shares >= 2 radicals in order
   with the rest of the full-word root, or the remainder is rootless, the
   remainder's reading is used. The remainder must be attested as a standalone
   AGE word, unless the full-word root has >= 4 radicals.

Usage: python correct_readings.py RUN_A_CACHE REMAINDER_JSONL NAMES_JSON LEXICON_ROWS OUT_JSON AGE_TYPES_JSON
"""
import json, sys, collections

PROCLITICS = ["ወእም", "ወዘ", "ወለ", "ወበ", "ዘእም", "ዘኢ", "ወኢ", "እም", "ወ", "ዘ", "ለ", "በ"]
CONS = {"ወ": "ወ", "ዘ": "ዘ", "ለ": "ለ", "በ": "በ", "እ": "አ"}
ENCLITICS = ["ሰ", "ኒ", "ሂ", "ኑ", "ኬ"]

FUNCTION_WORDS = {
    "pronoun": "አነ አንተ አንቲ ውእቱ ይእቲ ንሕነ አንትሙ አንትን ውእቶሙ ውእቶን ውእተ ይእተ ኪያየ ኪያከ ኪያኪ ኪያሁ ኪያሃ ኪያነ ኪያክሙ ኪያሆሙ ኪያሆን ባሕቲትየ ባሕቲቱ".split(),
    "demonstrative": "ዝ ዝንቱ ዛ ዛቲ ዝኩ ዝክቱ እሉ እሎንቱ እላ እማንቱ እልክቱ ዘንተ ዛተ ዘኩ ዝየ ህየ ከመዝ ከማሁ ከማሃ ከማሆሙ ከማየ ከማከ".split(),
    "preposition": "ውስተ ላዕለ ዲበ ታሕተ ምስለ እምነ ኀበ በእንተ መንገለ እስከ ማእከለ ጥቃ እንበለ ዘእንበለ አምሳለ ቅድመ ድኅረ".split(),
    "conjunction": "እስመ ከመ እንዘ አመ ሶበ እመ እመሰ ባሕቱ ዳእሙ አው ወሚመ ሚመ".split(),
    "adverb": "እንከ ጥቀ ዓዲ ይእዜ ዮም ጌሠመ ትማልም እምዝ እምድኅረዝ ናሁ እፎ ዝየ በበ አኮ አኮኑ".split(),
    "particle": "ቦ አልቦ ኢ ምንት ምንተ መኑ አይቴ ማእዜ ለምንት ኵሉ ኵሎ ኵላ ኵሎሙ ኵሎን".split(),
}


def strip_proclitic(w):
    for p in PROCLITICS:
        if w.startswith(p) and len(w) - len(p) >= 2:
            return p, w[len(p):]
    return None, w


def overlap_in_order(a, b):
    """Length of the longest common subsequence of two radical lists."""
    m = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for x in range(len(a)):
        for y in range(len(b)):
            m[x + 1][y + 1] = m[x][y] + 1 if a[x] == b[y] else max(m[x][y + 1], m[x + 1][y])
    return m[-1][-1]


def main(run_a, rem_jsonl, names_json, lex_rows, out_path, types_json):
    R = json.load(open(run_a, encoding="utf-8"))
    readings = R["readings"]
    age_types = set(json.load(open(types_json, encoding="utf-8")))
    rem = {}
    for line in open(rem_jsonl, encoding="utf-8"):
        x = json.loads(line)
        f = dict(p.split(":", 1) for p in x["analysis"].split("|") if ":" in p)
        f = {k.strip(): v.strip() for k, v in f.items()}
        rt = None if f.get("root", "none") == "none" else f.get("root")
        rem[x["word"]] = {"root": rt, "type": f.get("type"), "stem": f.get("stem"), "tense": f.get("tense"),
                          "person": f.get("person"), "gender": f.get("gender"), "number": f.get("number"),
                          "object": f.get("object"), "pos": x["pos"].replace("type:", "").strip()}
    lexicon = set()
    for line in open(lex_rows, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "train" and r.get("root"):
            lexicon.add(r["root"])
    names = set(json.load(open(names_json, encoding="utf-8")))
    fw = {w: cls for cls, ws in FUNCTION_WORDS.items() for w in ws}

    def lookup(w):
        return readings.get(w) or rem.get(w)

    def function_class(w):
        if w in fw:
            return fw[w]
        b = w
        for s in ENCLITICS:
            if b.endswith(s) and b[:-len(s)] in fw:
                return fw[b[:-len(s)]]
        p, rest = strip_proclitic(w)
        if p:
            return function_class(rest)
        return None

    # curated rootless forms from the dataset (closed-class list + NO_ROOT rows)
    import csv, os
    cc_path = os.path.join(os.path.dirname(lex_rows), "../../../geez-morphology-model/data/closed_class_words_without_root.tsv")
    curated_rootless = set()
    if os.path.exists(cc_path):
        curated_rootless |= {r["word"].strip() for r in csv.DictReader(open(cc_path, encoding="utf-8"), delimiter="\t")}
    for line in open(lex_rows, encoding="utf-8"):
        r = json.loads(line)
        if r.get("split") == "train" and not r.get("status") and r.get("word_class") in ("particle", "preposition", "pronoun", "demonstrative", "conjunction"):
            curated_rootless.add(r["form"])
    freq = collections.Counter(json.load(open(types_json + ".freq", encoding="utf-8"))) if os.path.exists(types_json + ".freq") else collections.Counter()

    stats = collections.Counter()
    memo = {}

    def correct(w, depth=0):
        if w in memo:
            return memo[w]
        r = lookup(w)
        fc = function_class(w)
        if fc:
            out, tag = {"root": None, "type": fc, "pos": fc}, "function_word"
        elif w in names or w in curated_rootless:
            out, tag = {"root": None, "type": "name", "pos": "name"}, "name"
        else:
            out, tag = r, "unchanged"
            p, rest = strip_proclitic(w)
            if p and depth < 3 and (rest in readings or rest in rem):
                rr, _ = correct(rest, depth + 1)
                if rr is not None and rr.get("root") is None and (rr.get("type") in ("name",) or function_class(rest) or rest in names or rest in curated_rootless):
                    out, tag = dict(rr, clitic=p), "clitic_to_rootless"
                elif r and r.get("root") and rr is not None:
                    rw = r["root"].split("-")
                    attested = rest in age_types
                    rr_root = rr.get("root")
                    if rw[0] == CONS.get(p[0]) and (attested or len(rw) >= 4):
                        if r["root"] not in lexicon:
                            ok = rr_root is None or (rr_root != r["root"] and overlap_in_order(rw[1:], rr_root.split("-")) >= 2)
                        else:   # both readings are real roots: prefer the remainder only if it is the more frequent word
                            ok = bool(rr_root) and len(rr_root.split("-")) >= 3 and attested and freq[rest] >= max(1, freq[w])
                        if ok:
                            out, tag = dict(rr, clitic=p), ("clitic_rerouted" if rr_root else "clitic_to_rootless")
        memo[w] = (out, tag)
        return out, tag

    new = {}
    for w in readings:
        out, tag = correct(w)
        new[w] = out
        stats[tag] += 1
    meta = dict(R["meta"], corrections=dict(stats), function_words=len(fw), names=len(names))
    json.dump({"meta": meta, "readings": new}, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    print(json.dumps(dict(stats)))


if __name__ == "__main__":
    main(*sys.argv[1:7])
