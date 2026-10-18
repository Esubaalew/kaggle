#!/usr/bin/env python3
"""Build cleaned, geez-dataset-verified morphological label caches for drfn_fix and drfn_nllb_fix.

Fixes five defects in the previous label builder:
1. Rootless words (all entries in `closed_class_words_without_root.tsv`, all `NO_ROOT` /
   particle / preposition / pronoun / demonstrative / numeral / adverb rows in
   `geez_morphology_rows.jsonl`, cliticized function words, biblical proper names, and
   non-Ethiopic / punctuation tokens) are explicitly assigned `root = 0` (`<none>`) instead
   of receiving hallucinated ByT5 verb/noun roots (such as `ውስተ -> ወ-ሰ-ተ`, `ከመ -> ከ-መ-መ`,
   `እግዚአብሔር -> በ-ሐ-ረ`, `፤ -> ፈ-የ-አ`).
2. Strict split hygiene: only `split == "train"` rooted rows from `geez_morphology_rows.jsonl`
   populate `gold_map`, preventing val/test gold or book annotations from leaking into dev probes.
3. Settled train forms take their curated root and pattern directly, and proclitic-absorbed
   `ይቤ / ወይቤ` forms unify under `በ-ሀ-ለ`. Remaining ByT5 readings in `analyser_cache.json`
   must pass `carries_root(word, root)` (ordered radical subsequence check allowing weak glides
   `ወ/የ` and geminate collapse, with >= 3 radicals); otherwise `root = 0` (`<none>`).
4. Pattern bank hygiene (`k_pattern = 31` -> 32 slots): normalizes malformed raw pattern strings
   (`basic_astem: aste_past -> aste_past`, `basic_basic_tense: past -> basic_past`), excludes
   rootless POS tags (`particle`, `preposition`, `pronoun`, `numeral`, `demonstrative`, `adverb`)
   from the morphological pattern inventory, and keeps all valid rooted morphological patterns
   instead of truncating at 16.
5. Out-of-bank rooted words (`rt is not None` but `rt not in rid`) are marked with `root = -1`
   (while retaining their valid `pattern >= 1`), instead of collapsing to `root = 0` (`<none>`).
   This prevents `gate_bce` from training the memory gate to close (`lambda_t -> 0`) on 41.9%
   of real Ge'ez verbs and nouns whose roots fall outside the top 255 slots.

Produces three label inventories in `out_dir`:
  - `labels_analyser.json`       (k_root = 255 + `<none>` = 256 slots, k_pattern = 31 + `<none>` = 32)
  - `labels_analyser_wide.json`  (k_root = 1023 + `<none>` = 1024 slots, k_pattern = 31 + `<none>` = 32)
  - `labels_analyser_full.json`  (k_root = 2047 + `<none>` = 2048 slots, k_pattern = 31 + `<none>` = 32)
and their corresponding training-derived Amharic target root maps:
  - `labels_tgt_analyser.json`
  - `labels_tgt_analyser_wide.json`
  - `labels_tgt_analyser_full.json`
"""
import collections
import csv
import json
import os
import shutil

SRC_DATA = "/Users/esubalew/is/researcher/kaggle/drfn_clean/pack/data"
DEC_DATA = "/Users/esubalew/is/researcher/kaggle/drfn_dec/pack/data"
AC_PATH = "/Users/esubalew/is/researcher/kaggle/run_drfn_labels/pack/runs/shared/analyser_cache.json"
UNI_PATH = "/Users/esubalew/is/researcher/geez-dataset-v1/data/unified/geez_morphology_rows.jsonl"
CC_PATH = "/Users/esubalew/is/researcher/geez-morphology-model/data/closed_class_words_without_root.tsv"

# Homophone folding matching dataset_loader / geez_script
HOMO = {2: 0, 16: 0, 4: 6, 26: 20, 40: 39}
WEAK = {(ord("ወ") - 0x1200) // 8, (ord("የ") - 0x1200) // 8}
ROOTLESS_CLASSES = ("particle", "preposition", "pronoun", "demonstrative", "numeral", "adverb")
PAT_NORM = {
    "basic_astem: aste_past": "aste_past",
    "astem: aste_past": "aste_past",
    "basic_basic_tense: past": "basic_past",
    "basic_tense: past": "basic_past",
}


def bases(s):
    out = []
    for ch in s:
        cp = ord(ch)
        if 0x1200 <= cp <= 0x135A:
            b = (cp - 0x1200) // 8
            out.append(HOMO.get(b, b))
    return out


def contains_seq(word_bases, sig):
    if not sig:
        return False
    i = 0
    for b in word_bases:
        if i < len(sig) and b == sig[i]:
            i += 1
    return i == len(sig)


def carries_root(word, root):
    w = bases(word)
    r = bases(root.replace("-", ""))
    if len(r) < 3:
        return False
    at, prev = 0, None
    for rad in r:
        if rad in w[at:]:
            at = w.index(rad, at) + 1
        elif rad not in WEAK and rad != prev:
            return False
        prev = rad
    return True


def load_pairs(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2 and p[0].strip() and p[1].strip():
                out.append((p[0].strip(), p[1].strip()))
    return out


def normalize_pattern(pat):
    if not pat:
        return None
    pat = PAT_NORM.get(pat, pat)
    if pat in ROOTLESS_CLASSES:
        return None
    return pat


def build_clean_readings():
    cc_rootless = set()
    with open(CC_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cc_rootless.add(row["word"].strip())

    uni_rootless = set()
    gold_map = {}
    with open(UNI_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            w = r["form"]
            wc = r.get("word_class")
            if (r.get("status") == "NO_ROOT"
                    or wc in ROOTLESS_CLASSES
                    or r.get("root") in (None, "none")):
                uni_rootless.add(w)
            elif r.get("split") == "train" and not r.get("status") and r.get("root"):
                st = r.get("stem") or "basic"
                tn = r.get("tense") or "notense"
                pat = f"{st}_{tn}" if wc == "verb" else (wc or "noun")
                gold_map[w] = (r["root"], normalize_pattern(pat) or "noun")

    extra_rootless = {
        "ውስተ", "ውስቴቱ", "ውስቴቶሙ", "ውስቴታ", "በውስተ", "እምውስተ",
        "እንተ", "በእንተ", "በእንቲኣሁ", "በእንቲኣሆሙ", "በእንቲኣየ", "በእንቲኣከ", "በእንቲኣክሙ", "በእንቲኣነ",
        "ላዕለ", "ላዕሌሁ", "ላዕሌሆሙ", "ላዕሌየ", "ላዕሌከ", "ላዕሌክሙ", "ላዕሌነ", "ላዕሌሃ", "በላዕለ", "እምላዕለ",
        "ታሕተ", "ታሕቴሁ", "ማእከለ", "ማእከሌሆሙ", "ማእከሌክሙ", "ቅድመ", "ቅድሜሁ", "ቅድሜሆሙ", "ቅድሜየ", "ቅድሜከ", "በቅድመ", "እምቅድመ",
        "ድኅረ", "ድኅሬሁ", "ድኅሬሆሙ", "እምድኅረ", "ኀበ", "ኀቤሁ", "ኀቤሆሙ", "ኀቤየ", "ኀቤከ", "ኀቤክሙ", "እምኀበ",
        "ምስለ", "ምስሌሁ", "ምስሌሆሙ", "ምስሌየ", "ምስሌከ", "ምስሌክሙ", "ምስሌነ",
        "እምነ", "እምኔሁ", "እምኔሆሙ", "እምኔየ", "እምኔከ", "እምኔክሙ",
        "ከመ", "በከመ", "ዘከመ", "እስመ", "እስከ", "እንዘ", "ሶበ", "አመ", "እመ", "አላ", "እፎ", "ናሁ", "ህየ", "ዮም",
        "እለ", "ዘ", "እም", "ለ", "በ", "ወ", "ሂ", "ኒ", "ሰ", "ኬ", "ኑ", "ሁ",
        "አነ", "አንተ", "አንቲ", "ውእቱ", "ይእቲ", "ንሕነ", "አንትሙ", "አንትን", "ውእቶሙ", "ውእቶን", "እሙንቱ", "እማንቱ",
        "ዝ", "ዝንቱ", "ዛቲ", "እሉ", "እሎንቱ", "ዝኩ", "እልክቱ", "ኩሉ", "ኵሉ", "ኵሎ", "ኵሎሙ", "ለኵሉ", "ወኵሉ",
        "ሎቱ", "ሎሙ", "ላቲ", "ሊተ", "ለከ", "ለክሙ", "ለነ", "ቦቱ", "ቦሙ", "ባቲ", "ኪያሁ", "ኪያሆሙ", "ኪያየ", "ኪያከ", "ኪያክሙ",
        "እግዚአብሔር", "እግዚእ", "ኢየሱስ", "ክርስቶስ", "እስራኤል", "ኢየሩሳሌም", "ዳዊት", "ሙሴ", "አብርሃም", "ያዕቆብ",
        "ዮሴፍ", "ጳውሎስ", "ጴጥሮስ", "ዮሐንስ", "ማርያም", "ፈርዖን", "ግብጽ", "ባቢሎን", "ሰሎሞን", "ሳኦል", "አሮን",
    }
    all_rootless = cc_rootless | uni_rootless | extra_rootless

    def is_rootless(w):
        if not any(0x1200 <= ord(c) <= 0x135A for c in w):
            return True
        if w in all_rootless and w not in gold_map:
            return True
        for pref in ("ወ", "ዘ", "ለ", "በ", "እም", "ወለ", "ወበ", "ወእም", "ዘበ", "ዘእም"):
            if w.startswith(pref) and len(w) > len(pref) + 1:
                rem = w[len(pref):]
                if rem in all_rootless and rem not in gold_map:
                    return True
                for suf in ("ኒ", "ሰ", "ኬ", "ሂ", "ኑ"):
                    if rem.endswith(suf) and rem[:-len(suf)] in all_rootless:
                        return True
        for suf in ("ኒ", "ሰ", "ኬ", "ሂ", "ኑ"):
            if w.endswith(suf) and w[:-len(suf)] in all_rootless and w not in gold_map:
                return True
        return False

    raw_readings = json.load(open(AC_PATH, encoding="utf-8"))["readings"]
    clean = {}
    for w, r in raw_readings.items():
        if is_rootless(w):
            clean[w] = (None, None)
            continue
        if w in gold_map:
            clean[w] = gold_map[w]
            continue
        if not r:
            clean[w] = (None, None)
            continue
        rt = r.get("root")
        t = r.get("type")
        if t in ROOTLESS_CLASSES:
            clean[w] = (None, None)
            continue
        st = r.get("stem") or "basic"
        tn = r.get("tense") or "notense"
        raw_pat = f"{st}_{tn}" if t in (None, "verb") and (r.get("stem") or r.get("tense")) else t
        pat = normalize_pattern(raw_pat)
        # Unify irregular ይቤ / ወይቤ / ትቤ / እቤ / ንቤ forms under በ-ሀ-ለ
        if rt in ("ወ-በ-ለ", "ወ-የ-በ-ለ", "የ-በ-ለ") and any(s in w for s in ("ይቤ", "ትቤ", "እቤ", "ንቤ")):
            rt = "በ-ሀ-ለ"
            if not pat:
                pat = "basic_past"
        if rt and (rt == "በ-ሀ-ለ" or carries_root(w, rt)):
            clean[w] = (rt, pat or "noun")
        else:
            clean[w] = (None, None)
    return clean


def build_cache_dict(train_words, all_words, clean_map, k_root=255, k_pattern=31):
    rc, pc = collections.Counter(), collections.Counter()
    for w in train_words:
        rt, pat = clean_map.get(w, (None, None))
        if rt:
            rc[rt] += 1
        if pat:
            pc[pat] += 1
    roots = [r for r, _ in rc.most_common(k_root)]
    pats = [p for p, _ in pc.most_common(k_pattern)]
    rid = {r: i + 1 for i, r in enumerate(roots)}
    pid = {p: i + 1 for i, p in enumerate(pats)}

    entries = {}
    for w in all_words:
        if not any(0x1200 <= ord(c) <= 0x135A for c in w):
            entries[w] = {"root": 0, "pattern": 0}
            continue
        rt, pat = clean_map.get(w, (None, None))
        # Root encoding:
        #   >= 1 : in-bank root slot k (supervises routing CE + value CE + gate target = 1)
        #     -1 : verified rooted word whose root is outside top-K bank (excluded from
        #          1..K root CE and NOT forced to gate=0; still gets pattern supervision!)
        #      0 : explicitly rootless word (gate target = 0)
        if rt is None:
            r_val = 0
        else:
            r_val = rid.get(rt, -1)
        p_val = pid.get(pat, 0) if pat else 0
        entries[w] = {"root": r_val, "pattern": p_val}

    sup_r = sum(1 for w in train_words if entries.get(w, {}).get("root", 0) >= 1)
    oob_r = sum(1 for w in train_words if entries.get(w, {}).get("root", 0) == -1)
    zero_r = sum(1 for w in train_words if entries.get(w, {}).get("root", 0) == 0)
    sup_p = sum(1 for w in train_words if entries.get(w, {}).get("pattern", 0) >= 1)
    return {
        "version": 3,
        "analyzer": "geez-dataset train-only unified + consonant-verified ByT5 analyser (oob-safe gate)",
        "provisional": False,
        "root_classes": ["<none>"] + roots,
        "pattern_classes": ["<none>"] + pats,
        "entries": entries,
        "metadata": {
            "root_slots": len(roots) + 1,
            "pattern_slots": len(pats) + 1,
            "distinct_train_roots_total": len(rc),
            "distinct_train_patterns_total": len(pc),
            "train_tokens": len(train_words),
            "tokens_root_in_bank": sup_r,
            "tokens_root_oob_minus1": oob_r,
            "tokens_root_null_slot0": zero_r,
            "tokens_pattern_in_bank": sup_p,
        },
    }


def build_tgt_map(train_pairs, src_entries, root_classes, min_pair=3, min_dice=0.18, min_cond=0.5):
    """Build Amharic word -> Ge'ez root class map strictly from training pairs,
    using anchor co-occurrence + 3-consonant stem signature propagation."""
    src_cnt = collections.Counter()
    tgt_cnt = collections.Counter()
    cooc = collections.Counter()
    for src, tgt in train_pairs:
        s_roots = {src_entries.get(w, {}).get("root", 0) for w in src.split()}
        s_roots = {r for r in s_roots if r >= 1}
        t_words = {w for w in tgt.split() if any(0x1200 <= ord(c) <= 0x135A for c in w)}
        for r in s_roots:
            src_cnt[r] += 1
        for w in t_words:
            tgt_cnt[w] += 1
            for r in s_roots:
                cooc[(w, r)] += 1

    # Step 1: Direct high-precision anchor words
    anchors = {}
    by_root = collections.defaultdict(list)
    for (w, r), c in cooc.items():
        if c < min_pair:
            continue
        cond = c / tgt_cnt[w]
        dice = (2.0 * c) / (tgt_cnt[w] + src_cnt[r])
        if cond >= min_cond and dice >= min_dice:
            prev = anchors.get(w)
            if prev is None or dice > prev[1]:
                anchors[w] = (r, dice)
            by_root[r].append((dice, w))

    # Step 2: 3-consonant signature per root (from top anchor word, or cognate Ge'ez root)
    signatures = {}
    for r in range(1, len(root_classes)):
        if by_root[r]:
            by_root[r].sort(reverse=True)
            for _, aw in by_root[r]:
                wb = bases(aw)
                if len(wb) >= 3:
                    signatures[str(r)] = wb[:3]
                    break
        if str(r) not in signatures:
            rb = [b for b in bases(root_classes[r].replace("-", "")) if b not in WEAK]
            if len(rb) >= 3:
                signatures[str(r)] = rb[:3]

    # Step 3: Propagate to all training Amharic words that carry signature[r] AND co-occur with r
    final_map = {w: r for w, (r, _) in anchors.items()}
    sig_to_roots = collections.defaultdict(list)
    for r_str, sig in signatures.items():
        sig_to_roots[tuple(sig)].append(int(r_str))

    for w in tgt_cnt:
        if w in final_map:
            continue
        wb = bases(w)
        if len(wb) < 3:
            continue
        best_r, best_c = None, 0
        for sig, r_list in sig_to_roots.items():
            if contains_seq(wb, sig):
                for r in r_list:
                    c = cooc.get((w, r), 0)
                    if c > best_c:
                        best_c = c
                        best_r = r
        if best_r is not None and best_c >= 1:
            final_map[w] = best_r

    return {
        "map": final_map,
        "signature": signatures,
        "root_classes": root_classes,
        "report": {
            "source_classes_with_signature": len(signatures),
            "source_classes_seen": len(src_cnt),
            "amharic_types_labelled": len(final_map),
            "amharic_types_total": len(tgt_cnt),
            "thresholds": {"min_pair": min_pair, "min_dice": min_dice, "min_cond": min_cond},
        },
    }


def main():
    train_pairs = load_pairs(os.path.join(SRC_DATA, "age_train.tsv"))
    dev_pairs = load_pairs(os.path.join(SRC_DATA, "age_dev.tsv"))
    train_words = [w for s, _ in train_pairs for w in s.split()]
    dev_words = [w for s, _ in dev_pairs for w in s.split()]
    all_words = sorted(set(train_words) | set(dev_words))

    clean_map = build_clean_readings()

    for out_dir in (
        "/Users/esubalew/is/researcher/kaggle/drfn_fix/pack/data",
        "/Users/esubalew/is/researcher/kaggle/drfn_nllb_fix/pack/data",
    ):
        os.makedirs(out_dir, exist_ok=True)
        for fn in ("age_train.tsv", "age_dev.tsv", "tokenizer.json", "labels_geezrpa.json"):
            src_f = os.path.join(SRC_DATA, fn)
            if os.path.exists(src_f):
                shutil.copyfile(src_f, os.path.join(out_dir, fn))
        tgt_geez = os.path.join(DEC_DATA, "labels_tgt_geezrpa.json")
        if os.path.exists(tgt_geez):
            shutil.copyfile(tgt_geez, os.path.join(out_dir, "labels_tgt_geezrpa.json"))

        for suffix, k_root in (("", 255), ("_wide", 1023), ("_full", 2047)):
            c = build_cache_dict(train_words, all_words, clean_map, k_root=k_root, k_pattern=31)
            with open(os.path.join(out_dir, f"labels_analyser{suffix}.json"), "w", encoding="utf-8") as f:
                json.dump(c, f, ensure_ascii=False)
            t = build_tgt_map(train_pairs, c["entries"], c["root_classes"])
            with open(os.path.join(out_dir, f"labels_tgt_analyser{suffix}.json"), "w", encoding="utf-8") as f:
                json.dump(t, f, ensure_ascii=False)
            if out_dir.endswith("drfn_fix/pack/data"):
                print(f"=== analyser{suffix} (k_root={k_root}) ===")
                print("  metadata:", c["metadata"])
                print("  pattern_classes:", c["pattern_classes"])
                print("  tgt report:", t["report"])


if __name__ == "__main__":
    raise SystemExit("Historical output paths protected. Use CMM/research_20261017/translation/g2/build_labels.py with explicit --dataset, --model-repo, --analyser-cache and --data-dir arguments; see RESEARCH_PLAN.md.")
