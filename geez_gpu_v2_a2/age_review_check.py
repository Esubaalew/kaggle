"""Run A vs A2 on the 1,435 AGE forms of the 18 Oct review (geez-dataset 2.1.0, data/unified/review_2026-10-18.tsv),
plus a short list of forms the website analysed wrongly. Reference = the review's word class and root.
NOT a held-out test: forms whose review settled them (status '' in the reviewed rows) are in A2's training data.
They are reported separately from the forms that stayed held (not trained on by either model).
    python age_review_check.py <dataset_dir> <runA_best> <A2_best> <out_json>
"""
import csv, json, os, sys, collections, torch
sys.path.insert(0, os.environ.get("MODEL_CODE", "/Users/esubalew/is/researcher/geez-morphology-model"))
import train_tpu as tt
from transformers import AutoModelForSeq2SeqLM
D, MA, MA2, OUT = sys.argv[1:5]
rev = list(csv.DictReader(open(os.path.join(D, "data/unified/review_2026-10-18.tsv"), encoding="utf-8"), delimiter="\t"))
status = {}
for l in open(os.path.join(D, "data/unified/geez_morphology_rows_reviewed.jsonl"), encoding="utf-8"):
    r = json.loads(l); status[r["id"]] = (r["status"], r["split"])
EXTRA = ["ሕዝብ", "ለሕዝብ", "ሕዝበ", "ቃለ", "ቃል", "ምድረ", "ምድር", "ብእሲ", "ሀገር", "ዓለም", "ኵሉ", "ናሁ", "እለ", "ከመ",
         "እስመ", "ይቤ", "ወይቤ", "ወይቤሎ", "እግዚአብሔር", "ሙሴ", "ወሙሴ", "ዳዊት", "ኢየሩሳሌም", "ንጉሥ", "ንጉሠ", "ወንጉሥ"]
forms = sorted({r["form"] for r in rev} | set(EXTRA))
torch.set_num_threads(os.cpu_count())
def run(model_dir):
    m = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.float32).eval(); out = {}
    ins = [f"{t}: {f}" for f in forms for t in ("analyze", "pos")]
    order = sorted(range(len(ins)), key=lambda i: len(ins[i].encode()))
    with torch.no_grad():
        for k in range(0, len(order), 48):
            idx = order[k:k + 48]; a = max(tt.bucket(len(ins[i].encode()) + 1, tt.INPUT_BUCKETS) for i in idx)
            x = torch.full((len(idx), a), tt.PAD, dtype=torch.long)
            for r, i in enumerate(idx):
                e = tt.encode(ins[i], a); x[r, :len(e)] = torch.tensor(e)
            o = m.generate(input_ids=x, attention_mask=(x != tt.PAD).long(), max_new_tokens=129, do_sample=False, num_beams=1)
            for i, s in zip(idx, o): out[ins[i]] = tt.decode(s[1:].tolist())
    return out
def root_of(t):
    h = t.split("|")[0].strip()
    return h[5:].strip() if h.startswith("root:") else ""
def type_of(t):
    for part in t.split("|"):
        if part.strip().startswith("type:"): return part.strip()[5:].strip()
    return ""
CLOSED = {"preposition", "pronoun", "particle", "adverb", "numeral", "demonstrative", "conjunction"}
P = {"A": run(MA), "A2": run(MA2)}
NONE = {"", "none", "-", "no root", "null"}
res = {"forms": len(forms), "groups": {}, "examples": {}}
for r in rev:
    st = {status.get(i, ("?", "?")) for i in r["row_ids"].split()}
    grp = "settled_in_A2_training" if any(s == "" and sp == "train" for s, sp in st) else "held_not_trained"
    wc, rr = r["review_word_class"].strip(), r["review_root"].strip(); w = int(r["age_corpus_count"] or 0)
    for k in ("A", "A2"):
        pr, pt = root_of(P[k]["analyze: " + r["form"]]), type_of(P[k]["pos: " + r["form"]])
        root_ok = (pr.lower() in NONE) if not rr else (pr == rr)
        type_ok = (pt == wc) or (wc in CLOSED and pt in CLOSED | {"particle"})
        g = res["groups"].setdefault(grp, {}).setdefault(k, collections.Counter())
        g["forms"] += 1; g["root_ok"] += root_ok; g["type_ok"] += type_ok; g["tokens"] += w
        g["root_ok_tokens"] += w * root_ok; g["type_ok_tokens"] += w * type_ok
for grp, d in res["groups"].items():
    for k, c in d.items():
        d[k] = {"forms": c["forms"], "root_agree": round(c["root_ok"] / c["forms"], 4), "type_agree": round(c["type_ok"] / c["forms"], 4),
                "root_agree_token_weighted": round(c["root_ok_tokens"] / max(1, c["tokens"]), 4),
                "type_agree_token_weighted": round(c["type_ok_tokens"] / max(1, c["tokens"]), 4)}
for f in EXTRA:
    res["examples"][f] = {k: {"analyze": P[k]["analyze: " + f], "pos": P[k]["pos: " + f]} for k in P}
json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
print(json.dumps(res["groups"], ensure_ascii=False))
