"""Score an analyser checkpoint on a held-out TEST set exactly as the trainer scores dev.

Same scoring code as score_test.py (training_mixture.sample, train_tpu.encode/greedy/decode, the dev_report rule),
from the geez-morphology-model checkout given by MODEL_CODE (default: the local repo). Test sets:
    old     gold test + unified book test of the given data dir (run A's test)
    rows    as old, unified rows read from --rows (e.g. the reviewed rows of 2.1.0)
    clitic  the rule-made clitic forms of the test split (sources clitic_rule, --rows = the clitic file)
    python score_test2.py <set> <dataset_dir> <model_dir> <out_json> [--rows FILE] [--limit N] [--threads N]
"""
import argparse, collections, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("set", choices=["old", "rows", "clitic"]); ap.add_argument("data_dir"); ap.add_argument("model_dir"); ap.add_argument("out_json")
ap.add_argument("--rows", default=None); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--threads", type=int, default=os.cpu_count())
A = ap.parse_args()
sys.path.insert(0, os.environ.get("MODEL_CODE", "/Users/esubalew/is/researcher/geez-morphology-model"))
import torch
import dataset_loader, training_mixture as tm, train_tpu as tt, analyzer
from transformers import AutoModelForSeq2SeqLM
data_dir, model_dir, out_json, limit = A.data_dir, A.model_dir, A.out_json, A.limit
FAST = os.environ.get('SLOW_GREEDY') != '1'
if A.set == "clitic":
    pairs = [dict(p, dev="clitic_test") for p in dataset_loader.load_unified(data_dir, "test", ("clitic_rule",), path=A.rows)]
else:
    pairs = [dict(p, dev="gold_test") for p in dataset_loader.load_gold_split(data_dir, "test")]
    pairs += [dict(p, dev="unified_test") for p in dataset_loader.load_unified(data_dir, "test", ("book",), path=A.rows)]
if limit:
    pairs = pairs[:limit]
rows = []
for p in pairs:
    for task in [dataset_loader.ANALYZE_TASK, tm.POS_TASK]:
        rows.append(dict(tm.sample(p, p["dev"], task), dev=p["dev"]))
print(len(pairs), "test words,", len(rows), "rows", flush=True)

torch.set_num_threads(A.threads)
model = AutoModelForSeq2SeqLM.from_pretrained(model_dir, dtype=torch.float32).eval()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["input_text"].encode()))   # length-sorted batches
decoded, bs, t0 = {}, 48, time.time()
with torch.no_grad():
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        a = max(tt.bucket(len(rows[i]["input_text"].encode()) + 1, tt.INPUT_BUCKETS) for i in idx)
        inp = torch.full((len(idx), a), tt.PAD, dtype=torch.long)
        for r, i in enumerate(idx):
            x = tt.encode(rows[i]["input_text"], a); inp[r, :len(x)] = torch.tensor(x)
        if FAST:
            outs = model.generate(input_ids=inp, attention_mask=(inp != tt.PAD).long(), max_new_tokens=129,
                                  do_sample=False, num_beams=1)
            outs = [o[1:].tolist() for o in outs]            # drop decoder start token
        else:
            outs = tt.greedy(model, inp, (inp != tt.PAD).long(), 129, torch, None)
        for i, o in zip(idx, outs):
            decoded[i] = tt.decode(o)
        if (k // bs) % 10 == 0:
            print(f"{k + len(idx)}/{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)

per, confusion, preds = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter), []
for i, s in enumerate(rows):
    text = decoded[i]
    c = per[f"{s['dev']}/{s['task']}/{s['cls']}"]
    c["n"] += 1; c["exact"] += text == s["target_text"]
    if s["task"] == tm.POS_TASK:
        confusion[s["cls"]][text.replace("type:", "").strip() or "?"] += 1
    else:
        c["root"] += text.split("|")[0].strip() == s["target_text"].split("|")[0].strip()
        c["parsed"] += analyzer.parse_neural_output(text) is not None
    preds.append({"dev": s["dev"], "task": s["task"], "cls": s["cls"], "input": s["input_text"], "gold": s["target_text"], "pred": text})
tot = collections.Counter(); pos = collections.Counter()
for k, v in per.items():
    (tot if "/analyze/" in k else pos).update(v)
report = {"set": A.set, "rows_file": A.rows, "data_dir": data_dir, "model_dir": model_dir, "words": len(pairs), "rows": len(rows),
          "by_slice_task_class": {k: dict(v) for k, v in sorted(per.items())},
          "pos_confusion": {k: dict(v) for k, v in sorted(confusion.items())},
          "analyze_total": {"n": tot["n"], "exact": tot["exact"] / max(1, tot["n"]), "root": tot["root"] / max(1, tot["n"])},
          "pos_total": {"n": pos["n"], "exact": pos["exact"] / max(1, pos["n"])}}
report["score"] = (report["analyze_total"]["exact"] + report["analyze_total"]["root"]) / 2
json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=1)
json.dump(preds, open(out_json.replace(".json", "_predictions.json"), "w"), ensure_ascii=False)
print("TEST", json.dumps(report["analyze_total"]), "pos", json.dumps(report["pos_total"]), "score", round(report["score"], 4), flush=True)
