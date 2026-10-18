"""Score run-A best/ on the held-out TEST split exactly as the trainer scored dev.

Same code as training (geez-morphology-model a1ad2b6: training_mixture.sample, train_tpu.encode/greedy/decode,
the dev_report scoring rule) and, by default, the training data snapshot (geez-dataset 04f45a7).
    python score_test.py <dataset_dir> <model_dir> <out_json> [limit]
"""
import collections, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "model"))
import torch
import dataset_loader, training_mixture as tm, train_tpu as tt, analyzer
from transformers import AutoModelForSeq2SeqLM

data_dir, model_dir, out_json = sys.argv[1:4]
limit = int(sys.argv[4]) if len(sys.argv) > 4 else 0
FAST = os.environ.get('SLOW_GREEDY') != '1'
pairs = [dict(p, dev="gold_test") for p in dataset_loader.load_gold_split(data_dir, "test")]
pairs += [dict(p, dev="unified_test") for p in dataset_loader.load_unified(data_dir, "test", ("book",))]
if limit:
    pairs = pairs[:limit]
rows = []
for p in pairs:
    for task in [dataset_loader.ANALYZE_TASK, tm.POS_TASK]:
        rows.append(dict(tm.sample(p, p["dev"], task), dev=p["dev"]))
print(len(pairs), "test words,", len(rows), "rows", flush=True)

torch.set_num_threads(os.cpu_count())
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
report = {"data_dir": data_dir, "model_dir": model_dir, "words": len(pairs), "rows": len(rows),
          "by_slice_task_class": {k: dict(v) for k, v in sorted(per.items())},
          "pos_confusion": {k: dict(v) for k, v in sorted(confusion.items())},
          "analyze_total": {"n": tot["n"], "exact": tot["exact"] / max(1, tot["n"]), "root": tot["root"] / max(1, tot["n"])},
          "pos_total": {"n": pos["n"], "exact": pos["exact"] / max(1, pos["n"])}}
report["score"] = (report["analyze_total"]["exact"] + report["analyze_total"]["root"]) / 2
json.dump(report, open(out_json, "w"), ensure_ascii=False, indent=1)
json.dump(preds, open(out_json.replace(".json", "_predictions.json"), "w"), ensure_ascii=False)
print("TEST", json.dumps(report["analyze_total"]), "pos", json.dumps(report["pos_total"]), "score", round(report["score"], 4), flush=True)
