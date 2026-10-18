#!/usr/bin/env python3
"""The Ge'ez ByT5 root analyser reads every distinct Ge'ez word of the training and dev pairs once. Writes:
  analyser_cache.json   the readings (root, word class, stem, tense, person, gender, number, object)
  analyser_labels.json  root + pattern label cache in the k256 format (top 255 roots, top 16 patterns, ranked on TRAIN)

  python build_analyser_cache.py --data-dir data --morphology-repo <geez-morphology checkout> \
      --model-dir <ByT5 weights folder> --out-dir runs/shared

Greedy decoding of the "analyze:" task, parsed with the repo's own analyzer.parse_neural_output.
"""
import argparse, hashlib, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from analyser_tokenizer import build_label_cache  # noqa: E402


def sentences(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 2 and p[0].strip() and p[1].strip():
                out.append(p[0].split())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--morphology-repo", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit-words", type=int, default=None, help="smoke test on the first N words")
    a = ap.parse_args()

    sys.path.insert(0, a.morphology_repo)
    import analyzer, dataset_loader
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    train = [w for s in sentences(os.path.join(a.data_dir, "age_train.tsv")) for w in s]
    dev = [w for s in sentences(os.path.join(a.data_dir, "age_dev.tsv")) for w in s]
    words = sorted(set(train) | set(dev), key=lambda w: (-len(w.encode("utf-8")), w))   # long first: less padding
    if a.limit_words:
        words = words[:a.limit_words]
    print(f"train tokens {len(train)}, dev tokens {len(dev)}, distinct words {len(words)}", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(a.model_dir).to(device).eval()

    readings, unparsed, t0 = {}, 0, time.time()
    for i in range(0, len(words), a.batch_size):
        batch = words[i:i + a.batch_size]
        enc = tok([dataset_loader.input_text(w, dataset_loader.ANALYZE_TASK) for w in batch],
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            ids = model.generate(**enc, max_new_tokens=a.max_new_tokens, num_beams=1, do_sample=False)
        for w, reply in zip(batch, tok.batch_decode(ids, skip_special_tokens=True)):
            p = analyzer.parse_neural_output(reply)
            if p is None:
                readings[w] = None
                unparsed += 1
                continue
            readings[w] = {"root": "-".join(c for c in p["root"] if c not in "- ") if p["root"] else None,
                           "type": None if p["type"] in ("verb", "root_only") else p["type"],
                           **{k: p.get(k) for k in ("stem", "tense", "person", "gender", "number", "object")}}
        if (i // a.batch_size) % 25 == 0:
            print(f"{i + len(batch)}/{len(words)}  {time.time() - t0:.0f}s", flush=True)

    cfg = os.path.join(a.model_dir, "config.json")
    meta = {"model_dir": os.path.abspath(a.model_dir),
            "config_sha256": hashlib.sha256(open(cfg, "rb").read()).hexdigest() if os.path.exists(cfg) else None,
            "words": len(words), "unparsed": unparsed,
            "with_root": sum(1 for v in readings.values() if v and v["root"]),
            "seconds": round(time.time() - t0)}
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "analyser_cache.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "readings": readings}, f, ensure_ascii=False)
    labels = build_label_cache(train, readings, model_note=meta["model_dir"])
    with open(os.path.join(a.out_dir, "analyser_labels.json"), "w", encoding="utf-8") as f:
        json.dump(labels, f, ensure_ascii=False)
    print(meta, flush=True)
    print(labels["metadata"], flush=True)
    print("patterns:", labels["pattern_classes"], flush=True)


if __name__ == "__main__":
    main()
