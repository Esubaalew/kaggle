#!/usr/bin/env python3
"""Fetch the small outputs of the two TPU runs (logs, reports, eval log), or also the best model.

    python3 fetch_small.py a            # run A: logs + json/txt reports into out_a/
    python3 fetch_small.py b --model    # run B: the same plus finetune/best/ (about 300 MB)

Needs KAGGLE_API_TOKEN in the environment (see COMMANDS.md).
"""
import argparse
import subprocess
import sys

SLUG = {"a": "esubalewchekol/geez-byt5-tpu-v2-a", "b": "esubalewchekol/geez-byt5-tpu-v2-b"}
SMALL = r".*\.(log|json|jsonl|txt)$"
MODEL = r".*(best/)?(model\.safetensors|config\.json|generation_config\.json|tokenizer_config\.json|added_tokens\.json)$"

ap = argparse.ArgumentParser()
ap.add_argument("run", choices=sorted(SLUG))
ap.add_argument("--model", action="store_true")
a = ap.parse_args()
pattern = SMALL if not a.model else f"({SMALL})|({MODEL})"
cmd = ["kaggle", "kernels", "output", SLUG[a.run], "-p", f"out_{a.run}", "--file-pattern", pattern, "-o"]
print(" ".join(cmd), flush=True)
sys.exit(subprocess.call(cmd))
