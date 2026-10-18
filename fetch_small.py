#!/usr/bin/env python3
"""Download a Kaggle notebook's output WITHOUT the big weight files.
  python3 fetch_small.py run_books_s1 [owner/kernel-slug]
Same notebook as launch_kaggle_run.py. Skips *.safetensors, *.pt, *.bin, optimizer/scheduler files.
"""
import os, sys, urllib.request
from kaggle.api.kaggle_api_extended import KaggleApi

KERNEL = sys.argv[2] if len(sys.argv) > 2 else "esubaalew/geez-morphology-byt5-training"
SKIP_EXT = (".safetensors", ".pt", ".bin", ".pth", ".ckpt")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1] if len(sys.argv) > 1 else "run_fetched")
owner, slug = KERNEL.split("/")
api = KaggleApi(); api.authenticate()

if hasattr(api, "kernel_output_with_http_info"):                 # older client
    resp = api.process_response(api.kernel_output_with_http_info(owner, slug))
    files = resp.get("files", [])
else:                                                            # newer client
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest(); r.user_name = owner; r.kernel_slug = slug
        files = k.kernels.kernels_api_client.list_kernel_session_output(r).files

def get(f, a, b):
    return f[a] if isinstance(f, dict) else getattr(f, b)

print(len(files), "files in the output")
for f in files:
    name, url = get(f, "fileName", "file_name"), f["url"] if isinstance(f, dict) else f.url
    if name.lower().endswith(SKIP_EXT):
        print("skip", name); continue
    dst = os.path.join(out, name); os.makedirs(os.path.dirname(dst), exist_ok=True)
    urllib.request.urlretrieve(url, dst); print("got ", name)
