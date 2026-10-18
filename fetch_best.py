#!/usr/bin/env python3
"""Download a Kaggle notebook's output: everything small, plus best.pt for
runs that have a routing memory. Skips last.pt and every other weight file.

  python3 fetch_best.py OUT_DIR owner/kernel-slug [--all-best]

Same API logic as fetch_small.py. By default best.pt is fetched only for run
directories that are NOT bpe_* / plain_* (those models have no memory, so the
routing probe has nothing to measure). --all-best fetches every best.pt.

Large files are downloaded in chunks with retries and resume, because the
full-weights fetch has been known to stall. Re-running the same command skips
files that are already complete.
"""
import os
import sys
import time
import urllib.request

from kaggle.api.kaggle_api_extended import KaggleApi

if len(sys.argv) < 3:
    sys.exit(__doc__)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), sys.argv[1])
KERNEL = sys.argv[2]
ALL_BEST = "--all-best" in sys.argv[3:]
WEIGHTS = (".safetensors", ".pt", ".bin", ".pth", ".ckpt")
NO_MEMORY_PREFIXES = ("bpe_", "plain_")

owner, slug = KERNEL.split("/")
api = KaggleApi()
api.authenticate()

if hasattr(api, "kernel_output_with_http_info"):                  # older client
    resp = api.process_response(api.kernel_output_with_http_info(owner, slug))
    files = resp.get("files", [])
else:                                                             # newer client
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest()
        r.user_name = owner
        r.kernel_slug = slug
        files = k.kernels.kernels_api_client.list_kernel_session_output(r).files


def get(f, a, b):
    return f[a] if isinstance(f, dict) else getattr(f, b)


def wanted(name):
    low = name.lower()
    if not low.endswith(WEIGHTS):
        return True
    if os.path.basename(low) != "best.pt":
        return False
    run = os.path.basename(os.path.dirname(name))
    return ALL_BEST or not run.startswith(NO_MEMORY_PREFIXES)


def download(url, dst, tries=6, chunk=1 << 20):
    """Chunked download with resume via HTTP Range; returns bytes on disk."""
    part = dst + ".part"
    for attempt in range(1, tries + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:      # server ignored Range
                    have = 0
                total = resp.headers.get("Content-Length")
                total = (int(total) + have) if total else None
                with open(part, "ab" if have else "wb") as fh:
                    done, last = have, time.time()
                    while True:
                        buf = resp.read(chunk)
                        if not buf:
                            break
                        fh.write(buf)
                        done += len(buf)
                        if total and time.time() - last > 5:
                            print(f"    {done / 1e6:7.1f} / {total / 1e6:.1f} MB",
                                  flush=True)
                            last = time.time()
            os.replace(part, dst)
            return os.path.getsize(dst)
        except Exception as exc:                  # network stall or reset
            print(f"    attempt {attempt}/{tries} failed: {exc}; retrying",
                  flush=True)
            time.sleep(min(30, 5 * attempt))
    raise RuntimeError(f"could not download {dst}")


print(len(files), "files in the output")
got_weights = 0
for f in files:
    name = get(f, "fileName", "file_name")
    url = f["url"] if isinstance(f, dict) else f.url
    if not wanted(name):
        print("skip", name)
        continue
    dst = os.path.join(OUT, name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    is_weight = name.lower().endswith(WEIGHTS)
    if is_weight and os.path.exists(dst):
        print("have", name)
        continue
    if is_weight:
        print("get ", name, "(weights)", flush=True)
        got_weights += download(url, dst)
    else:
        urllib.request.urlretrieve(url, dst)
        print("got ", name)
print(f"done. weights downloaded this time: {got_weights / 1e9:.2f} GB -> {OUT}")
