#!/usr/bin/env python3
"""Download selected files from a Kaggle notebook's output, resuming after broken connections.

    python fetch_file.py <out_dir> <owner/kernel> <regex> [<regex> ...]
    e.g. python fetch_file.py geez_gpu_v2/out_v6_full esubechekol/geez-byt5-gpu-v2-a '^best/' 'eval_log'

Only files whose output path matches one of the regexes are downloaded. A partial file is continued
with an HTTP Range request (a fresh signed URL is fetched on every attempt), so a dropped
connection costs only the bytes since the last retry. The final size is checked against Kaggle's listing.
"""
import os, re, sys, time, urllib.request
from kaggle.api.kaggle_api_extended import KaggleApi

out, kernel, pats = sys.argv[1], sys.argv[2], [re.compile(p) for p in sys.argv[3:]]
if not pats:
    sys.exit(__doc__)
owner, slug = kernel.split("/")
api = KaggleApi(); api.authenticate()


def listing():
    if hasattr(api, "kernel_output_with_http_info"):
        files = api.process_response(api.kernel_output_with_http_info(owner, slug)).get("files", [])
        return [(f["fileName"], f["url"], f.get("size")) for f in files]
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    with api.build_kaggle_client() as k:
        r = ApiListKernelSessionOutputRequest(); r.user_name = owner; r.kernel_slug = slug
        files = k.kernels.kernels_api_client.list_kernel_session_output(r).files
    return [(f.file_name, f.url, getattr(f, "size", None)) for f in files]


def remote_size(url):
    req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range")            # "bytes 0-0/TOTAL"
        return int(cr.split("/")[-1]) if cr else int(r.headers.get("Content-Length", 0))


wanted = [n for n, _, _ in listing() if any(p.search(n) for p in pats)]
print(len(wanted), "files match:", wanted)
for name in wanted:
    dst = os.path.join(out, name); os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    total = None
    for attempt in range(1, 61):
        url = dict((n, u) for n, u, _ in listing())[name]       # fresh signed URL each attempt
        try:
            total = remote_size(url)
            have = os.path.getsize(dst) if os.path.exists(dst) else 0
            if have == total:
                break
            if have > total:
                os.remove(dst); have = 0
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=120) as r:
                if have and r.status != 206:                # server ignored Range: start over
                    have = 0; open(dst, "wb").close()
                f = open(dst, "ab")
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    f.write(chunk); have += len(chunk)
                    print(f"\r{name}: {have / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="", flush=True)
                f.close()
            print()
        except Exception as e:                                  # broken pipe, IncompleteRead, timeout
            print(f"\n{name}: attempt {attempt} interrupted ({type(e).__name__}); resuming in 5 s")
            time.sleep(5)
    size = os.path.getsize(dst) if os.path.exists(dst) else 0
    if total is None or size != total:
        sys.exit(f"FAILED {name}: {size} of {total} bytes after 60 attempts; re-run the same command to continue")
    print(f"OK {name} ({size / 1e6:,.1f} MB)")
