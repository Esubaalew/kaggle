"""Two GPU queue, fixed seeds and controls; preserve weights and fail visibly."""
import concurrent.futures, hashlib, json, os, signal, subprocess, sys, time, threading
from pathlib import Path
ROOT=Path(__file__).resolve().parent
TIMEOUT=11*3600; START=time.time()  # 9 runs of ~75 min on 2 T4s need ~6.5 h; Kaggle session cap is 12 h
JOBS=[(v,s) for s in [11,22,33] for v in ["plain_bpe","bpe_aux","gmm"]]
# Immutable file manifest is checked before any training.
manifest=json.loads((ROOT/"manifest.json").read_text())
for name,expected in manifest.items():
    actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    if actual!=expected: raise RuntimeError("input changed: "+name)
N=int(subprocess.check_output([sys.executable,"-c","import torch; print(torch.cuda.device_count())"],text=True))
if N<1:raise RuntimeError("GPU queue requires a GPU")
N=min(2,N)

def worker(gpu):
    for variant,seed in JOBS[gpu::N]:
        remaining=TIMEOUT-(time.time()-START)
        if remaining<15*60:raise RuntimeError("budget reached; remaining jobs not started")
        tag=f"{variant}_G2_wide_s{seed}";out=ROOT/"runs"/tag
        env=os.environ.copy();env["CUDA_VISIBLE_DEVICES"]=str(gpu);env["OMP_NUM_THREADS"]="2"
        args=[sys.executable,"-u",str(ROOT/"train.py"),"--variant",variant,"--seed",str(seed),"--data-dir",str(ROOT/"data"),"--tokenizer",str(ROOT/"data/tokenizer.json"),"--label-cache",str(ROOT/"data/labels_analyser_wide.json"),"--run-dir",str(out)]
        if (ROOT/"data/labels_tgt_analyser_wide.json").exists():args += ["--tgt-label-map",str(ROOT/"data/labels_tgt_analyser_wide.json")]
        if (out/"last.pt").exists():args += ["--resume",str(out/"last.pt")]
        print("START",tag,"gpu",gpu,flush=True)
        out.mkdir(parents=True,exist_ok=True)
        with (out/"console.log").open("a") as log:
            p=subprocess.Popen(args,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,start_new_session=True)
            def expire():
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
            timer=threading.Timer(remaining-120,expire);timer.start()
            try:
                for line in p.stdout:
                    log.write(line);log.flush();print(f"[{tag}] {line}",end="",flush=True)
                rc=p.wait()
            finally:timer.cancel()
        if rc:raise RuntimeError(f"{tag} failed ({rc}); see console.log")
        meta=json.loads((out/"metadata.json").read_text())
        if "error" in meta.get("final_eval",{}):raise RuntimeError("final evaluation failed: "+tag)
        print("DONE",tag,meta.get("final_eval",{}).get("beam5_nr4"),flush=True)
        if variant=="gmm" and seed==11:
            subprocess.run([sys.executable,"-u",str(ROOT/"routing_ablation.py"),"--run-dir",str(out),"--data-dir",str(ROOT/"data")],env=env,cwd=ROOT,check=True,timeout=max(60,TIMEOUT-(time.time()-START)-120))
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
    for result in pool.map(worker,range(N)):pass
print("ALL PREDECLARED JOBS COMPLETED",flush=True)
