"""Ge'ez ByT5 v2, Run A (fine-tune only) on Kaggle GPU T4 x2.

Launcher mode (default, what Kaggle runs): finds the attached dataset (tarballs or the folders Kaggle extracts
them to), copies the model code and the dataset, picks up checkpoints from a previous version of this kernel if
attached, then starts the trainer with torchrun on all GPUs (DDP). If DDP fails before the first optimizer step,
it reruns on one GPU.

Trainer mode (--worker): google/byt5-small, Adafactor lr 1e-3, 500 warmup steps, linear decay to 10%, 10 epochs,
512 rows per optimizer step. On 2 GPUs one step is 32 rows x 2 GPUs x 8 accumulation micro-batches; on 1 GPU it
is 32 x 16. fp32 (T4 has no bf16). Dev eval every 1000 steps, sharded across the GPUs. Lean checkpoints every
30 minutes (and when best/ improves), keep 2, and a clean stop with a final checkpoint about 11h15m after the
session started. The final greedy dev report is sharded across the GPUs.

Resume in a later version: add "kernel_sources": ["esubechekol/geez-byt5-gpu-v2-a"] to kernel-metadata.json,
so this kernel's previous output (ckpt/, best/) is attached under /kaggle/input.
"""
import argparse, glob, json, math, os, random, shutil, subprocess, sys, tarfile, time

RUN = "A"
KERNEL = "geez-byt5-gpu-v2-a"
p = argparse.ArgumentParser()
p.add_argument("--worker", action="store_true")
p.add_argument("--code", default="")
p.add_argument("--data", default="")
p.add_argument("--out", default="/kaggle/working")
p.add_argument("--model_commit", default="")
p.add_argument("--data_commit", default="")
p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
p.add_argument("--micro", type=int, default=32)
p.add_argument("--global_batch", type=int, default=512)
p.add_argument("--pt_micro", type=int, default=32)
p.add_argument("--pt_global_batch", type=int, default=512)
p.add_argument("--pt_epochs", type=float, default=6)
p.add_argument("--max_steps", type=int, default=0, help="smoke tests only (fine-tune phase)")
p.add_argument("--pt_max_steps", type=int, default=0, help="smoke tests only (pretrain phase)")
p.add_argument("--train_limit", type=int, default=0, help="smoke tests only, applied after the load check")
p.add_argument("--dev_limit", type=int, default=0, help="smoke tests only")
p.add_argument("--pt_limit", type=int, default=0, help="smoke tests only: pretrain chunks")
p.add_argument("--save_minutes", type=float, default=30)
p.add_argument("--keep", type=int, default=2)
p.add_argument("--eval_every", type=int, default=1000)
p.add_argument("--log_every", type=int, default=50)
p.add_argument("--budget_hours", type=float, default=11.25, help="stop cleanly this long after the session start")
p.add_argument("--session_start", type=float, default=0.0)
p.add_argument("--resume_dir", default="", help="read-only folder with a previous ckpt/ (and best/)")
p.add_argument("--nproc", type=int, default=0, help="launcher: GPUs to use (0 = all)")
p.add_argument("--no_final_report", action="store_true")
p.add_argument("--model_name", default="google/byt5-small", help="smoke tests only")
a, _ = p.parse_known_args()
if not a.session_start:
    a.session_start = time.time()
EXPECT_TRAIN, EXPECT_DEV = 562403, 3472
IN = os.environ.get("GEEZ_INPUT_ROOT", "/kaggle/input")
WORK = os.environ.get("GEEZ_WORK", "/tmp/geez")
PHASES = ["pretrain", "finetune"] if RUN == "B" else ["finetune"]


def log(*x):
    print(*x, flush=True)


# =====================================================================================  launcher
def find_input():
    hits = sorted(glob.glob(IN + "/**/MODEL_COMMIT", recursive=True))
    hits = [h for h in hits if os.path.exists(os.path.join(os.path.dirname(h), "DATA_COMMIT"))]
    if not hits:
        raise SystemExit("dataset esubechekol/geez-morphology-gpu-v2 is not attached (no MODEL_COMMIT under /kaggle/input)")
    return os.path.dirname(hits[0])


def find_resume():
    hits = [os.path.dirname(os.path.dirname(h)) for h in glob.glob(IN + "/**/ckpt/step_*.pt", recursive=True)]
    hits = sorted(set(hits), key=lambda d: (KERNEL not in d, d))
    return hits[0] if hits else ""


def unpack(tgz, dest):
    if not os.path.exists(tgz):
        d = tgz[:-len(".tar.gz")]
        if not os.path.isdir(d):
            raise SystemExit(f"neither {tgz} nor extracted {d} found")
        while True:
            ents = [e for e in os.listdir(d) if not e.startswith(".")]
            if len(ents) == 1 and os.path.isdir(os.path.join(d, ents[0])) and not os.path.exists(os.path.join(d, "README.md")):
                d = os.path.join(d, ents[0]); continue
            break
        log(f"using Kaggle-extracted folder {d}")
        if not os.path.isdir(dest):
            shutil.copytree(d, dest)
        return dest
    if not os.path.isdir(dest):
        os.makedirs(dest + ".part", exist_ok=True)
        with tarfile.open(tgz) as t:
            try:
                t.extractall(dest + ".part", filter="data")
            except TypeError:
                t.extractall(dest + ".part")
        inner = [os.path.join(dest + ".part", n) for n in os.listdir(dest + ".part")]
        shutil.move(inner[0] if len(inner) == 1 and os.path.isdir(inner[0]) else dest + ".part", dest)
        shutil.rmtree(dest + ".part", ignore_errors=True)
    return dest


def launcher():
    import torch
    log(f"torch {torch.__version__}, CUDA available {torch.cuda.is_available()}, GPU count {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        pr = torch.cuda.get_device_properties(i)
        log(f"GPU {i}: {pr.name}, {pr.total_memory / 2**30:.1f} GB")
    try:
        import transformers
        ver = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    except Exception:
        ver = (0, 0)
    if ver < (4, 56):
        log(f"installing transformers>=4.56 (found {ver})")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "transformers>=4.56", "safetensors"])
    import transformers
    log(f"transformers {transformers.__version__}")

    inp = find_input()
    code = unpack(os.path.join(inp, "model.tar.gz"), os.path.join(WORK, "model"))
    data = unpack(os.path.join(inp, "geez-dataset.tar.gz"), os.path.join(WORK, "geez-dataset"))
    mc = open(os.path.join(inp, "MODEL_COMMIT")).read().strip()
    dc = open(os.path.join(inp, "DATA_COMMIT")).read().strip()
    log(f"MODEL commit {mc}\nDATA commit  {dc}")
    resume = find_resume()
    log(f"previous checkpoints: {resume + '/ckpt' if resume else 'none attached, starting from google/byt5-small'}")

    me = os.path.abspath(__file__)
    common = ["--worker", "--code", code, "--data", data, "--out", a.out, "--model_commit", mc, "--data_commit", dc,
              "--session_start", repr(a.session_start), "--resume_dir", resume]
    for k in ("micro", "global_batch", "pt_micro", "pt_global_batch", "pt_epochs", "save_minutes", "keep",
              "budget_hours", "eval_every", "log_every"):
        common += [f"--{k}", str(getattr(a, k))]
    for k in ("max_steps", "pt_max_steps", "train_limit", "dev_limit", "pt_limit"):
        if getattr(a, k):
            common += [f"--{k}", str(getattr(a, k))]
    if a.no_final_report:
        common.append("--no_final_report")
    common += ["--model_name", a.model_name]
    if a.device != "auto":
        common += ["--device", a.device]
    n = a.nproc or torch.cuda.device_count()
    started = os.path.join(a.out, ".first_step_done")
    if os.path.exists(started):
        os.remove(started)
    if n >= 2:
        log(f"launching DDP on {n} GPUs: torchrun --nproc_per_node {n}")
        r = subprocess.run([sys.executable, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={n}", me] + common)
        if r.returncode == 0:
            return
        if os.path.exists(started):
            raise SystemExit(f"DDP training stopped with exit code {r.returncode} after it had started; see the log above")
        log(f"DDP FAILED AT STARTUP (exit code {r.returncode}); FALLING BACK TO A SINGLE GPU")
    else:
        log("one GPU (or none) visible: single-process training")
    r = subprocess.run([sys.executable, me] + common)
    if r.returncode != 0:
        raise SystemExit(f"training stopped with exit code {r.returncode}")


# =====================================================================================  trainer
def worker():
    import contextlib, io
    import torch
    import torch.distributed as dist
    rank, world, local = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))
    master = rank == 0
    dev_name = a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    if dev_name == "cuda":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    cuda = dev_name == "cuda"
    ddp = world > 1
    if ddp:
        import datetime
        dist.init_process_group("nccl" if cuda else "gloo", timeout=datetime.timedelta(minutes=60))

    def say(*x):
        if master:
            log(*x)

    code, data, out = map(os.path.abspath, (a.code, a.data, a.out))
    sys.path.insert(0, code)
    os.chdir(code)
    import train_tpu as tt
    from transformers import AutoModelForSeq2SeqLM
    from transformers.optimization import Adafactor

    RUN_A = ["--stage", "finetune", "--model_name", "google/byt5-small", "--per_core_batch", "64", "--lr", "1e-3",
             "--epochs", "10", "--warmup_steps", "500", "--eval_every", str(a.eval_every), "--save_every", "1000",
             "--unified_upsample", "3", "--rare_class_min", "400"]
    RUN_A[RUN_A.index("--model_name") + 1] = a.model_name
    if a.dev_limit:
        RUN_A += ["--report_cap", str(a.dev_limit)]
    args = tt.parse_args(RUN_A + ["--dataset_dir", data, "--output_dir", out])
    os.makedirs(out, exist_ok=True)
    say(f"RUN {RUN}\nMODEL commit {a.model_commit}\nDATA commit  {a.data_commit}\nworld size {world}, device {dev_name}, torch {torch.__version__}")
    if cuda and master:
        for i in range(torch.cuda.device_count()):
            say(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # ---- load check: exact counts + leak audit (rank 0 writes the audit, the others read it)
    if master:
        train, dev, _ = tt.prepare(args, write=True)
    import gc; gc.collect()
    if ddp:
        dist.barrier()
    if not master:
        with contextlib.redirect_stdout(io.StringIO()):
            train, dev, _ = tt.prepare(args, write=False)
    clean = "RESULT: CLEAN" in open(os.path.join(out, "leak_audit.txt"), encoding="utf-8").read()
    say(f"LOAD CHECK train {len(train):,} (expect {EXPECT_TRAIN:,}), dev {len(dev):,} (expect {EXPECT_DEV:,}), "
        f"leak audit {'CLEAN' if clean else 'NOT CLEAN'}")
    if len(train) != EXPECT_TRAIN or len(dev) != EXPECT_DEV or not clean:
        raise SystemExit("LOAD CHECK FAILED: counts or leak audit differ from the expected Run A data. Stopping.")
    say("LOAD CHECK OK")
    if a.train_limit:
        train = random.Random(args.seed).sample(train, a.train_limit)
    if a.dev_limit:
        dev = dev[:a.dev_limit]

    # ---- pretrain corpus (deterministic from the dataset, rebuilt every session; rank 0 writes it)
    corpus = os.path.join(out, "dapt_corpus.txt")
    if RUN == "B":
        pargs = tt.parse_args(["--stage", "dapt", "--dataset_dir", data, "--output_dir", out, "--dapt_corpus", corpus])
        if master:
            import build_dapt_corpus
            rep = build_dapt_corpus.build(data, corpus)
            with open(os.path.join(out, "dapt_report.json"), "w", encoding="utf-8") as f:
                json.dump(rep, f, ensure_ascii=False, indent=1)
            say(f"PRETRAIN CORPUS {rep['lines']:,} lines, {rep['bytes']:,} bytes")
        if ddp:
            dist.barrier()
        n_pt = len(tt.dapt_samples(corpus, 0, args.seed))
        if a.pt_limit:
            n_pt = min(n_pt, a.pt_limit)
        say(f"PRETRAIN chunks per epoch {n_pt:,}")

    def phase_cfg(ph):
        if ph == "pretrain":
            G, micro = a.pt_global_batch, a.pt_micro
            spe = max(1, n_pt // G)
            total = a.pt_max_steps or int(math.ceil(spe * a.pt_epochs))
            return dict(G=G, micro=micro, spe=spe, total=total, epochs=a.pt_epochs, warmup=max(1, min(500, total // 10)))
        G, micro = a.global_batch, a.micro
        spe = max(1, len(train) // G)
        total = a.max_steps or int(math.ceil(spe * args.epochs))
        return dict(G=G, micro=micro, spe=spe, total=total, epochs=args.epochs, warmup=args.warmup_steps)

    for ph in PHASES:
        c = phase_cfg(ph)
        assert c["G"] % (c["micro"] * world) == 0, f"{ph}: global batch must be a multiple of micro x world size"
        say(f"PHASE {ph}: {c['G']} rows/step, steps/epoch {c['spe']}, epochs {c['epochs']}, total steps {c['total']}, "
            f"warmup {c['warmup']}")

    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, dtype=torch.float32).to(device)

    def new_opt():
        return Adafactor(model.parameters(), lr=args.lr, scale_parameter=False, relative_step=False, warmup_init=False)

    opt = new_opt()
    state = {"phase": PHASES[0], "step": 0, "epoch": 0, "batch": 0, "best": -1.0, "history": []}

    # ---- resume: own ckpt/ first (rerun in the same session), else the attached previous output
    CK = os.path.join(out, "ckpt")
    os.makedirs(CK, exist_ok=True)
    cands = sorted(glob.glob(os.path.join(CK, "step_*.pt")), reverse=True)
    if a.resume_dir:
        cands += sorted(glob.glob(os.path.join(a.resume_dir, "ckpt", "step_*.pt")), reverse=True)
    resumed = None
    for path in cands:
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            if ck.get("run", "A") != RUN:
                raise ValueError(f"checkpoint of run {ck.get('run', 'A')}, not run {RUN}")
            model.load_state_dict(ck["model"])
            opt = new_opt()
            opt.load_state_dict(ck["opt"])
            state = ck["state"]
            state.setdefault("phase", "finetune")      # checkpoints written before phases existed are Run A fine-tune
            rng = ck["rng"].get(rank) or ck["rng"].get(0)
            if rng:
                random.setstate(rng["python"]); torch.set_rng_state(rng["torch"])
                if cuda and rng.get("cuda") is not None:
                    torch.cuda.set_rng_state(rng["cuda"])
            resumed = path
            del ck
            break
        except Exception as e:
            say(f"[!] could not load {path}: {e}; trying an older one")
    if resumed:
        say(f"RESUMED from {resumed}: phase {state['phase']}, step {state['step']} (epoch {state['epoch']}, batch {state['batch']})")
        src = os.path.dirname(os.path.dirname(resumed))
        if master and src != out:      # carry the previous session's best/ and logs into this output
            for n in ("best", "pretrained"):
                if os.path.isdir(os.path.join(src, n)) and not os.path.isdir(os.path.join(out, n)):
                    shutil.copytree(os.path.join(src, n), os.path.join(out, n))
            for n in ("eval_log.jsonl",):
                if os.path.exists(os.path.join(src, n)) and not os.path.exists(os.path.join(out, n)):
                    shutil.copy(os.path.join(src, n), os.path.join(out, n))
    else:
        say("no checkpoint found, starting from google/byt5-small")
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        net = DDP(model, device_ids=[local] if cuda else None)
    else:
        net = model

    def gather_rng():
        r = {"python": random.getstate(), "torch": torch.get_rng_state(),
             "cuda": torch.cuda.get_rng_state() if cuda else None}
        if not ddp:
            return {0: r}
        allr = [None] * world
        dist.all_gather_object(allr, r)
        return dict(enumerate(allr))

    def ck_name():
        if RUN == "A":
            return f"step_{state['step']:06d}.pt"
        return f"step_{PHASES.index(state['phase'])}{state['phase'][:2]}_{state['step']:06d}.pt"

    def save_ckpt(why=""):
        rng = gather_rng()          # collective: every rank calls save_ckpt, only rank 0 writes
        if not master:
            return
        path = os.path.join(CK, ck_name())
        olds = sorted(glob.glob(os.path.join(CK, "step_*.pt")), key=os.path.getmtime)
        olds = [o for o in olds if o != path]
        need = (os.path.getsize(olds[-1]) if olds else 1.5 * 2**30) * 1.15
        while olds and shutil.disk_usage(CK).free < need:      # not enough room: drop the oldest first
            say(f"[!] low disk ({shutil.disk_usage(CK).free / 2**30:.2f} GB free): removing {os.path.basename(olds[0])} before saving")
            os.remove(olds.pop(0))
        torch.save({"model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "opt": opt.state_dict(),
                    "state": state, "rng": rng, "total": phase_cfg(state["phase"])["total"], "run": RUN}, path + ".tmp")
        os.replace(path + ".tmp", path)
        for old in olds[:max(0, len(olds) + 1 - a.keep)]:
            os.remove(old)
        say(f"checkpoint saved ckpt/{os.path.basename(path)}{' (' + why + ')' if why else ''}")

    def to_dev(*ts):
        if cuda:
            return [t.pin_memory().to(device, non_blocking=True) for t in ts]
        return [t.to(device) for t in ts]

    def evaluate():
        """Teacher-forced dev scores; batches sharded across ranks, counts kept on device, one all_reduce."""
        model.eval()
        t = torch.zeros(5, device=device)
        batches = tt.make_batches(dev, 64, 0, 0, drop_last=False)
        with torch.no_grad():
            for chunk, A, B in batches[rank::world]:
                inp, att, lab = tt.tensors(dev, chunk, A, B, torch)
                is_pos = torch.tensor([i >= 0 and dev[i]["task"] == tt.tm.POS_TASK for i in chunk])
                is_an = torch.tensor([i >= 0 and dev[i]["task"] != tt.tm.POS_TASK for i in chunk])
                inp, att, lab, is_pos, is_an = to_dev(inp, att, lab, is_pos, is_an)
                logits = model(input_ids=inp, attention_mask=att, labels=lab).logits
                ex, ro = tt.score_batch(logits, lab, torch)
                t += torch.stack([is_an.sum(), (ex & is_an).sum(), (ro & is_an).sum(), is_pos.sum(), (ex & is_pos).sum()]).float()
        if ddp:
            dist.all_reduce(t)
        model.train()
        n, e, r, pn, pe = t.tolist()
        m = {"dev_exact": e / max(1, n), "dev_root": r / max(1, n), "dev_pos": pe / max(1, pn)}
        m["dev_score"] = (m["dev_exact"] + m["dev_root"]) / 2
        return m

    def save_weights(d, dtype):
        from safetensors.torch import save_file
        os.makedirs(d, exist_ok=True)
        sd = {k: v.detach().cpu().to(dtype).clone().contiguous() for k, v in model.state_dict().items()}
        save_file(sd, os.path.join(d, "model.safetensors.tmp"), metadata={"format": "pt"})
        os.replace(os.path.join(d, "model.safetensors.tmp"), os.path.join(d, "model.safetensors"))
        model.config.save_pretrained(d)
        try:
            from transformers import AutoTokenizer
            AutoTokenizer.from_pretrained(args.model_name).save_pretrained(d)
        except Exception as e:
            say(f"[!] tokenizer files not saved: {e}")

    def save_best():
        save_weights(os.path.join(out, "best"), torch.bfloat16)

    class Gather:       # train_tpu.final_report gathers through xm.rendezvous; this does it with torch.distributed
        @staticmethod
        def rendezvous(tag, payload):
            allp = [None] * world
            dist.all_gather_object(allp, payload)
            return allp

    deadline = a.session_start + a.budget_hours * 3600
    flag = os.path.join(out, ".first_step_done")
    first_done = False
    stopped_early = False
    net.train()

    while not stopped_early:
        ph = state["phase"]
        c = phase_cfg(ph)
        G, micro, total = c["G"], c["micro"], c["total"]
        per_rank = G // world
        n_acc = per_rank // micro
        say(f"=== PHASE {ph}: micro-batch {micro} x {world} GPU(s) x {n_acc} accumulation = {G} rows/step, precision fp32, "
            f"from step {state['step']}/{total}")

        def lr_at(step):
            if step < c["warmup"]:
                return args.lr * (step + 1) / c["warmup"]
            return args.lr * max(0.1, 1 - 0.9 * (step - c["warmup"]) / max(1, total - c["warmup"]))

        t0, last_save, start_step = time.time(), time.time(), state["step"]
        t_log, s_log, loss_sum, loss_n = time.time(), state["step"], 0.0, 0
        while state["step"] < total and not stopped_early:
            if ph == "pretrain":
                items = tt.dapt_samples(corpus, state["epoch"], args.seed)[:n_pt]
                order = list(range(len(items)))
                random.Random(args.seed + state["epoch"]).shuffle(order)
                batches = [(order[k:k + G], tt.DAPT_INPUT, tt.DAPT_TARGET) for k in range(0, len(order) - G + 1, G)]
            else:
                batches = tt.make_batches(train, G, args.seed, state["epoch"])
            for bi in range(state["batch"], len(batches)):
                chunk, A, B = batches[bi]
                mine = chunk[rank * per_rank:(rank + 1) * per_rank]
                filler = next(i for i in chunk if i >= 0)
                for g in opt.param_groups:
                    g["lr"] = lr_at(state["step"])
                step_loss = torch.zeros((), device=device)
                for k in range(n_acc):
                    part = mine[k * micro:(k + 1) * micro]
                    empty = all(i < 0 for i in part)     # only padding: run a real row with weight 0 so DDP stays in step
                    if empty:
                        part = [filler] + list(part[1:])
                    if ph == "pretrain":
                        inp = torch.full((len(part), A), tt.PAD, dtype=torch.long)
                        lab = torch.full((len(part), B), -100, dtype=torch.long)
                        for r_, i in enumerate(part):
                            x, y = items[i]
                            inp[r_, :len(x)] = torch.tensor(x)
                            lab[r_, :len(y)] = torch.tensor(y)
                        att = (inp != tt.PAD).long()
                    else:
                        inp, att, lab = tt.tensors(train, part, A, B, torch)
                    inp, att, lab = to_dev(inp, att, lab)
                    sync = contextlib.nullcontext() if (not ddp or k == n_acc - 1) else net.no_sync()
                    with sync:
                        loss = net(input_ids=inp, attention_mask=att, labels=lab).loss
                        w = 0.0 if empty else 1.0 / n_acc
                        (loss * w).backward()
                    step_loss += loss.detach() * w
                opt.step()
                opt.zero_grad(set_to_none=True)
                state["step"] += 1
                state["batch"] = bi + 1
                s = state["step"]
                now = time.time()
                # rank 0 decides time-based saving / stopping; one small all_reduce keeps everyone in step
                ctl = torch.stack([step_loss.float(),
                                   torch.tensor(float(master and now - last_save > a.save_minutes * 60), device=device),
                                   torch.tensor(float(master and now > deadline), device=device)])
                if ddp:
                    dist.all_reduce(ctl)
                lv, want_save, want_stop = ctl.tolist()      # the one host sync of the step
                want_save, want_stop = want_save > 0, want_stop > 0
                loss_sum += lv / world; loss_n += 1
                if not first_done and master:
                    open(flag, "w").write(f"{ph} {s}")
                first_done = True
                last = s >= total
                if s % a.log_every == 0 or s == start_step + 1 or last or want_stop:
                    sps = (s - s_log) / max(1e-9, now - t_log)
                    overall = (s - start_step) / max(1e-9, now - t0)
                    eta_run = (total - s) / max(1e-9, overall)
                    eta_sess = min(eta_run, max(0.0, deadline - now))
                    say(f"[{ph}] step {s}/{total} epoch {state['epoch']} loss {loss_sum / loss_n:.4f} lr {lr_at(s):.2e} "
                        f"{sps:.3f} steps/s elapsed {(now - a.session_start) / 3600:.2f}h "
                        f"ETA {ph} {eta_run / 3600:.2f}h, this session {eta_sess / 3600:.2f}h "
                        f"(~{min(total, s + int(overall * max(0.0, deadline - now)))} steps by the session stop)")
                    t_log, s_log, loss_sum, loss_n = now, s, 0.0, 0
                improved = False
                if ph == "finetune" and (s % a.eval_every == 0 or last):
                    m = evaluate(); m["step"] = s          # every rank: sharded eval, identical result
                    if master:
                        state["history"].append(m)
                        say("eval " + json.dumps(m))
                        with open(os.path.join(out, "eval_log.jsonl"), "a") as f:
                            f.write(json.dumps(m) + "\n")
                    if m["dev_score"] > state["best"]:
                        state["best"] = m["dev_score"]
                        improved = True
                        if master:
                            save_best()
                            say(f"best/ updated (dev_score {m['dev_score']:.4f})")
                    if ddp:
                        dist.barrier()
                if last or want_save or want_stop or improved:
                    save_ckpt("best updated" if improved and not (last or want_save or want_stop) else "")
                    last_save = time.time()
                if want_stop and not last:
                    say(f"SESSION TIME BUDGET REACHED ({a.budget_hours}h): stopped cleanly in phase {ph} at step {s}/{total}. "
                        f"Resume in a new version with kernel_sources [esubechekol/{KERNEL}].")
                    stopped_early = True
                    break
                if last:
                    break
            else:
                state["epoch"] += 1
                state["batch"] = 0
        if stopped_early:
            break
        # phase done
        i = PHASES.index(ph)
        if i + 1 >= len(PHASES):
            break
        if master:
            save_weights(os.path.join(out, "pretrained"), torch.float32)
            say(f"PHASE {ph} DONE at step {state['step']}: weights in pretrained/; starting {PHASES[i + 1]} "
                "from them with a fresh optimizer")
        opt = new_opt()
        state = {"phase": PHASES[i + 1], "step": 0, "epoch": 0, "batch": 0, "best": -1.0, "history": [],
                 "done_phases": state.get("done_phases", []) + [ph]}
        save_ckpt("phase start")

    ph = state["phase"]
    finished = ph == PHASES[-1] and state["step"] >= phase_cfg(ph)["total"]
    if master:
        if stopped_early and ph == "finetune" and state["best"] < 0:   # no eval yet: still leave a best/ snapshot
            save_best()
        with open(os.path.join(out, "run_info.json"), "w") as f:
            json.dump({"run": f"{RUN} " + "+".join(PHASES), "phases": {q: phase_cfg(q) for q in PHASES},
                       "world": world, "precision": "fp32", "finished": finished,
                       "state": {k: v for k, v in state.items() if k != "history"}, "model_commit": a.model_commit,
                       "data_commit": a.data_commit, "torch": torch.__version__}, f, indent=1)
    if ddp:
        dist.barrier()
    if finished:
        say(f"TRAINING DONE at {ph} step {state['step']}, best dev_score {state['best']:.4f}")
        if not a.no_final_report and os.path.exists(os.path.join(out, "best")):
            del net, model, opt, train      # free the training copies before final_report loads best/
            import gc; gc.collect()
            if cuda:
                torch.cuda.empty_cache()
            say("final greedy dev report (sharded across ranks)")
            tt.final_report(args, dev, torch, device, None, rank=rank, world=world, master=master,
                            xm=Gather if ddp else None)
    else:
        say(f"SESSION DONE in phase {ph} at step {state['step']}/{phase_cfg(ph)['total']}, best dev_score "
            f"{state['best']:.4f}; checkpoint in ckpt/")
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    worker() if a.worker else launcher()
