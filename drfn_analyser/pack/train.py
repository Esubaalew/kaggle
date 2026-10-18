"""Training loop for the CMM thesis rebuild.

Protocol (2026-08-01 report):
    Adam, lr 5e-4, linear warmup 400 steps, then reduce-on-plateau on dev
    translation loss. Batch 32, max len 128, patience 8 epochs, label
    smoothing 0.1. Checkpoint selection by DEV translation loss.
    Seeds 11 / 22 / 33.

Loss (per report; the 0.5/0.5/0.01/0.01 weights come from the report text,
not from any earlier plan):
    L = L_trans(label_smooth=0.1)
      + 0.5 * CE(root_head, root_labels)
      + 0.5 * CE(pattern_head, pattern_labels)
      - 0.01 * mean_routing_entropy        (anti-collapse bonus)
      + 0.01 * KL(batch_mean_assignment || uniform)   (balance penalty)

Checkpoint resume: model, optimizer, scheduler state, epoch, global step,
and all RNG states (torch cpu/cuda, numpy, python). --resume continues a run.

Per-run metadata JSON: git hash (if available), full config, seed, sha256 of
train/dev data, tokenizer and label cache, timestamps, torch version.

The TEST split is never loaded here. evaluate.py refuses test paths too.
"""

import argparse
import copy
import json
import math
import os
import random
import socket
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as data_mod
from data import (load_pairs, train_bpe, load_tokenizer, LabelCache,
                  make_loader, sha256_file)
from model import build_model, count_params, VARIANTS


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def git_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(
                os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def routing_stats(routings, eps=1e-9):
    """Entropy bonus and balance penalty over CMM/flat routing distributions."""
    if not routings:
        return None, None
    ents, bals = [], []
    for info in routings:
        for key in ("root_scores", "pattern_scores", "scores"):
            if key not in info:
                continue
            s = info[key].clamp_min(eps)  # (b, t, slots)
            ent = -(s * s.log()).sum(-1).mean()
            mean_assign = s.mean(dim=(0, 1))
            uniform = torch.full_like(mean_assign, 1.0 / mean_assign.size(-1))
            bal = (mean_assign * (mean_assign.log() - uniform.log())).sum()
            ents.append(ent)
            bals.append(bal)
    if not ents:
        return None, None
    return torch.stack(ents).mean(), torch.stack(bals).mean()


def compute_loss(out, tgt, root_lab, pat_lab, cfg):
    # Model was fed tgt[:, :-1]; logits align with tgt[:, 1:].
    tgt_out = tgt[:, 1:]
    logits = out["logits"]
    l_trans = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1),
        ignore_index=cfg["pad_id"], label_smoothing=cfg["label_smoothing"])
    total = l_trans
    terms = {"trans": l_trans.detach()}
    if "root_logits" in out:
        rl = out["root_logits"]
        l_root = F.cross_entropy(rl.reshape(-1, rl.size(-1)),
                                 root_lab.reshape(-1),
                                 ignore_index=cfg["aux_ignore_index"])
        pl = out["pattern_logits"]
        l_pat = F.cross_entropy(pl.reshape(-1, pl.size(-1)),
                                pat_lab.reshape(-1),
                                ignore_index=cfg["aux_ignore_index"])
        total = total + cfg["aux_root_weight"] * l_root + \
            cfg["aux_pattern_weight"] * l_pat
        terms["root"] = l_root.detach()
        terms["pattern"] = l_pat.detach()
    ent, bal = routing_stats(out.get("routings", []))
    if ent is not None:
        total = total - cfg["entropy_reg"] * ent + cfg["balance_reg"] * bal
        terms["entropy"] = ent.detach()
        terms["balance"] = bal.detach()
    return total, terms


@torch.no_grad()
def dev_loss(model, loader, cfg, device):
    model.eval()
    tot, n = 0.0, 0
    for src, tgt, root_lab, pat_lab in loader:
        src, tgt = src.to(device), tgt.to(device)
        root_lab, pat_lab = root_lab.to(device), pat_lab.to(device)
        out = model(src, tgt[:, :-1])
        logits = out["logits"]
        l = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                            tgt[:, 1:].reshape(-1),
                            ignore_index=cfg["pad_id"])
        tot += l.item() * src.size(0)
        n += src.size(0)
    return tot / max(n, 1)


@torch.no_grad()
def greedy_chrf(model, pairs, src_tokenizer, tgt_tokenizer, cfg, device,
                limit=256):
    from sacrebleu import corpus_chrf
    model.eval()
    hyps, refs = [], []
    for src, tgt in pairs[:limit]:
        ids, _ = data_mod.encode_words(src_tokenizer, src, cfg["bos_id"],
                                       cfg["eos_id"], cfg["max_len"])
        src_t = torch.tensor([ids], dtype=torch.long, device=device)
        ys = model.greedy_decode(src_t, cfg["max_len"], cfg["bos_id"],
                                 cfg["eos_id"])[0].tolist()
        toks = []
        for t in ys[1:]:
            if t == cfg["eos_id"]:
                break
            if t in (cfg["pad_id"], cfg["bos_id"]):
                continue
            toks.append(t)
        hyps.append(tgt_tokenizer.decode(toks))
        refs.append(tgt)
    return corpus_chrf(hyps, [refs], word_order=2).score if hyps else 0.0


def save_checkpoint(path, model, optimizer, scheduler, epoch, step,
                    best_dev, cfg):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "step": step, "best_dev": best_dev,
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available() else None,
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
        "cfg": cfg,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler and ck.get("scheduler"):
        scheduler.load_state_dict(ck["scheduler"])
    torch.set_rng_state(ck["rng_torch"].cpu())
    if ck.get("rng_cuda") and torch.cuda.is_available():
        try:
            states = []
            for s in ck["rng_cuda"]:
                if not torch.is_tensor(s):
                    s = torch.tensor(list(s))
                states.append(s.detach().cpu().byte())
            torch.cuda.set_rng_state_all(states)
        except Exception as exc:
            print(f"skip cuda rng: {exc}", flush=True)
    np.random.set_state(ck["rng_numpy"])
    random.setstate(ck["rng_python"])
    return ck["epoch"], ck["step"], ck["best_dev"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--label-cache", default=None)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU run: 256 train pairs, 100 steps max")
    ap.add_argument("--eval-chrf-limit", type=int, default=256)
    ap.add_argument("--no-param-match", action="store_true")
    ap.add_argument("--src-tokenizer", default="bpe", choices=["bpe", "morpheme", "bpe_root", "analyser_input"],
                    help="source-side tokenizer; target is always BPE "
                         "(task 2: geezRPA morphemes instead of BPE)")
    ap.add_argument("--ffn-dim", type=int, default=None,
                    help="fix the FFN width and switch parameter matching off (only embedding rows then differ)")
    ap.add_argument("--analyser-cache", default=None,
                    help="analyser readings JSON (analyser_cache.json) for --src-tokenizer analyser_input")
    ap.add_argument("--min-root-count", type=int, default=2,
                    help="analyser_input: a word enters as its reading only if its root occurs this often in train")
    ap.add_argument("--analyser-labels", default=None,
                    help="analyser label cache JSON for --src-tokenizer bpe_root")
    ap.add_argument("--morpheme-vocab", default=None,
                    help="morpheme vocab JSON (required with "
                         "--src-tokenizer morpheme)")
    ap.add_argument("--aux-root-weight", type=float, default=None,
                    help="override cfg aux_root_weight (0 = pattern-only; "
                         "default 0.5 keeps both-label bpe_aux)")
    ap.add_argument("--aux-pattern-weight", type=float, default=None,
                    help="override cfg aux_pattern_weight (0 = root-only; "
                         "default 0.5 keeps both-label bpe_aux)")
    ap.add_argument("--n-root-slots", type=int, default=None,
                    help="override configs/base.json n_root_slots (task 4)")
    ap.add_argument("--n-pattern-slots", type=int, default=None,
                    help="override configs/base.json n_pattern_slots")
    args = ap.parse_args()

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "configs", "base.json")) as f:
        cfg = json.load(f)
    cfg["variant"] = args.variant
    cfg["seed"] = args.seed
    if args.n_root_slots:
        cfg["n_root_slots"] = args.n_root_slots
    if args.n_pattern_slots:
        cfg["n_pattern_slots"] = args.n_pattern_slots
    # Root-only / pattern-only: same bpe_aux heads, other aux weight zeroed.
    # Omitted flags leave base.json (0.5/0.5) so both-label bpe_aux is unchanged.
    if args.aux_root_weight is not None:
        cfg["aux_root_weight"] = args.aux_root_weight
    if args.aux_pattern_weight is not None:
        cfg["aux_pattern_weight"] = args.aux_pattern_weight
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} variant={args.variant} seed={args.seed}", flush=True)

    os.makedirs(args.run_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "age_train.tsv")
    dev_path = os.path.join(args.data_dir, "age_dev.tsv")
    train_pairs, skip_tr = load_pairs(train_path)
    dev_pairs, skip_dv = load_pairs(dev_path)
    if args.smoke:
        train_pairs = train_pairs[:256]
        dev_pairs = dev_pairs[:64]
    print(f"train={len(train_pairs)} (skipped {skip_tr}) "
          f"dev={len(dev_pairs)} (skipped {skip_dv})", flush=True)

    tok_path = args.tokenizer or os.path.join(args.run_dir, "tokenizer.json")
    if not os.path.exists(tok_path):
        print("training BPE ...", flush=True)
        train_bpe(train_pairs, cfg["vocab_size"], tok_path, seed=args.seed)
    tokenizer = load_tokenizer(tok_path)
    # Keep the run self-contained: copy the tokenizer into the run dir.
    run_tok = os.path.join(args.run_dir, "tokenizer.json")
    if os.path.abspath(tok_path) != os.path.abspath(run_tok):
        import shutil
        shutil.copyfile(tok_path, run_tok)
    tokenizer_hash = sha256_file(tok_path)

    # Source-side tokenizer: BPE by default (unchanged behavior); the
    # morpheme variant uses geezRPA segmentation for Ge'ez while the
    # Amharic target always stays BPE.
    morpheme_vocab_hash = None
    if args.src_tokenizer == "morpheme":
        if not args.morpheme_vocab or not os.path.exists(args.morpheme_vocab):
            raise SystemExit("--src-tokenizer morpheme needs --morpheme-vocab")
        src_tokenizer = data_mod.MorphemeTokenizer(args.morpheme_vocab)
        run_mv = os.path.join(args.run_dir, "morpheme_vocab.json")
        if os.path.abspath(args.morpheme_vocab) != os.path.abspath(run_mv):
            import shutil as _sh
            _sh.copyfile(args.morpheme_vocab, run_mv)
        morpheme_vocab_hash = sha256_file(args.morpheme_vocab)
        cfg["src_vocab_size"] = len(src_tokenizer)
        cfg["tgt_vocab_size"] = cfg["vocab_size"]
        print(f"src tokenizer: morpheme ({len(src_tokenizer)} tokens), "
              f"tgt tokenizer: BPE ({cfg['vocab_size']})", flush=True)
    elif args.src_tokenizer == "bpe_root":
        # BPE pieces of each Ge'ez word plus one token for the analyser's root class (top-255 roots).
        if not (args.analyser_labels and os.path.exists(args.analyser_labels)):
            raise SystemExit("--src-tokenizer bpe_root needs --analyser-labels")
        from analyser_tokenizer import BpeRootTokenizer
        import shutil as _sh
        src_tokenizer = BpeRootTokenizer(tokenizer, args.analyser_labels, cfg["vocab_size"])
        dst = os.path.join(args.run_dir, "analyser_labels_for_input.json")
        if os.path.abspath(args.analyser_labels) != os.path.abspath(dst):
            _sh.copyfile(args.analyser_labels, dst)
        morpheme_vocab_hash = sha256_file(args.analyser_labels)
        cfg["src_vocab_size"] = len(src_tokenizer)
        cfg["tgt_vocab_size"] = cfg["vocab_size"]
        print(f"src tokenizer: BPE + analyser root token ({len(src_tokenizer)} ids), "
              f"tgt tokenizer: BPE ({cfg['vocab_size']})", flush=True)
    elif args.src_tokenizer == "analyser_input":
        # The analyser replaces BPE on the Ge'ez side: a word with a reading (root occurring >= min_root_count times in
        # train) enters as root + feature tokens; every other word keeps its BPE pieces.
        if not (args.analyser_cache and os.path.exists(args.analyser_cache)):
            raise SystemExit("--src-tokenizer analyser_input needs --analyser-cache")
        from analyser_tokenizer import AnalyserInputTokenizer
        import shutil as _sh
        with open(args.analyser_cache, encoding="utf-8") as _f:
            _readings = json.load(_f)["readings"]
        _train_words = [w for src_, _ in train_pairs for w in src_.split()]
        src_tokenizer = AnalyserInputTokenizer(tokenizer, _readings, _train_words, cfg["vocab_size"],
                                               args.min_root_count)
        src_tokenizer.save(os.path.join(args.run_dir, "analyser_input_vocab.json"))
        morpheme_vocab_hash = sha256_file(args.analyser_cache)
        cfg["src_vocab_size"] = len(src_tokenizer)
        cfg["tgt_vocab_size"] = cfg["vocab_size"]
        print(f"src tokenizer: analyser input ({len(src_tokenizer)} ids; "
              f"{src_tokenizer.n_analysed}/{src_tokenizer.n_words} train words enter as their reading, "
              f"the rest as BPE pieces), tgt tokenizer: BPE ({cfg['vocab_size']})", flush=True)
    else:
        src_tokenizer = tokenizer
        cfg["src_vocab_size"] = cfg["vocab_size"]
        cfg["tgt_vocab_size"] = cfg["vocab_size"]
    tgt_tokenizer = tokenizer

    needs_labels = args.variant != "plain_bpe"
    cache = None
    if needs_labels:
        if not args.label_cache:
            raise SystemExit("variant needs --label-cache")
        cache = LabelCache(args.label_cache)
        print(f"label cache: {cache.n_root} root classes, "
              f"{cache.n_pattern} pattern classes", flush=True)
    n_root = cache.n_root if cache else 2
    n_pat = cache.n_pattern if cache else 2
    cfg["n_root_classes"] = n_root
    cfg["n_pattern_classes"] = n_pat

    placebo = args.variant in ("bpe_placebo", "cmm_placebo")
    train_loader = make_loader(train_pairs, tokenizer, cache, cfg, "train",
                               shuffle=True, placebo=placebo,
                               placebo_seed=args.seed,
                               src_tokenizer=src_tokenizer,
                               tgt_tokenizer=tgt_tokenizer)
    dev_loader = make_loader(dev_pairs, tokenizer, cache, cfg, "dev",
                             shuffle=False,
                             src_tokenizer=src_tokenizer,
                             tgt_tokenizer=tgt_tokenizer)

    if args.ffn_dim:
        # Fixed transformer body (same FFN width as the run it is compared with); parameter matching is off, so a
        # larger source vocabulary only adds embedding rows. build_model matches whenever cfg has target_params,
        # so it is removed from the copy passed in.
        cfg["ffn_dim"] = args.ffn_dim
        _cfg = {k: v for k, v in cfg.items() if k != "target_params"}
        model = build_model(_cfg, n_root, n_pat, target_params=None)
    else:
        target = None if args.no_param_match else cfg["target_params"]
        model = build_model(cfg, n_root, n_pat, target_params=target,
                            tol=cfg["param_match_tol"])
    cfg["ffn_dim"] = model.cfg["ffn_dim"]  # record the matched width
    print(f"params={count_params(model)} ffn_dim={model.cfg['ffn_dim']}",
          flush=True)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"],
                           betas=tuple(cfg["adam_betas"]), eps=cfg["adam_eps"])
    scheduler = ReduceLROnPlateau(opt, mode="min",
                                  factor=cfg["plateau_factor"],
                                  patience=cfg["plateau_patience"])

    start_epoch, step, best_dev = 0, 0, float("inf")
    # Continue a dead run from last.pt even when the shell forgot --resume.
    resume_path = args.resume
    if not resume_path:
        candidate = os.path.join(args.run_dir, "last.pt")
        if os.path.isfile(candidate):
            resume_path = candidate
    if resume_path:
        start_epoch, step, best_dev = load_checkpoint(
            resume_path, model, opt, scheduler, device)
        print(f"resumed epoch={start_epoch} step={step} "
              f"best_dev={best_dev:.4f} path={resume_path}", flush=True)

    meta = {
        "variant": args.variant, "seed": args.seed, "cfg": cfg,
        "git_hash": git_hash(), "hostname": socket.gethostname(),
        "torch_version": torch.__version__, "device": str(device),
        "data_hashes": {"train": sha256_file(train_path),
                        "dev": sha256_file(dev_path)},
        "tokenizer_hash": tokenizer_hash,
        "src_tokenizer": args.src_tokenizer,
        "morpheme_vocab_hash": morpheme_vocab_hash,
        "label_cache_hash": sha256_file(args.label_cache)
        if args.label_cache else None,
        "params": count_params(model),
        "placebo": placebo,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    with open(os.path.join(args.run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)

    log_path = os.path.join(args.run_dir, "train_log.jsonl")
    logf = open(log_path, "a" if resume_path else "w")
    warmup = cfg["warmup_steps"]
    max_steps = 100 if args.smoke else None
    epochs = 1 if args.smoke else args.epochs
    bad_epochs = 0
    stop = False

    for epoch in range(start_epoch, epochs):
        model.train()
        for src, tgt, root_lab, pat_lab in train_loader:
            if max_steps and step >= max_steps:
                stop = True
                break
            src, tgt = src.to(device), tgt.to(device)
            root_lab, pat_lab = root_lab.to(device), pat_lab.to(device)
            if step < warmup:  # linear warmup
                lr = cfg["lr"] * (step + 1) / warmup
                for pg in opt.param_groups:
                    pg["lr"] = lr
            opt.zero_grad()
            out = model(src, tgt[:, :-1])
            loss, terms = compute_loss(out, tgt, root_lab, pat_lab, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % 50 == 0:
                rec = {"step": step, "epoch": epoch,
                       **{k: round(v.item(), 4) for k, v in terms.items()},
                       "lr": opt.param_groups[0]["lr"]}
                logf.write(json.dumps(rec) + "\n")
                logf.flush()
                print(rec, flush=True)

        dloss = dev_loss(model, dev_loader, cfg, device)
        chrf = greedy_chrf(model, dev_pairs, src_tokenizer, tgt_tokenizer,
                           cfg, device,
                           limit=args.eval_chrf_limit if not args.smoke else 32)
        print(f"epoch={epoch} dev_loss={dloss:.4f} dev_chrf++={chrf:.2f}",
              flush=True)
        logf.write(json.dumps({"epoch": epoch, "dev_loss": dloss,
                               "dev_chrf": chrf}) + "\n")
        logf.flush()
        if step >= warmup:
            scheduler.step(dloss)
        if dloss < best_dev:
            best_dev, bad_epochs = dloss, 0
            save_checkpoint(os.path.join(args.run_dir, "best.pt"), model, opt,
                            scheduler, epoch + 1, step, best_dev, cfg)
            print(f"  new best dev_loss={dloss:.4f}", flush=True)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["early_stop_patience"]:
                print(f"early stopping at epoch {epoch}", flush=True)
                stop = True
        save_checkpoint(os.path.join(args.run_dir, "last.pt"), model, opt,
                        scheduler, epoch + 1, step, best_dev, cfg)
        if stop:
            break

    meta["ended_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["best_dev_loss"] = best_dev
    meta["final_step"] = step
    with open(os.path.join(args.run_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=1)
    logf.close()
    print(f"done. best_dev_loss={best_dev:.4f} run_dir={args.run_dir}")


if __name__ == "__main__":
    main()
