"""Apply to version D train.py (advisor-update-2019 code): adds --src-tokenizer bpe_root and --analyser-labels."""
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
a = 'choices=["bpe", "morpheme"],'
assert a in s, "not the version D train.py"
s = s.replace(a, 'choices=["bpe", "morpheme", "bpe_root"],')
b = '    ap.add_argument("--morpheme-vocab", default=None,'
s = s.replace(b, '    ap.add_argument("--analyser-labels", default=None,\n'
                 '                    help="analyser label cache JSON for --src-tokenizer bpe_root")\n' + b, 1)
c = '    else:\n        src_tokenizer = tokenizer\n        cfg["src_vocab_size"] = cfg["vocab_size"]'
assert c in s
new = '''    elif args.src_tokenizer == "bpe_root":
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
'''
s = s.replace(c, new + c, 1)
open(p, "w", encoding="utf-8").write(s)
print("patched", p)
