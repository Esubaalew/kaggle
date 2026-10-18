"""Report: Ge'ez ByT5 morphological analyser, GPU run A (fine-tune only). Reads the run files directly."""
import json, re, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

K = '/Users/esubalew/is/researcher/kaggle/'
OUT = K + 'geez_gpu_v2/out_v6/'
info = json.load(open(OUT + 'run_info.json'))
dev = json.load(open(OUT + 'dev_report.json'))
mix = json.load(open(OUT + 'mixture_report.json'))
leak = open(OUT + 'leak_audit.txt').read()
cfg = json.load(open(OUT + 'best/config.json'))


def log_lines(path):
    return [e for e in json.load(open(path))]


sessions = [K + 'log_a/geez-byt5-gpu-v2-a.log', OUT + 'geez-byt5-gpu-v2-a.log']
loss, evals_a = [], []
sess_hours = []
for p in sessions:
    L = log_lines(p); sess_hours.append(L[-1]['time'] / 3600)
    for e in L:
        d = e['data']
        m = re.match(r'\[finetune\] step (\d+)/\d+ epoch (\d+) loss ([\d.]+) lr ([\d.e+-]+) ([\d.]+) steps/s', d)
        if m: loss.append((int(m[1]), float(m[3]), float(m[5]), float(m[4])))
        if d.startswith('eval '): evals_a.append(json.loads(d[5:]))
for line in open(K + 'log_a/eval_log.jsonl'):
    if line.strip():
        r = json.loads(line)
        if r['step'] not in [x['step'] for x in evals_a]: evals_a.append(r)
evals_a.sort(key=lambda r: r['step'])
evals_b = [json.loads(l) for l in open(K + 'log_b/eval_log.jsonl') if l.strip()]
loss = np.array(loss)

# ---- best-checkpoint dev numbers -------------------------------------------------------------------
SL = dev['by_slice_task_class']
def agg(slc, task, key):
    rows = {k.split('/')[2]: v for k, v in SL.items() if k.startswith(f'{slc}/{task}/')}
    return rows
an_n = dev['analyze_total']['n']; exact = dev['analyze_total']['exact']; root = dev['analyze_total']['root']
pos_ok = sum(v['exact'] for k, v in SL.items() if '/pos/' in k); pos_n = sum(v['n'] for k, v in SL.items() if '/pos/' in k)
pos = pos_ok / pos_n
assert abs((exact + root) / 2 - info['state']['best']) < 1e-9, 'selection score must equal mean(exact, root)'

# ---- figures ------------------------------------------------------------------------------------------
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
                     'legend.fontsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 300})
CA, CB = '#1f4e8c', '#c46b1a'
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
gap = lambda x, y: (np.insert(np.asarray(x, float), np.searchsorted(x, 5000), np.nan), np.insert(np.asarray(y, float), np.searchsorted(x, 5000), np.nan))
a1.plot(*gap(loss[:, 0], loss[:, 1]), color=CA, lw=0.9, marker='.', ms=2.5)
a1.set_yscale('log'); a1.set_xlabel('optimiser step'); a1.set_ylabel('training loss (log scale)')
a1.axvspan(3676, 7530, color='#dddddd', lw=0)
a1.text(5600, 0.5, 'session 2\nlog not\ndownloaded', ha='center', va='center', fontsize=6.5, color='#555555')
a1.set_title('Training loss falls from 5.7 to about 0.01', loc='left')
a1.set_yticks([0.01, 0.1, 1, 5]); a1.set_yticklabels(['0.01', '0.1', '1', '5']); a1.margins(x=0.02)
steps = [r['step'] for r in evals_a]
for key, lab, mk in [('dev_root', 'root accuracy', 'o'), ('dev_pos', 'part of speech', 's'), ('dev_exact', 'full analysis exact', '^')]:
    a2.plot(*gap(steps, [100 * r[key] for r in evals_a]), color=CA, marker=mk, ms=3.5, lw=1.0)
    a2.plot([r['step'] for r in evals_b], [100 * r[key] for r in evals_b], color=CB, marker=mk, ms=3.5, lw=0.9, ls='--', mfc='white')
    a2.text(11250, 100 * evals_a[-1][key], lab, va='center', fontsize=6.5, color=CA)
a2.axvspan(3676, 7530, color='#dddddd', lw=0)
a2.set_xlim(0, 15500); a2.set_xticks([0, 2500, 5000, 7500, 10000])
a2.set_xlabel('optimiser step'); a2.set_ylabel('dev accuracy (%)'); a2.set_ylim(58, 95)
a2.set_title('Dev accuracy plateaus after about 3,000 steps', loc='left')
a2.plot([], [], color=CA, label='run A (fine-tune only)'); a2.plot([], [], color=CB, ls='--', label='run B (pretrain + fine-tune), cancelled')
a2.legend(frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.0))
for ax, l in [(a1, 'a'), (a2, 'b')]:
    ax.text(-0.14, 1.06, l, transform=ax.transAxes, fontweight='bold', fontsize=10)
fig.tight_layout(); fig.savefig('fig_training.png'); plt.close(fig)

# POS confusion (true rows with >= 4 items)
conf = dev['pos_confusion']
rows = [c for c in sorted(conf, key=lambda c: -sum(conf[c].values())) if sum(conf[c].values()) >= 4]
cols = sorted({p for c in rows for p in conf[c]}, key=lambda p: (p not in rows, rows.index(p) if p in rows else 0, p))
M = np.array([[conf[r].get(c, 0) for c in cols] for r in rows], float)
Mn = M / M.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(6.2, 3.0))
im = ax.imshow(Mn, cmap='Blues', vmin=0, vmax=1, aspect='auto')
for i in range(len(rows)):
    for j in range(len(cols)):
        if M[i, j]:
            ax.text(j, i, int(M[i, j]), ha='center', va='center', fontsize=6.5, color='white' if Mn[i, j] > 0.55 else '#222222')
ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha='right')
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{r} (n={int(M[i].sum())})' for i, r in enumerate(rows)])
ax.set_xlabel('predicted part of speech'); ax.set_ylabel('true part of speech')
ax.set_title('Part-of-speech errors: infinitives read as nouns, nouns as adjectives', loc='left')
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label('share of the true class')
for s in ax.spines.values(): s.set_visible(False)
fig.tight_layout(); fig.savefig('fig_pos_confusion.png'); plt.close(fig)

# ---- PDF ---------------------------------------------------------------------------------------------
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
KEFA = '/System/Library/Fonts/Supplemental/KefaIII.ttf'
ETH = 'Helvetica'
if os.path.exists(KEFA): pdfmetrics.registerFont(TTFont('Kefa', KEFA)); ETH = 'Kefa'
E = lambda t: f'<font name="{ETH}">{t}</font>'
ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=15.5, textColor=colors.HexColor('#1f3a5f'), spaceAfter=4)
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=12, textColor=colors.HexColor('#1f3a5f'), spaceBefore=8, spaceAfter=3)
B = ParagraphStyle('B', parent=ss['BodyText'], fontSize=9.4, leading=12.8, spaceAfter=4)
S = ParagraphStyle('S', parent=B, fontSize=8.2, leading=10.6)
BOX = ParagraphStyle('BOX', parent=B, backColor=colors.HexColor('#eef3f8'), borderPadding=6, borderColor=colors.HexColor('#9db4cc'), borderWidth=0.6, spaceBefore=4, spaceAfter=8)
P = lambda t, st=B: Paragraph(t, st)
pct = lambda x: f'{100 * x:.1f}%'


def table(rows, widths, fs=8.2, align_right_from=1):
    data = [[Paragraph(str(c), ParagraphStyle('t', parent=S, fontSize=fs, leading=fs + 2.3)) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#b0b0b0')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                           ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d6e2ef')), ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    return t


ph = info['phases']['finetune']
n_params = None
story = [P("Ge'ez Morphological Analyser (ByT5): Training Report, Run A", H1),
         P(f"geez-morphology-model commit {info['model_commit'][:7]} &middot; geez-dataset commit {info['data_commit'][:7]} &middot; "
           f"Kaggle notebook esubechekol/geez-byt5-gpu-v2-a &middot; report built 18 October 2026", S),
         P(f"<b>Result.</b> Run A trained to completion ({ph['total']:,} steps, {ph['epochs']:.0f} epochs). The selected checkpoint "
           f"analyses held-out Ge'ez words with <b>{pct(root)} root accuracy</b>, <b>{pct(exact)} exact full analysis</b> (root and every feature) "
           f"and <b>{pct(pos)} part-of-speech accuracy</b> on {an_n:,} held-out validation words whose roots never occur in training. "
           f"This model is the planned label source for the next translation experiments. These are validation numbers used for model selection; "
           f"the separate held-out test set (2,310 pairs) has not been scored yet.", BOX)]

story += [P('1. What the model does', H2),
          P(f"The analyser reads one Ge'ez word, character by character (bytes), and writes its morphological analysis as text. "
            f"It is trained on three tasks at once, marked by a task prefix: <b>analyse</b> (word &rarr; root + pattern features), "
            f"<b>generate</b> (root + features &rarr; word) and <b>part of speech</b> (word &rarr; word class). "
            f"For example, {E('ሰገዱ')} should be analysed as root s-g-d with perfective, 3rd person, plural, masculine features, "
            f"and tagged as a verb."),
          table([['Item', 'Value'],
                 ['Architecture', f"ByT5-small encoder&ndash;decoder (T5, gated-GELU): {cfg['num_layers']} encoder / {cfg['num_decoder_layers']} decoder layers, "
                                  f"d_model {cfg['d_model']}, feed-forward {cfg['d_ff']}, {cfg['num_heads']} heads, byte vocabulary {cfg['vocab_size']}"],
                 ['Run type', 'A = fine-tuning from the public ByT5-small checkpoint (no Ge\'ez pretraining). Run B adds Ge\'ez span-corruption pretraining first.'],
                 ['Hardware', f"Kaggle, 2 &times; NVIDIA T4, PyTorch {info['torch']} DDP, {info['precision']}"],
                 ['Batch', f"{ph['micro']} rows &times; {info['world']} GPUs &times; 8 accumulation = {ph['G']} rows per step; {ph['spe']:,} steps per epoch"],
                 ['Schedule', f"{ph['total']:,} steps = {ph['epochs']:.0f} epochs; warm-up {ph['warmup']} steps; logged learning rate peaks at {loss[:, 3].max():.1e} and decays to {loss[-1, 3]:.0e}"],
                 ['Model selection', 'every 1,000 steps on the validation set; score = mean of exact-analysis accuracy and root accuracy; best checkpoint kept in best/'],
                 ['Wall-clock', f"three Kaggle sessions of about 11 h each (session 1: {sess_hours[0]:.1f} h, steps 1&ndash;3,676; session 2: steps 3,677&ndash;7,530; "
                                f"session 3: {sess_hours[1]:.1f} h, steps 7,531&ndash;10,980 plus the final report), about 33 h in total, "
                                f"at a median {np.median(loss[:, 2]):.3f} steps/s"]],
                [3.0 * cm, 14.0 * cm])]

ps = mix['pairs_by_source']; cls = mix['samples_by_class']
src_rows = [['Source', 'Word pairs', 'What it is']]
desc = {'synthetic': 'forms generated from checked roots and paradigm rules', 'unified_book': 'forms transcribed from printed grammars and books (unified table)',
        'gold_nominal': 'hand-checked noun/adjective paradigm cells', 'gold_verb': 'hand-checked verb paradigm cells', 'dictionary': 'dictionary headwords with checked roots',
        'dictionary_verb': 'dictionary verbs with checked roots', 'extra_synthetic': 'additional checked generated cells', 'word_without_root': 'function words labelled as rootless'}
for k, v in sorted(ps.items(), key=lambda kv: -kv[1]):
    src_rows.append([k, f'{v:,}', desc.get(k, '')])
src_rows.append(['total', f"{sum(ps.values()):,}", f"expanded to {mix['train_samples']:,} training samples over the three tasks"])
top_cls = sorted(cls.items(), key=lambda kv: -kv[1])
story += [P('2. Training data', H2),
          P("All data come from the author's geez-dataset (commit 04f45a7). Before training, the script checks that the data load exactly as expected and "
            "that no validation or test root, and no validation or test word form, appears in training."),
          table(src_rows, [3.2 * cm, 2.4 * cm, 11.4 * cm]),
          Spacer(1, 4),
          P('<b>Leakage audit (from the run).</b> Training: 562,403 samples, 1,946 distinct roots. Validation: 1,736 pairs, 147 roots; test: 2,310 pairs, 217 roots. Training pairs sharing a root, a word form, or a spelling-variant root with '
            'validation or test: <b>0 in every check</b> (RESULT: CLEAN).', S),
          P('<b>Class balance.</b> Verbs dominate the training samples: ' +
            ', '.join(f'{k} {v:,} ({100 * v / mix["train_samples"]:.1f}%)' for k, v in top_cls[:4]) +
            f'; all other classes together {100 * sum(v for _, v in top_cls[4:]) / mix["train_samples"]:.1f}%. '
            f'Synthetic forms are {100 * ps["synthetic"] / sum(ps.values()):.0f}% of word pairs.', S)]

story += [PageBreak(), P('3. Training curve', H2),
          Image('fig_training.png', width=17 * cm, height=17 * cm * 2.7 / 7.0),
          P('<b>Figure 1.</b> (a) Training loss per logged step. (b) Validation accuracy every 1,000 steps for run A (solid) and the cancelled run B '
            '(dashed, steps 1,000&ndash;3,000 only). The grey band marks session 2 (steps 3,677&ndash;7,530), whose log and evaluations have not been '
            'downloaded; the best checkpoint was selected inside this band (selection score 0.8088, above every logged evaluation).', S),
          table([['Step'] + [f"{r['step']:,}" for r in evals_a],
                 ['Root accuracy'] + [pct(r['dev_root']) for r in evals_a],
                 ['Exact analysis'] + [pct(r['dev_exact']) for r in evals_a],
                 ['Part of speech'] + [pct(r['dev_pos']) for r in evals_a],
                 ['Selection score'] + [f"{r['dev_score']:.4f}" for r in evals_a]],
                [2.6 * cm] + [2.05 * cm] * len(evals_a)),
          P(f"Validation accuracy rises quickly to step 3,000 and then changes by less than one point. The final step (10,980) scores "
            f"{evals_a[-1]['dev_score']:.4f}, slightly below the best checkpoint ({info['state']['best']:.4f}), so the last epochs mainly fit the "
            f"training set further (loss about 0.01) without improving held-out words. Run B matched run A at step 3,000 on root accuracy "
            f"({pct(evals_b[-1]['dev_root'])} both) but was lower on exact analysis ({pct(evals_b[-1]['dev_exact'])} vs {pct(evals_a[2]['dev_exact'])}); "
            f"B was cancelled, so no conclusion about pretraining is drawn.")]

def cls_rows(slc):
    out = []
    for c, v in sorted(agg(slc, 'analyze', None).items(), key=lambda kv: -kv[1]['n']):
        pv = SL.get(f'{slc}/pos/{c}')
        out.append([c, v['n'], pct(v['root'] / v['n']), pct(v['exact'] / v['n']), pct(pv['exact'] / pv['n']) if pv else '-'])
    return out
gold = cls_rows('gold_val'); uni = cls_rows('unified_val')
ex = lambda rows, mn=20: [float(r[3][:-1]) for r in rows if r[1] >= mn]
g_lo, g_hi = min(ex(gold)), max(ex(gold)); u_lo, u_hi = min(ex(uni)), max(ex(uni))
noun_tot = sum(conf['noun'].values()); noun_adj = conf['noun'].get('adjective', 0); inf_tot = sum(conf['infinitive'].values()); inf_noun = conf['infinitive'].get('noun', 0)
story += [PageBreak(), P('4. Results of the selected checkpoint', H2),
          P(f"Validation words come from two sources: <b>gold paradigm</b> cells (hand-checked tables) and <b>book/unified</b> rows "
            f"(forms transcribed from printed sources). Root accuracy counts the predicted root only; exact analysis requires the root and every feature to be right."),
          table([['Source / class', 'Words', 'Root', 'Exact analysis', 'Part of speech']] +
                [['<b>gold paradigms</b>', '', '', '', '']] + gold + [['<b>book / unified rows</b>', '', '', '', '']] + uni +
                [['<b>all validation words</b>', f'{an_n:,}', f'<b>{pct(root)}</b>', f'<b>{pct(exact)}</b>', f'<b>{pct(pos)}</b>']],
                [4.2 * cm, 1.8 * cm, 2.6 * cm, 3.4 * cm, 3.0 * cm]),
          Spacer(1, 6),
          Image('fig_pos_confusion.png', width=15.5 * cm, height=15.5 * cm * 3.0 / 6.2),
          P('<b>Figure 2.</b> Part-of-speech confusion on validation words (classes with at least 4 words; counts in cells, colour = share of the true class).', S)]

story += [P('5. What the numbers mean', H2),
          P('&bull; <b>Roots are reliable.</b> 92% of held-out words get the right root, including words whose root was never seen in training. '
            'This is the property the translation memory needs, because memory slots are tied to roots.<br/>'
            f'&bull; <b>Full analyses are harder.</b> {100*exact:.0f}% are exactly right. Gold paradigm cells are easier ({g_lo:.0f}&ndash;{g_hi:.0f}% by class) than book rows ({u_lo:.0f}&ndash;{u_hi:.0f}% for classes with at least 20 words), which are more varied and closer to running text.<br/>'
            f'&bull; <b>Weak spots.</b> Infinitives are almost always tagged as nouns ({inf_noun} of {inf_tot}); nouns are tagged as adjectives in {noun_adj} of {noun_tot} cases; '
            'pronouns, prepositions and demonstratives are rare in training (under 0.2% of samples each) and are often wrong.<br/>'
            '&bull; <b>Over-fitting is mild.</b> Training loss keeps falling to about 0.01 while validation stays flat after about 3,000 steps; '
            'about a third of the compute would have given almost the same model.'),
          P('6. Limitations', H2),
          P('1. All numbers are validation numbers, and the same set chose the checkpoint, so they are slightly optimistic. The 2,310-pair test set has not been scored.<br/>'
            '2. The session-2 log (steps 3,677&ndash;7,530), including the evaluation that selected the best checkpoint, has not been downloaded.<br/>'
            '3. The model weights are not in the local folder: out_v6/best/ holds only the configuration and tokenizer files.<br/>'
            '4. Validation words are dictionary and paradigm forms, not words in running sentences; accuracy on AGE sentences can be lower, especially for names and clitics.<br/>'
            '5. Run B (with Ge\'ez pretraining) was cancelled, so the effect of pretraining is unknown.'),
          P('7. Next steps', H2),
          P('1. Download best/ with its weights and the session-2 log (eval_log.jsonl).<br/>'
            '2. Score the held-out test set once with the selected checkpoint.<br/>'
            '3. Re-label the AGE corpus with this model and compare the new labels with the older analyser labels used so far in the translation experiments '
            '(coverage, agreement, root changes), then rebuild the memory label caches from it.<br/>'
            '4. Use its part-of-speech output as an additional signal for translation (for example, a part-of-speech slot bank or word-order features).<br/>'
            '5. Run B on TPU once the TPU runtime is stable, and compare A and B at matched steps.')]


def footer(c, doc):
    c.setFont('Helvetica', 7.5); c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Ge'ez ByT5 analyser, run A: training report, p. {doc.page}")


doc = SimpleDocTemplate('ANALYSER_RUN_A_REPORT.pdf', pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('pos', round(pos, 4), 'exact', round(exact, 4), 'root', round(root, 4), 'sessions h', [round(h, 2) for h in sess_hours])
print('gold', gold); print('uni', uni)
