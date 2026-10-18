"""Plain-language overview of the thesis so far: terms, workflow, architecture, taxonomy, results, next plan."""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Polygon
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

KEFA = '/System/Library/Fonts/Supplemental/KefaIII.ttf'
ETH = 'Helvetica'
if os.path.exists(KEFA):
    pdfmetrics.registerFont(TTFont('Kefa', KEFA)); ETH = 'Kefa'

ss = getSampleStyleSheet()
H1 = ParagraphStyle('H1', parent=ss['Heading1'], fontSize=16, spaceAfter=6, textColor=colors.HexColor('#1f3a5f'))
H2 = ParagraphStyle('H2', parent=ss['Heading2'], fontSize=12.5, spaceBefore=8, spaceAfter=4, textColor=colors.HexColor('#1f3a5f'))
B = ParagraphStyle('B', parent=ss['BodyText'], fontSize=9.6, leading=13.2, spaceAfter=4)
S = ParagraphStyle('S', parent=B, fontSize=8.4, leading=11)
BOX = ParagraphStyle('BOX', parent=B, backColor=colors.HexColor('#eef3f8'), borderPadding=6,
                     borderColor=colors.HexColor('#9db4cc'), borderWidth=0.6, spaceBefore=4, spaceAfter=8)
E = lambda t: f'<font name="{ETH}">{t}</font>'
P = lambda t, st=B: Paragraph(t, st)
BLUE, GREY, GREEN, ORANGE = '#dbe7f3', '#eeeeee', '#dff0d8', '#fbe5c8'


def table(rows, widths, head=True, fs=8.3):
    data = [[Paragraph(str(c), ParagraphStyle('t', parent=S, fontSize=fs, leading=fs + 2.4)) for c in r] for r in rows]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#b0b0b0')), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
          ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]
    if head:
        st.append(('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#d6e2ef')))
    t.setStyle(TableStyle(st)); return t


def box(d, x, y, w, h, text, fill=BLUE, fs=7.6):
    d.add(Rect(x, y, w, h, rx=4, ry=4, fillColor=colors.HexColor(fill), strokeColor=colors.HexColor('#5a7896'), strokeWidth=0.7))
    lines = text.split('\n'); y0 = y + h / 2 + (len(lines) - 1) * (fs + 1) / 2 - fs / 3
    for i, ln in enumerate(lines):
        d.add(String(x + w / 2, y0 - i * (fs + 1), ln, fontName='Helvetica', fontSize=fs, textAnchor='middle'))


def arrow(d, x1, y1, x2, y2):
    d.add(Line(x1, y1, x2, y2, strokeColor=colors.HexColor('#333333'), strokeWidth=0.9))
    import math
    a = math.atan2(y2 - y1, x2 - x1); s = 4.5
    d.add(Polygon([x2, y2, x2 - s * math.cos(a - 0.4), y2 - s * math.sin(a - 0.4),
                   x2 - s * math.cos(a + 0.4), y2 - s * math.sin(a + 0.4)], fillColor=colors.HexColor('#333333'), strokeWidth=0))


def workflow():
    d = Drawing(17 * cm, 6.6 * cm); w, h = 2.85 * cm, 1.35 * cm; y1, y2 = 4.6 * cm, 1.2 * cm
    top = ['1. Parallel data\nAGE Ge\'ez-Amharic\n13,932 train / 1,741 dev', '2. Morphological\nanalyser (ByT5)\nroot + pattern per word',
           '3. Clean labels\ncheck with dictionary;\nfunction words -> 0', '4. Label inventory\ntop 255 or 1,023 roots\n+ 16 patterns']
    bot = ['5. Train systems\nsame data, same size,\nseeds 11 / 22 / 33', '6. Translate dev set\nbeam 5, no repeated\n4-grams', '7. Score\nchrF / BLEU +\npaired bootstrap',
           '8. Inspect memory\nprobes (detection)\n+ causal tests']
    xs = [0.2 * cm + i * 4.2 * cm for i in range(4)]
    for i, t in enumerate(top):
        box(d, xs[i], y1, w, h, t, BLUE)
        if i: arrow(d, xs[i - 1] + w, y1 + h / 2, xs[i], y1 + h / 2)
    for i, t in enumerate(bot):
        box(d, xs[i], y2, w, h, t, GREEN if i < 2 else ORANGE)
        if i: arrow(d, xs[i - 1] + w, y2 + h / 2, xs[i], y2 + h / 2)
    arrow(d, xs[3] + w / 2, y1, xs[0] + w / 2, y2 + h)
    d.add(String(4.2 * cm, 3.2 * cm, 'labels go into training', fontName='Helvetica-Oblique', fontSize=7.2, textAnchor='middle'))
    return d


def architecture():
    d = Drawing(17 * cm, 9.2 * cm); cx = 1.0 * cm
    box(d, cx, 0.2 * cm, 4.2 * cm, 0.9 * cm, 'Ge\'ez sentence -> subword tokens', GREY)
    box(d, cx, 1.6 * cm, 4.2 * cm, 1.0 * cm, 'Encoder layers (lower)', BLUE)
    box(d, cx, 3.3 * cm, 4.2 * cm, 2.6 * cm, '', ORANGE)
    d.add(String(cx + 2.1 * cm, 5.55 * cm, 'MEMORY BLOCK', fontName='Helvetica-Bold', fontSize=8, textAnchor='middle'))
    box(d, cx + 0.2 * cm, 4.35 * cm, 1.85 * cm, 0.9 * cm, 'Router: which\nroot slot?', '#ffffff', 7)
    box(d, cx + 2.15 * cm, 4.35 * cm, 1.85 * cm, 0.9 * cm, 'Gate: open\nor closed?', '#ffffff', 7)
    box(d, cx + 0.2 * cm, 3.45 * cm, 3.8 * cm, 0.7 * cm, 'out = h + gate x LN(slot vector)', '#ffffff', 6.4)
    box(d, cx, 6.6 * cm, 4.2 * cm, 1.0 * cm, 'Encoder layers (upper)', BLUE)
    box(d, cx, 8.1 * cm, 4.2 * cm, 0.9 * cm, 'Decoder -> Amharic sentence', GREY)
    for y0, y1 in [(1.1, 1.6), (2.6, 3.3), (5.9, 6.6), (7.6, 8.1)]:
        arrow(d, cx + 2.1 * cm, y0 * cm, cx + 2.1 * cm, y1 * cm)
    # slot bank
    bx = 7.0 * cm
    box(d, bx, 3.4 * cm, 4.4 * cm, 2.5 * cm, '', GREY)
    d.add(String(bx + 2.2 * cm, 5.6 * cm, 'Slot bank (learned vectors)', fontName='Helvetica-Bold', fontSize=7.5, textAnchor='middle'))
    for i, lab in enumerate(['0: none', '1: s-g-d', '2: f-t-r', '...', '1023']):
        box(d, bx + 0.15 * cm + i * 0.85 * cm, 4.3 * cm, 0.78 * cm, 0.8 * cm, lab, '#ffffff', 5.6)
    d.add(String(bx + 2.2 * cm, 3.7 * cm, '+ 17 pattern slots', fontName='Helvetica', fontSize=7, textAnchor='middle'))
    arrow(d, bx, 4.7 * cm, cx + 4.05 * cm, 4.7 * cm)
    # labels
    lx = 12.3 * cm
    box(d, lx, 3.4 * cm, 4.4 * cm, 2.5 * cm, 'GROUNDING (training only)\nanalyser label of each word:\nroot id -> teaches the router\nhas root? -> teaches the gate\nroot id -> teaches slot content', GREEN, 7)
    arrow(d, lx, 4.65 * cm, bx + 4.4 * cm, 4.65 * cm)
    d.add(String(14.5 * cm, 2.8 * cm, 'Ungrounded memory = same block, no green box:', fontName='Helvetica-Oblique', fontSize=7, textAnchor='middle'))
    d.add(String(14.5 * cm, 2.45 * cm, 'the router decides by itself what slots mean.', fontName='Helvetica-Oblique', fontSize=7, textAnchor='middle'))
    return d


def taxonomy_tree():
    d = Drawing(17 * cm, 6.3 * cm)
    box(d, 5.5 * cm, 5.2 * cm, 6 * cm, 0.9 * cm, 'How are morphological labels used?', '#d6e2ef', 8)
    xs = [0.1, 4.35, 8.6, 12.85]; w = 4.0 * cm
    names = [('A. Not at all', 'plain_bpe  |  ft (NLLB)', GREY),
             ('B. Auxiliary task', 'bpe_aux  |  aux (NLLB)\nlabels = extra training\ntarget; nothing stored', BLUE),
             ('C. Memory, ungrounded', 'cmm_spec  |  gmm_free\nslots exist, but labels\nare NOT used', ORANGE),
             ('D. Memory, grounded', 'gmm (proposed)\nlabels decide slot,\ngate and slot content', GREEN)]
    for x, (a, b, f) in zip(xs, names):
        box(d, x * cm, 3.15 * cm, w, 0.75 * cm, a, f, 8)
        box(d, x * cm, 0.6 * cm, w, 2.2 * cm, b, '#ffffff', 7.3)
        arrow(d, 8.5 * cm, 5.2 * cm, x * cm + w / 2, 3.9 * cm)
        arrow(d, x * cm + w / 2, 3.15 * cm, x * cm + w / 2, 2.8 * cm)
    d.add(String(8.5 * cm, 0.15 * cm, 'Each family exists in the small model (20.3M) and, where marked, in NLLB-600M. Grounded memory comes with 256 or 1,024 root slots.',
                 fontName='Helvetica-Oblique', fontSize=6.9, textAnchor='middle'))
    return d


story = []
story += [P('Grounded Morphological Memory for Ge\'ez&rarr;Amharic Translation', H1),
          P('<b>Plain-language overview of the work so far</b> &middot; Esubalew Chekol, MSc in AI, Addis Ababa University &middot; 15 October 2026', B),
          P('<b>Purpose of this document.</b> A reader with no knowledge of the project history should be able to answer: '
            'what problem is studied, what each system is, how the systems differ, what was run, what was found, and what comes next. '
            'All numbers are on the development set; the test set has not been used.', BOX)]

story += [P('1. The problem in one paragraph', H2),
          P('Ge\'ez is a morphologically rich Semitic language. One <b>root</b> (usually three consonants) produces many word forms. '
            f'For example the root s-g-d ("bow, worship") gives {E("ሰገደ")} (he bowed), {E("ሰገዱ")} (they bowed) and {E("ትስግዱ")} (you bow). '
            'A translation system trained on 14,000 sentence pairs sees most of these forms rarely or never. '
            'The thesis asks: <b>if we give the translator an explicit memory that is organised by roots, does it recognise roots of unseen word forms, '
            'does it actually use that memory, and does translation get better?</b>'),
          P('<b>Short answer so far.</b> Recognition: yes, clearly. Use: yes in the small model, hardly in the large pretrained NLLB model. '
            'Translation: a gain of about +0.4 chrF in the small model with 1,024 root slots (not yet significant across seeds); no gain in NLLB.', BOX)]

story += [P('2. Terms used in this document', H2),
          table([['Term', 'Meaning in plain words'],
                 ['Root', 'The consonant skeleton shared by related words (s-g-d). Carries the core meaning.'],
                 ['Pattern', 'The template of vowels/affixes that turns a root into a specific form (tense, person, noun type). 16 pattern classes are used.'],
                 ['Morphological analyser', 'A separate model (the author\'s ByT5 analyser) that reads a word and predicts its root and pattern.'],
                 ['Label', 'The analyser\'s answer for a word, e.g. "root = s-g-d, pattern = perfective 3rd plural". Labels are used only during training.'],
                 ['Rootless word', 'Function words (prepositions, particles, pronouns), names and punctuation. They get label 0 = "no root".'],
                 ['Label inventory', 'Only the most frequent roots get their own label: top 255 (256-slot setting) or top 1,023 (1,024-slot setting). Rarer roots get 0.'],
                 ['Auxiliary task ("aux")', 'An extra small output layer trained to predict the label from the encoder. It only adds a training signal; nothing is stored or looked up during translation.'],
                 ['Memory', 'An extra block inside the encoder holding a bank of learned vectors (<b>slots</b>). For each word, the block picks a slot and adds its vector to the word\'s representation.'],
                 ['Router', 'The part of the memory that decides <i>which</i> slot a word uses.'],
                 ['Gate', 'A number between 0 and 1 that decides <i>how much</i> of the slot is added. Closed (near 0) means the memory is ignored for that word.'],
                 ['Ungrounded memory', 'Memory whose slots are learned freely; nobody tells it what a slot should mean.'],
                 ['Grounded memory', 'Memory where slot k is <b>tied to root k</b>: labels train the router (go to your root\'s slot), the gate (open only if the word has a root) and the slot content.'],
                 ['Detection (own-slot accuracy)', 'For words never seen in training: the share that the router sends to the slot of their true root. 0% = no root knowledge.'],
                 ['chrF / BLEU', 'Automatic translation-quality scores (higher is better). chrF counts matching characters, which suits rich morphology; it is the main score.'],
                 ['Seed', 'The random starting point of training. Re-running with another seed shows how much a result is due to luck (about &plusmn;0.23 chrF here).'],
                 ['Beam 5, no repeated 4-grams', 'The decoding method: keep the 5 best partial translations and forbid repeating any 4-word sequence. Used for all reported scores.']],
                [4.0 * cm, 13.0 * cm])]

story += [PageBreak(), P('3. The four families of systems (taxonomy)', H2),
          P('Every system is the <b>same translator</b> trained on the <b>same data</b>, with the same number of parameters (within 0.05%). '
            'They differ only in <b>how the morphological labels are used</b>:'),
          taxonomy_tree(),
          P('<b>Why both B and C are needed as comparisons.</b> If grounded memory (D) beats the auxiliary task (B), the gain is not just "labels help". '
            'If D beats ungrounded memory (C), the gain is not just "an extra memory block helps". Only a win over both shows that <i>tying slots to roots</i> matters.', BOX),
          table([['Code name', 'Family', 'Model', 'What it is'],
                 ['plain_bpe', 'A', 'small', 'Transformer on subword tokens, no labels'],
                 ['bpe_aux', 'B', 'small', 'plain_bpe + auxiliary heads predicting root and pattern (strongest label baseline)'],
                 ['cmm_spec', 'C', 'small', 'memory module from the thesis specification, learned without labels'],
                 ['gmm', 'D', 'small and NLLB', 'grounded morphological memory (the proposal); the small version also keeps the auxiliary heads'],
                 ['ft', 'A', 'NLLB', 'NLLB-200-distilled-600M fine-tuned on AGE, no memory'],
                 ['aux', 'B', 'NLLB', 'ft + auxiliary heads'],
                 ['gmm_free', 'C', 'NLLB', 'exactly the gmm block and size, but no grounding'],
                 ['*_wide', '-', 'both', 'the same system with 1,024 root slots instead of 256']],
                [2.4 * cm, 1.4 * cm, 2.4 * cm, 10.8 * cm]),
          P('3.1 How a word gets its label ("is this word grounded?")', H2),
          P('1) The analyser predicts a root and pattern for every training word. '
            '2) The prediction is checked against the author\'s Ge\'ez dataset (geez-dataset): dictionary forms take the curated root; a predicted root is kept only if its consonants actually occur in the word in order. '
            '3) Function words, names, punctuation and non-Ge\'ez tokens are forced to label 0. '
            f'(Earlier labels wrongly gave roots to words like {E("ውስተ")} "in", {E("ከመ")} "like", {E("እግዚአብሔር")} "God" and even the punctuation mark {E("፤")}.) '
            '4) The 255 or 1,023 most frequent roots get their own slot; all others share label 0. '
            'So a word is "grounded" when it has a verified root inside the inventory; for such words the gate should open, for all others it should stay closed.')]

story += [PageBreak(), P('4. Workflow', H2), workflow(),
          P('Steps 1&ndash;4 build the labels once. Step 5 trains every system on identical data. Steps 6&ndash;7 measure translation; step 8 opens the memory to check what it learned and whether the translator depends on it.', S),
          P('5. Architecture of the grounded memory', H2), architecture(),
          P('The memory block sits between encoder layers. For each token, the <b>router</b> chooses a root slot (and a pattern slot), the <b>gate</b> decides how much to add, '
            'and the result is added back to the token representation. Grounding (green) is used only while training: at translation time the model has no access to labels '
            'and must recognise the root by itself, which is why detection on <i>unseen</i> words is the key test. In NLLB the block is inserted at encoder layers 5 and 11 and starts switched off (zero/low initialisation) so it cannot harm the pretrained model at the start.')]

story += [PageBreak(), P('6. What was run, in order', H2),
          table([['Stage', 'What', 'Outcome'],
                 ['G0 runs (12&ndash;13 Oct)', 'All systems, small model (seeds 11, 22, 3 seeds of plain_bpe) and NLLB (ft, aux, gmm, gmm_free)',
                  'Labels help (+1.3 to +1.9 chrF over plain_bpe). gmm detected roots well but translated 0.4 chrF <i>worse</i> than bpe_aux.'],
                 ['Evaluation audit', 'Seed repeats; repetition analysis', 'Greedy outputs looped in 49&ndash;58% of sentences, so all scores were redone with beam 5 + no repeated 4-grams. Single-seed differences below about 0.8 chrF are noise.'],
                 ['Bug review (14 Oct)', 'Inspection of the G0 memory code', 'Only 44&ndash;52% of tokens got a training signal; slot 0 was never trained; sharpening applied to all words; labels gave roots to function words.'],
                 ['G1 fixes', 'Clean labels, "no root" = 0, supervised gate, slot-content loss, normalisation change', 'All introduced together (individual effects not yet separated).'],
                 ['G1 runs (14&ndash;15 Oct)', 'Small: bpe_aux, cmm_spec, gmm at 256 slots (seeds 11, 22); bpe_aux, gmm at 1,024 slots (seeds 11, 22, 33). NLLB: gmm_free, gmm, gmm_wide (seeds 11, 22)',
                  'Results in section 7. G0 numbers are kept only as history.']],
                [3.2 * cm, 6.4 * cm, 7.4 * cm])]

story += [P('7. Results (current code, G1)', H2),
          P('<b>7.1 Does the memory detect roots?</b> (share of unseen words routed to their own root slot; gate = average opening)'),
          table([['Setting', 'Grounded gmm: detection', 'Gate on rooted / rootless words', 'Ungrounded control'],
                 ['Small, 256 slots', '78&ndash;80% (all-word consistency 0.90)', '0.60&ndash;0.63 / 0.08', 'cmm_spec: consistency 0.24 (chance 0.18&ndash;0.21); gate 0.13 / 0.15'],
                 ['Small, 1,024 slots', '65&ndash;68% (consistency 0.82&ndash;0.84)', '0.60&ndash;0.66 / 0.08&ndash;0.09', '-'],
                 ['NLLB, 256 slots', '88%', '0.87 / 0.07', 'gmm_free: 0.2%; gate does not separate'],
                 ['NLLB, 1,024 slots', '80%', '0.89&ndash;0.90 / 0.10', '-']],
                [3.0 * cm, 4.6 * cm, 4.0 * cm, 5.4 * cm]),
          Spacer(1, 5),
          P('<b>7.2 Does the translator use the memory?</b> (tests on trained models, no retraining)'),
          table([['Test', 'Small gmm', 'NLLB gmm'],
                 ['Scramble the slot of root words: extra loss on root tokens / other tokens', '+1.06 to +1.14 / about 0 (nats)', '-'],
                 ['Switch memory off for the word: correct word lost', '35&ndash;41%', '1&ndash;3%'],
                 ['Put another root\'s slot in: correct word lost', '52&ndash;56%', '1&ndash;3%']],
                [8.2 * cm, 4.4 * cm, 4.4 * cm]),
          P('The small translator clearly depends on the memory; NLLB already knows these words and largely ignores it.', S),
          Spacer(1, 4),
          P('<b>7.3 Does translation improve?</b> (chrF; mean over seeds; 95% interval from resampling dev sentences)'),
          table([['Setting', 'gmm (D)', 'bpe_aux (B)', 'cmm_spec / gmm_free (C)', 'D minus B'],
                 ['Small, 256 slots (2 seeds)', '14.88', '14.94', '14.91', '&minus;0.06 [&minus;0.30, +0.18]: tie'],
                 ['Small, 1,024 slots (3 seeds)', '15.39', '14.97', '-', '<b>+0.41 [+0.23, +0.60]</b>; per seed &minus;0.02, +0.60, +0.67'],
                 ['NLLB, 256 slots (2 seeds)', '32.29', 'aux &minus;0.5 to &minus;2.6 vs ft (G0)', 'gmm_free 32.48', 'D minus C: &minus;0.19 [&minus;0.41, +0.05]'],
                 ['NLLB, 1,024 slots (2 seeds)', '32.50', '-', 'gmm_free 32.48', 'D minus C: +0.02 [&minus;0.19, +0.25]']],
                [3.6 * cm, 1.5 * cm, 3.4 * cm, 3.5 * cm, 5.0 * cm]),
          P('Plain fine-tuned NLLB (ft) scores 32.33 on average; no memory system differs from it significantly. Zero-shot NLLB scores 13.5.', S)]

story += [PageBreak(), P('8. Status of each claim', H2),
          table([['Claim', 'Status', 'Main evidence'],
                 ['Grounded slots recognise roots, also of unseen words', 'Supported', '65&ndash;88% detection vs about 0% ungrounded'],
                 ['The gate learns which words have roots', 'Supported', '0.60&ndash;0.90 vs 0.07&ndash;0.10'],
                 ['The small translator uses the memory', 'Supported', 'removing/scrambling slots breaks 35&ndash;56% of root words'],
                 ['NLLB uses the memory', 'Not supported', '97&ndash;99% of words survive without it'],
                 ['Grounding improves small-model translation', 'Partly supported', '+0.41 chrF at 1,024 slots; 2 of 3 seeds; tie at 256'],
                 ['Grounding improves NLLB translation', 'Not supported', 'gmm &asymp; gmm_free &asymp; ft']],
                [6.4 * cm, 2.8 * cm, 7.8 * cm])]

story += [P('9. Next plan (advisor meeting, 15 Oct)', H2),
          P('<b>9.1 Does better detection give better translation? (detection ablation)</b>'),
          P('The existing runs already have very different detection levels, but they also differ in model, label inventory and code, so they cannot answer this question:'),
          table([['Existing run', 'Detection', 'chrF'],
                 ['small cmm_spec (ungrounded)', '~0%', '14.91'],
                 ['small gmm, 1,024 slots', '65&ndash;68%', '15.39'],
                 ['small gmm, 256 slots', '78&ndash;80%', '14.88'],
                 ['NLLB gmm_free (ungrounded)', '0.2%', '32.48'],
                 ['NLLB gmm, 1,024 / 256 slots', '80% / 88%', '32.50 / 32.29']],
                [7.0 * cm, 3.0 * cm, 3.0 * cm]),
          P('The planned experiment holds everything fixed and changes <b>only</b> detection, at the levels the advisor named (0%, about 40%, 80&ndash;82%, about 95%, plus 100%):'),
          P('&bull; <b>Part A: at translation time, no retraining.</b> Take a trained small gmm and, for a chosen share of words, send them to a random wrong slot (or to the correct slot, for 100%). '
            'Translate the dev set at each detection level and plot chrF against detection. This directly tests whether the translator benefits from correct detection.<br/>'
            '&bull; <b>Part B: during training.</b> Train gmm with a controlled share of root labels replaced by random roots (0%, 20%, 60%, 100% noise), which yields models with high to zero detection. '
            'This tests whether a model <i>trained</i> with better detection translates better. Two seeds per level; about 75 GPU-minutes per run.<br/>'
            '&bull; Repeat Part A on NLLB to see whether forcing correct detection helps a model that currently ignores the memory.'),
          P('<b>9.2 isiXhosa&rarr;English (language generality)</b>'),
          P('isiXhosa is agglutinative rather than root-and-pattern: words are built from a <b>stem</b> plus noun-class prefixes and verbal extensions. '
            'The same four families (A&ndash;D) will be run with slots tied to stems (instead of roots) and to affix classes (instead of patterns), using public parallel data and '
            'automatically derived segmentation labels. Success criterion: the same three questions (detection, use, translation) answered on a second, typologically different language.'),
          P('<b>9.3 Remaining items</b>'),
          P('&bull; Two more seeds (44, 55) of the 1,024-slot small arm to test the +0.41 gain across seeds.<br/>'
            '&bull; Ablate the G1 fixes one at a time (gate loss, slot-content loss, label cleaning) to see which caused the stronger memory use.<br/>'
            '&bull; Exact-match accuracy on unseen words for the small G1 models; plain_bpe re-run on the current code.<br/>'
            '&bull; A single final evaluation on the AGE test set at the very end.'),
          Spacer(1, 6),
          P('Full numbers: RESULTS_TAXONOMY.md and fix_results_all.csv (kaggle/analysis/). Each result comes from the run folders listed there.', S)]


def footer(c, doc):
    c.setFont('Helvetica', 7.5); c.setFillColor(colors.grey)
    c.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f'Grounded Morphological Memory: overview, p. {doc.page}')


doc = SimpleDocTemplate('THESIS_OVERVIEW.pdf', pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
doc.build(story, onFirstPage=footer, onLaterPages=footer)
print('ok')
