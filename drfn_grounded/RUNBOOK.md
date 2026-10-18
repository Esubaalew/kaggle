# drfn_grounded — runbook

Separate pack and kernels from `drfn_analyser`. Nothing in `drfn_analyser`
was modified, so the analyser runs stay reproducible. No analyser weights are
attached here: this grid uses the fixed `geezrpa_label_cache_k256.json`
shipped inside `pack/data/`, so it does **not** drift when a new
`geez-morphology-byt5-training` run lands.

Two notebooks, which is exactly your two-GPU-session allowance.

## First time

    cd /Users/esubalew/is/researcher/kaggle/drfn_grounded
    kaggle datasets create -p pack -r zip
    kaggle datasets status esubaalew/drfn-grounded-pack

Repeat the status command until it says ready, then push both kernels:

    kaggle kernels push -p kernel           # the deciding grid, 4 runs
    kaggle kernels push -p kernel_entropy   # the entropy-sign check, 2 runs

## After changing anything in pack/

    kaggle datasets version -p pack -r zip -m "what changed"
    kaggle datasets status esubaalew/drfn-grounded-pack
    kaggle kernels push -p kernel

Wait for ready before the push, same as the analyser pack.

## Progress

    kaggle kernels status esubaalew/drfn-grounded-runs
    kaggle kernels status esubaalew/drfn-grounded-entropy

## Fetch results without weights

    cd /Users/esubalew/is/researcher/kaggle
    /Users/esubalew/.local/share/uv/tools/kaggle/bin/python fetch_small.py \
        run_name esubaalew/drfn-grounded-runs

---

## After each GPU notebook finishes: fetch results and probe on the Mac

Kaggle only exposes a notebook's output once it is COMPLETE, so this waits
for training. The two notebooks can be fetched separately; the entropy one
finishes first.

    cd /Users/esubalew/is/researcher/kaggle
    /Users/esubalew/.local/share/uv/tools/kaggle/bin/python fetch_best.py \
        run_grounded_entropy esubaalew/drfn-grounded-entropy
    /Users/esubalew/.local/share/uv/tools/kaggle/bin/python fetch_best.py \
        run_grounded_grid esubaalew/drfn-grounded-runs

`fetch_best.py` downloads everything `fetch_small.py` does, plus `best.pt`
for the runs that have a routing memory (`gmm*`, `cmm*`). It skips `last.pt`
and the `bpe_*` checkpoints, since those models have no memory to probe.
That is about 1.2 GB for five checkpoints of 244 MB each, rather than the
full output. Downloads resume after a stall, and rerunning the same command
skips files that are already complete.

The probe itself (`probe_runs.py`) then runs locally in a few minutes, and
the train logs and metadata come down in the same fetch, so nothing has to
be pasted from the Kaggle viewer.

**Fallback, if a download keeps failing:** the CPU-only Kaggle notebook does
the same probe in the cloud, with no download and no GPU quota.

    cd /Users/esubalew/is/researcher/kaggle/drfn_grounded
    kaggle kernels push -p kernel_probe
    kaggle kernels status esubaalew/drfn-grounded-probe

Push it only after both GPU notebooks say COMPLETE, otherwise it attaches an
unfinished output.

**Why it exists.** The `routing consistency (root)` line printed at the end
of each training run counts every *occurrence* of a word. A frequent word
appearing many times in similar sentences routes the same way each time,
which inflates the score without the model having learned anything about
roots, and the token-shuffling baseline does not correct for it. Section 4.6
defines the probe over *surface forms*; this notebook implements that: one
vote per surface form, then root consistency across forms, against a baseline
that permutes root labels across forms.

It also reports the score for dev forms **unseen in training** together with
their seen siblings, and, for grounded runs, how often an unseen form routes
to its own root's slot.

**Do not compare these numbers with the published 0.337 / 0.330.** That was
measured on the *unsupervised* CMM with a different probe implementation.
Compare runs with each other, using this probe for all of them.

## What each notebook answers

### `kernel` — the deciding grid (seed 11)

Four runs sharing one BPE tokenizer, parameter-matched to ~20.34M:

| run | labels | supervision goes to |
|---|---|---|
| `bpe_aux` | real | two linear heads on the encoder; output discarded |
| `bpe_placebo` | shuffled | same heads |
| `gmm` | real | the memory's own routing distributions |
| `gmm_placebo` | shuffled | same routing |

The summary cell prints the two gaps and their difference. Read
`(gmm − gmm_placebo)` against `(bpe_aux − bpe_placebo)`; the head-side value
in the three-seed history is **+0.47 chrF++**.

The placebo permutes labels *across words*, so under shuffling two surface
forms of one root land in different slots and the memory cannot share between
them. A memory gap clearly larger than the head gap is the result the
architecture predicts. Equal gaps mean morphological *content* adds nothing
beyond auxiliary-task regularization at this data size — a controlled
negative result, and also worth reporting.

If the session runs out of time on four runs, split it: set
`VARIANTS = "bpe_aux bpe_placebo"` in cell 1, push, then
`VARIANTS = "gmm gmm_placebo"` and push again. The shared tokenizer lives in
`runs/shared/tokenizer.json`, so keep the seed fixed at 11 across both halves.

### `kernel_entropy` — is the CMM result about the idea, the sign, or the code?

Three runs, seed 11, auxiliary heads on in all three so supervision is held
constant. Each changes exactly one thing from the published configuration.

| run | module | entropy term |
|---|---|---|
| `cmm_s11_diffuse` | `cmm.py` as shipped | maximized (as shipped) |
| `cmm_s11_sharpen` | `cmm.py` as shipped | minimized (section 3.6) |
| `cmmspec_s11_sharpen` | protocol section 3.4 | minimized |

**The sign.** `train.py` had `total = total - entropy_reg * ent`, which
*maximizes* routing entropy and pushes every token toward uniform assignment
over slots. Section 3.6 says the opposite — Eq. (12) adds `+alpha * L_ent`,
and the text reads "Minimizing `L_ent` sharpens each token's routing toward
few slots". Diffuse routing is named right there as failure mode (i).

**The module.** `cmm.py` is *not* the CMM of section 3.4. It uses one
`root_bank` as both key and value with no query projection —
`softmax(h Bᵀ) B` — where Eqs. (4)–(5) specify `Q = H W_Q` with separate
`M_K` and `M_V`, and Eq. (9) specifies a per-token *scalar* gate rather than
the d-dimensional one in the code. With tied keys and values the retrieved
vector is pulled toward whichever slot already lies closest to `h`, biasing
slots toward surface similarity instead of abstract roots. **The architecture
the proposal describes has never been trained.** `cmm_spec` is it.

Reading the result against the published 0.337 (permutation 0.330):

- sharpened run moves off chance → the negative result was measured under a
  term pushing routing toward chance;
- only `cmm_spec` moves → the shipped module was the problem;
- neither moves → the mechanism genuinely fails to bind at 13.8K pairs, and
  that is now a *clean* negative result instead of a confounded one.

All three outcomes are usable. The current situation — where the thesis
cannot distinguish them — is the only one that isn't.

`--entropy-sign` now reaches both the grounded and the legacy path. Verified:
flipping the sign changes the loss by exactly `2 * entropy_reg * H`, and
**omitting the flag reproduces the published behaviour bit-for-bit**, so your
existing `cmm` numbers remain valid.

---

## Why geezRPA labels and not the ByT5 analyser

The analyser is the better analyser — 88.3% root accuracy against 39.9% for
rules-only. It is still the wrong choice *for this grid*, for one reason:
as label sources the two are indistinguishable in translation quality
(11.41 against 11.40, three seeds each). So label source is not the variable
under test here, and switching it would change two things at once — where
the supervision goes *and* what the labels are — leaving the decisive
comparison confounded.

geezRPA also keeps this grid commensurable with the entire published
history: the +0.47 placebo gap, the 11.40 both-labels baseline, and the
0.337 routing consistency were all measured on the k256 cache.

**The follow-up is worth running, though, and for a specific reason.**
Grounding depends on two surface forms of a root landing in the *same slot*,
so the quality of the root *grouping* matters to it in a way it never
mattered to a head. geezRPA's groupings are visibly worse — it files
አምለክሙ under root ለከመ alongside ለክሙ and ወለክሙ, which is a suffix match, not
a root match — while the analyser correctly groups ወኮከበ with ኮከብ and ከዋክብት.
So the analyser should help `gmm` specifically, and *not* help `bpe_aux`.

That is a sharp, falsifiable prediction and a second piece of evidence for
the architectural claim. Run it as a second arm once the grid reports:
rebuild the label cache from the analyser readings, then rerun `gmm`,
`gmm_placebo`, `bpe_aux`, `bpe_placebo` against it. Do not fold it into the
first grid.

## Verified before shipping

All variants run end-to-end on CPU from this pack, with parameter matching
against the 20,343,170 target:

| variant | params | deviation | module |
|---|---|---|---|
| `cmm` | 20,343,001 | 0.0008% | `cmm.py`, tied K/V, vector gate |
| `cmm_spec` | 20,343,563 | 0.0019% | section 3.4: untied K/V, `W_Q`, scalar gate |
| `bpe_aux`, `bpe_placebo` | 20,342,041 | 0.0055% | no memory |
| `gmm`, `gmm_placebo` | 20,341,818 | 0.0066% | section 3.4 + grounding + word-level |

`cmm` and `cmm_spec` both keep the auxiliary heads; the `gmm` variants drop
them, because the routing CE replaces them.

At initialization the routing CE sits at exactly chance — `route_root`
5.5432 against ln(256) = 5.545, `route_pattern` 2.8345 against ln(17) =
2.833 — and gradients reach all four memory matrices.

One thing the smoke test does **not** show: `gmm` and `gmm_placebo` print
identical dev loss for the first ~100 steps. That is the 400-step linear
warmup holding the learning rate near 1e-5, not a wiring fault. They diverge
once the warmup releases.

## Before the six-seed grid

Detach punctuation from Ge'ez words in the corpus. 7.45% of training tokens
have it glued (እግዚአብሔር።, ውእቱ።, ክርስቶስ።) while ። also appears as its own token
6,871 times — the corpus uses both conventions. Fixing it shrinks the
vocabulary 5.6% (43,703 → 41,276 types) and lifts analyser root coverage from
89.4% to 91.1% of tokens. Modest, but it affects every run, so do it before
the grid that counts rather than after.
