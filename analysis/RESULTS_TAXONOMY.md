# Grounded Morphological Memory: results taxonomy

*Ge'ez→Amharic, AGE corpus. State as of 15 Oct 2026. All numbers are on the dev set (1,741 pairs). The test split has not been touched.*

---

## 0. Conventions

| item | value |
|---|---|
| Train / dev | 13,932 / 1,741 pairs (AGE; github.com/HenokB/AGE-Dataset) |
| Main metric | character chrF (sacrebleu, `word_order=0`), invisible characters removed |
| Decoding | beam 5 + no-repeat 4-gram (`beam5_nr4`) unless marked *greedy* |
| Seeds | 11, 22 (+33 for the wide arm) |
| Intervals | 95% paired bootstrap over dev **sentences** of the seed-mean score (2,000 resamples). These do **not** include seed-to-seed variance. |
| Seed noise | single-run SD ≈ 0.23 chrF (beam), ≈ 0.35 (greedy) |
| Code generations | **G0** = original code (12–13 Oct, superseded). **G1** = fix code (14–15 Oct, current). Do not compare dev loss across G0 and G1. |

---

## 1. Taxonomy of systems

### 1.1 Small Transformer (4+4 layers, d=256, ≈20.34M parameters, parameter-matched within 0.05%)

| system | what it adds to the BPE Transformer | uses root labels? | role |
|---|---|---|---|
| `plain_bpe` | nothing | no | floor |
| `bpe_aux` | side heads predicting root and pattern from the encoder | yes, as auxiliary loss only | **strongest label baseline** |
| `cmm_spec` | specification memory (thesis §3.4), slots learned without labels | no | ungrounded memory |
| `gmm` | grounded memory: slots bound to root/pattern classes; labels supervise routing, gate and values; keeps side heads | yes, for the memory itself | **proposed** |

### 1.2 NLLB-200-distilled-600M (source tag `tir_Ethi`, target `amh_Ethi`)

| system | description |
|---|---|
| zero-shot | no fine-tuning |
| `ft` | full fine-tune, no memory |
| `aux` | fine-tune + root/pattern side heads |
| `gmm_free` | same memory, same parameters, **no grounding** (routing learned freely) |
| `gmm` | grounded memory inserted at encoder layers (G0: 4, 8; G1: 5, 11) |

### 1.3 Label sources

| labels | origin | classes |
|---|---|---|
| `geezRPA` | rule-based analyser | 256 root slots |
| `analyser` | user's ByT5 analyser (geez-morphology 3dc784e, run byt5_3dc784e_337cdf0_books_s1), G1 version cleaned against geez-dataset 337cdf0 | 256 root slots + 17 patterns |
| `analyser_wide` | same cleaned labels, larger inventory | 1,024 root slots + 17 patterns |

In G1, function words, names, punctuation and out-of-inventory roots get label 0 ("rootless"). Only BOS/EOS/PAD are ignored.

---

## 2. Code generations and what changed

### 2.1 G0 problems (found 14 Oct)

1. Only 44–52% of tokens received a routing target. Out-of-inventory roots were −100 and rootless tokens were 0, and both were ignored, so slot 0 was never trained.
2. Entropy sharpening was applied to every position, including rootless ones.
3. The analyser labels gave roots to function words, names and punctuation (e.g. ውስተ, ከመ, እግዚአብሔር, እለ, ኀበ, ፤).

### 2.2 G1 fixes (all introduced together)

- cleaned labels (256 and 1,024 slots);
- rootless words get label 0, and slot 0 is masked from routing;
- supervised gate: BCE(λ, root ≥ 1);
- value-alignment cross-entropy;
- pre-norm residual h + λ·LN(c), with masks applied after LayerNorm;
- NLLB only: memory at layers 5 and 11, gate bias −2, small-gain init, separate gradient clipping.

**Consequence:** G1 differs from G0 in several ways at once, so no single fix can be credited with an effect until ablations are run.

---

## 3. Results by research question

### Q-a. Do the slots identify roots, including roots of unseen word forms?

| setting | metric | grounded `gmm` | ungrounded control | gen |
|---|---|---|---|---|
| small, 256 | root consistency (all types) | 0.905 / 0.902 | `cmm_spec` 0.239 / 0.232 (permutation 0.18–0.21) | G1 |
| small, 256 | unseen forms → own root slot | 0.796 / 0.778 | — | G1 |
| small, 1024 | root consistency | 0.819 / 0.844 / 0.829 (perm 0.296) | — | G1 |
| small, 1024 | unseen forms → own root slot | 0.649 / 0.679 / 0.654 | — | G1 |
| NLLB, 256 | unseen forms → own root slot | 0.878 / 0.876 | `gmm_free` 0.002 / 0.002 | G1 |
| NLLB, 1024 | unseen forms → own root slot | 0.795 / 0.803 | — | G1 |
| small (G0) | unseen forms → own root slot | 80–82% | ≤ 1.6% | G0 |

**Gate (opening on words with a root vs rootless words):**

| setting | `gmm` | ungrounded |
|---|---|---|
| small, 256 | 0.63 / 0.60 vs 0.08 | `cmm_spec` 0.13 vs 0.15 (no separation) |
| small, 1024 | 0.60–0.66 vs 0.08–0.09 | — |
| NLLB, 256 | 0.88 vs 0.07 | `gmm_free` s11: 0.36 vs 0.31, s22: 0.83 vs 0.95 (no separation) |
| NLLB, 1024 | 0.89–0.90 vs 0.10 | — |

**Answer: supported in every setting.** Grounded slots identify roots, generalise to unseen forms, and the gate learns which words have a root. Ungrounded memories do none of this.

### Q-b. Does the translator use the slots? (causal tests)

| test | small G0 | small G1, 256 | small G1, 1024 | NLLB G1 |
|---|---|---|---|---|
| root-token loss when the slot is corrupted (nats) | +0.18 to +0.32 | **+1.14 / +1.08** | +1.07 / +1.06 / +1.12 | — |
| other-token loss when the slot is corrupted (nats) | −0.005 to −0.01 | +0.005 / +0.001 | +0.002 to +0.003 | — |
| correct word lost, another root swapped in | 15–18% | **52% / 52%** | 52% / 56% / 56% | 0.8–3.2% |
| correct word lost, memory off for that word | 7–12% | **37% / 39%** | 35% / 40% / 41% | 0.8–2.8% |
| swapped root's word produced | 0–0.6% | 1.8% / 2.7% | 1.3% | 0% |
| memory-off loss rise, unseen root tokens vs unlinked (nats) | 1.4–19× ratio (median 3.7×) | 0.53 / 0.49 vs 0.10 / 0.11 | 0.41–0.54 vs 0.15–0.17 | — |

**Answer: yes in the small model, and far more strongly under G1** (dependence on the memory roughly 3–6× larger than under G0). **NLLB barely uses it:** with the memory off, the correct word survives 97–99% of the time. The pretrained model already handles these words.

Swapping in another root's slot breaks the word but rarely produces the new root's word (≤ 2.7%). The slot carries "which root" information that the decoder needs, but it is not a clean lexical address.

### Q-c. Does translation improve?

#### Small model, G1

| labels | seed | `gmm` | `bpe_aux` | `cmm_spec` | gap gmm − bpe_aux |
|---|---|---|---|---|---|
| 256 | 11 | 15.13 | 14.93 | 14.99 | +0.20 |
| 256 | 22 | 14.64 | 14.95 | 14.84 | −0.31 |
| 1024 | 11 | 15.35 | 15.37 | — | −0.02 |
| 1024 | 22 | 15.56 | 14.96 | — | +0.60 |
| 1024 | 33 | 15.25 | 14.59 | — | +0.67 |

| contrast (G1) | Δ chrF | 95% CI (sentences) |
|---|---|---|
| gmm − bpe_aux, 256, 2 seeds | −0.06 | [−0.30, +0.18] |
| cmm_spec − bpe_aux, 256, 2 seeds | −0.03 | [−0.25, +0.19] |
| **gmm − bpe_aux, 1024, 3 seeds** | **+0.41** | **[+0.23, +0.60]** |
| gmm − bpe_aux, 1024, seed 33 only (same session) | +0.67 | [+0.34, +0.97] |
| gmm: 1024 − 256 (seeds 11, 22) | +0.57 | [+0.34, +0.81] |
| bpe_aux: 1024 − 256 (seeds 11, 22) | +0.23 | [+0.01, +0.43] |

Across seeds, the 1024-slot gaps are −0.02, +0.60 and +0.67 (mean 0.41, SD 0.38, t = 1.89, df = 2). That is **not yet significant across seeds**.

#### Small model, G0 (superseded, kept for the history chapter)

| finding | value |
|---|---|
| `plain_bpe`, 3 seeds | 13.44 / 13.39 / 13.19 |
| gain from labels over `plain_bpe` | bpe_aux +1.68 (geezRPA) / +1.86 (analyser); cmm_spec +1.93; gmm +1.27 to +1.49 |
| gmm − bpe_aux | −0.42 [−0.65, −0.18] geezRPA; −0.37 [−0.59, −0.13] analyser |
| cmm_spec − bpe_aux | +0.07 (n.s.) |

#### NLLB-600M

| system | chrF s11 / s22 | BLEU s11 / s22 | dev loss | gen |
|---|---|---|---|---|
| zero-shot | 13.52 | — | — | — |
| `ft` | 32.45 / 32.20 | ≈ 12.1 | — | G0 (no memory, unaffected by fixes) |
| `aux` (default weight) | 29.8 (−2.57 vs ft) | — | — | G0 |
| `aux` (weight 0.1) | 31.96 (−0.49 vs ft) | — | — | G0 |
| `gmm_free` | 32.52 / 32.44 | 12.51 / 12.10 | 1.8985 / 1.8971 | G1 |
| `gmm`, 256 | 32.36 / 32.23 | 12.08 / 12.03 | 1.8999 / 1.9013 | G1 |
| `gmm`, 1024 | 32.55 / 32.46 | 12.36 / 12.14 | 1.8921 / 1.8905 | G1 |

| contrast (NLLB, 2 seeds) | Δ chrF | 95% CI |
|---|---|---|
| gmm(256) − gmm_free | −0.19 | [−0.41, +0.05] |
| gmm(1024) − gmm_free | +0.02 | [−0.19, +0.25] |
| gmm(1024) − gmm(256) | +0.21 | [+0.04, +0.37] |
| any memory system − ft | — | all intervals include 0 |
| G0: gmm − gmm_free | +0.09 | n.s. |
| G0: unseen-word exact match, gmm − gmm_free | +0.5 | [−2.8, +3.8] |

**Answer:**
- **Small model:** yes with 1,024 root slots (+0.41, two of three seeds), no with 256 (tie).
- **NLLB:** no. The grounded memory matches but does not beat an identical ungrounded memory or plain fine-tuning.

The published AGE baseline is BLEU 12.29 on the authors' own split, so it is not directly comparable. Our NLLB fine-tunes reach BLEU 12.0–12.5 on our dev split.

---

## 4. Supporting findings (methodology)

| finding | value | consequence |
|---|---|---|
| greedy outputs repeat a word 3-gram | 49–58% of outputs (references 1.8%) | all reported scores use beam5_nr4 |
| beam5_nr4 gain over greedy | +1.4 to +2.1 chrF for every system | does not change rankings |
| seed noise | SD 0.23 (beam) / 0.35 (greedy) | single-seed gaps below ≈ 0.8 chrF are not interpretable |
| seed-11 G0 "gmm beats heads" | reversed at seed 22 | withdrawn |

---

## 5. Status of the thesis claims

| claim | status | strongest evidence |
|---|---|---|
| (a) grounded slots identify roots, incl. unseen forms | **supported** | small 0.90 consistency vs 0.24 ungrounded; NLLB 88% vs 0.2% |
| (a') the gate learns which words have roots | **supported (G1)** | 0.60–0.90 vs 0.07–0.10; ungrounded gates do not separate |
| (b) the small translator uses the slots | **supported, strongly (G1)** | corrupting the slot: +1.1 nats on root tokens, ≈ 0 elsewhere; 37–41% words lost with memory off |
| (b') NLLB uses the slots | **not supported** | 97–99% of words survive memory-off |
| (c) grounding improves small-model translation | **partially supported** | +0.41 [+0.23, +0.60] at 1024 slots (3 seeds, not significant across seeds); tie at 256 |
| (c') grounding improves NLLB translation | **not supported** | gmm ≈ gmm_free ≈ ft |
| labels help a small translator | **supported (G0)** | +1.3 to +1.9 chrF over plain_bpe |
| finer root inventories help memory more than side heads | **suggestive** | +0.57 vs +0.23 (2 seeds) |

---

## 6. Open items, in priority order

1. **More seeds of the 1024-slot arm** (seeds 44, 55: gmm + bpe_aux; one T4×2 session ≈ 2.5 h) to test the +0.41 gap across seeds.
2. **Ablations of the G1 fixes** (gate BCE off; value CE off; labels cleaned vs not) to attribute the larger dependence on the memory.
3. **Unseen-word exact-match test** on small G1 runs (`score_unseen.py --pairs`).
4. `plain_bpe` under G1 data, so the label gain can be re-stated with the current code.
5. isiXhosa→English with public data and unsupervised morphological labels (language-generality claim).
6. One final test-set evaluation at the very end.
7. Fix the README count inconsistencies in geez-dataset.

---

## 7. Where everything is

Host root: `/Users/esubalew/is/researcher/kaggle/`

| folder | contents | gen |
|---|---|---|
| `run_eval2/`, `run_seeds/`, `run_overnight/` | small-model G0 runs and beam re-evaluation | G0 |
| `run_nllb/`, `run_nllb2/` | NLLB G0 (ft, gmm, aux, gmm_free, aux_w01) | G0 |
| `run_fix_clean/` | small G1, 256 slots: bpe_aux, cmm_spec, gmm × s11, s22 | G1 |
| `run_fix_wide_colab/` | small G1, 1024 slots, Colab: gmm s11, s22; bpe_aux s11 (+ the original Drive zip with weights) | G1 |
| `run_fix_wide_s22/` | small G1, 1024 slots, Kaggle: bpe_aux s22, s33; gmm s33 | G1 |
| `run_nllb_fix/` | NLLB G1 v1: **only gmm_free s11, s22 are valid** (gmm runs crashed) | G1 |
| `run_nllb_fix2/` | NLLB G1 v2: gmm and gmm_wide × s11, s22 | G1 |
| `drfn_fix/`, `drfn_nllb_fix/`, `drfn_fix_s22/` | G1 code packs and notebooks | G1 |
| `analysis/` | scoring scripts and JSON outputs (this file, `fix_results_all.csv`, `fix_clean_compare.*`, `wide_compare.*`, `nllb_fix_compare.*`, older `seed_compare*`, `labels_effect*`, `repetition*`) | both |

Kaggle notebooks:
- `esubalewchekol/`: drfn-fix-clean, drfn-nllb-fix (v1 crashed; v2 valid);
- `esubaalew/`: drfn-fix-wide-s22.

Colab: `MyDrive/drfn/pack/runs/`.
