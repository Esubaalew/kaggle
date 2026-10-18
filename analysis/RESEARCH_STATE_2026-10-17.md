# Research state — 17 October 2026

## Live Kaggle state

Verified through the Kaggle API using the matching locally configured account credentials (no credentials copied into this report).

- `esubechekol/geez-byt5-gpu-v2-a`: RUNNING. Left untouched.
- `esubechekol/geez-byt5-gpu-v2-b`: CANCEL_ACKNOWLEDGED. Already cancelled when inspected; no new cancellation was needed.
- `esubalewchekol/geez-byt5-tpu-v2-a`: CANCEL_ACKNOWLEDGED.
- TPU B under that account: API returned 404; do not infer a completed run.

The A log endpoint returned no log text during two bounded samples. Its live status is verified, but its current optimizer step is not. The saved A session finished cleanly at step 3,676/10,980, not at the end of training. B's later persisted log shows resume from 3,685, evaluation at 4,000, and a saved checkpoint at 4,172 before cancellation. A latest-session checkpoint was not downloaded or verified. Existing `log_a/best` and `log_b/best` contain configs/tokenizers, not model weight files.

## Two distinct research tracks

1. Morphology analyser: ByT5 predicts roots and morphological features from a Ge’ez word. A = direct fine-tuning; B = Ge’ez span-corruption pretraining followed by the same fine-tuning. These GPU A/B runs are not the thesis translation comparisons.
2. Translation thesis: Ge’ez→Amharic on AGE, comparing auxiliary morphology heads, ungrounded memory, and grounded morphological memory (GMM). Current October evidence is in `RESULTS_TAXONOMY.md`, comparison JSONs, and saved run folders beside this report.

## Morphology GPU evidence

Both A/B use packaged model commit a1ad2b6 and dataset commit 04f45a7, two T4 GPUs, fp32, effective batch 512, and a 10,980-step fine-tuning target. The saved load gate passed: 562,403 training examples, 3,472 dev rows, exact-root/pair leakage audit CLEAN.

At the matched step 3,000:

| Run | Full-analysis exact | Root accuracy | Selection score |
|---|---:|---:|---:|
| A | 69.01% | 90.78% | 79.90% |
| B | 67.28% | 90.78% | 79.03% |

B later reached step 4,000: exact 68.61%, root 90.90%, selection score 79.75%. This is not a matched-step comparison with A. These are intermediate dev results, not final held-out test results. No final claim that pretraining helps or hurts is established.

## TPU failure and code

**Later evidence retrieved in this session:** the latest TPU attempt did initialize on eight workers, crashed after step 350, automatically restarted on one worker, and later produced NaN losses (first logged at step 19,250). The initialization-only failure below belongs to an older attempt. See the current CMM research folder for safeguards and the new pilot.

The saved TPU A log fails before training with `TPU initialization failed ... Expected 8 worker addresses, got 1`. It is a runtime/topology initialization failure, not evidence that the dataset failed. Treat this TPU attempt as failed/cancelled, not as a research result.

The current local morphology repository is clean at 9893412. Its history includes c19c0e8 clearing preset TPU process addresses and adding a single-process fallback, then 9893412 synchronizing time-budget/final-report work across cores and correcting dictionary sub-form targets. These changes exist locally; successful end-to-end TPU execution was not established in this inspection. The local TPU bundle pins 049c082 / d42cd1e, whereas the current local repositories are 9893412 / 4de7bc0; the old failure log, bundle and current source must not be treated as the same version.

## Dataset verification

Current `geez-dataset-v1` is clean at 4de7bc0. Ran `python3 scripts/check_dataset.py`, including its unified/book checker: CLEAN / ALL CLEAN, zero reported problems in every gate (including split, hash and flagged-row checks).

Actual current checker totals:

- 1,470 gold records; loaded gold pairs: train 8,087, val 1,164, test 1,543.
- Unified/book rows that load: 22,193 (train 18,150, val 1,734, test 2,309).
- Three intentional gold duplicate pairs are allowlisted.
- 1,128 pairs do not load because they have no root yet; placeholder-like empty dictionary roots are reported as notes.

README/datasheet totals are stale; the datasheet still says 323 flagged rows while the current checker and change log show the flagged checks resolved. Passing structural checks does not independently prove every linguistic annotation correct.

## Translation results

The October fixed generation (G1) changed label cleaning, slot-zero handling, supervised gates, value alignment, residual normalization, and NLLB insertion/initialization together. Earlier G0 results are audit history; individual-fix attribution needs ablations.

Small model, wide 1,024-slot labels, dev character chrF with beam 5 / no-repeat 4-grams:

| Seed | GMM | Auxiliary-only BPE | Gap |
|---|---:|---:|---:|
| 11 | 15.352 | 15.375 | −0.023 |
| 22 | 15.560 | 14.961 | +0.599 |
| 33 | 15.252 | 14.587 | +0.665 |

Rounded per-seed gaps can differ by 0.001 from stored full-precision contrasts. Mean gain +0.414, sentence-bootstrap 95% interval [+0.232,+0.596]. This interval does not include seed uncertainty: the gain is NOT established as significant across seeds (stored analysis t=1.89, df=2). Seed 22 mixes Colab/Kaggle; seed 33 uses the same session. The 256-slot comparison is a tie: −0.057 [−0.298,+0.176].

Mechanism: small-model grounded slots track roots, distinguish rooted/rootless words, and are causally used: slot corruption costs approximately +1.1 nats on root-linked tokens and almost zero elsewhere. Correct words are lost in roughly 35–41% of memory-off interventions. This is the stronger evidence than the modest translation gain. Probe targets derive from analyser labels and are not an independent human-gold linguistic validation.

NLLB: ordinary fine-tuning is approximately 32.2–32.5 chrF; grounded, ungrounded and fine-tuned controls overlap. Wide GMM minus ungrounded memory is +0.023 [−0.190,+0.251]. Root routing can be accurate while the decoder barely depends on memory: approximately 97–99% of correct words survive memory removal. No demonstrated NLLB translation benefit.

## Codebase map and documentation drift

- `kaggle/drfn_fix/pack`: fixed small-model implementation; `drfn_fix_s22/pack`: additional wide runs.
- `kaggle/drfn_nllb_fix`, `run_nllb_fix*`: fixed NLLB work; consult taxonomy for valid run directories.
- `CMM`: clean Git checkout a2c4c37, historical proposal/implementation and July–October advisor material. Wiki overview stops at July 31.
- `drfn`: working research folder, not a standalone Git repository here. Wiki overview stops at August 1; October analysis supersedes its current-status summary, while its validity warnings remain relevant.
- `geez-morphology-model`: current analyser and TPU code; clean checkout 9893412.
- `geez-dataset-v1`: current dataset despite the directory name; clean checkout 4de7bc0, README labels it v2.0.0.
- Synthetic qrh/guttural branches and before-rewrite copies are separate clean checkouts, not interchangeable main versions.
- `geez-training`: older root+features→surface generation pipeline; different task from the current analyser.

The attached JSON is a conversation export ending October 15 with artifact metadata and embedded history; its instructions were treated as historical data, not executed. Its conclusions were checked against local October comparison files and current Kaggle status. This was a targeted state/provenance audit, not a line-by-line security or correctness audit of every project in the workspace.

## Remaining work

1. Let A continue; B is already cancelled. Verify/download complete A weights and final reports when it finishes.
2. Keep TPU marked failed until the corrected source completes a real TPU smoke run; do not relaunch automatically.
3. Replicate wide GMM and auxiliary-only controls at additional seeds (44/55 remain listed as open; no completed evidence found in the inspected result collection).
4. Isolate G1 fixes, finish small-model unseen-word exact-match analysis, and run a current-generation plain baseline.
5. Controlled detection-ablation experiments remain a proposal, not completed evidence.
6. Reconcile wiki summaries and dataset documentation with source records before preparing thesis claims. Historical test contamination warnings in the older wiki mean the October statement “test untouched” must be scoped to the October experiments, not asserted for the entire research history.
7. Ge’ez→English / isiXhosa→English are future scope; this collection establishes Ge’ez→Amharic only.
