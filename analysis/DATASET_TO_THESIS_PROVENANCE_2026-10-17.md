# Ge’ez dataset → thesis provenance

Inspected 17 October 2026. No training jobs launched or modified.

## Verified chain

1. Older analyser output: `../run_drfn_labels/pack/runs/shared/analyser_cache.json`, field `meta.model_dir`, identifies `esubaalew/geez-morphology-byt5-training/byt5_3dc784e_337cdf0_books_s1/best_weights_by_val_score`. The run name encodes morphology code 3dc784e and dataset 337cdf0; this is a provenance label, not an independent weights hash verification. Cached config SHA256 is 9bd0505b394be8820b409c1e1e2dc88a19d0f3cc91f92a0bb7aad78b359ce608.
2. `../drfn_analyser/pack/build_analyser_cache.py` loads that ByT5 checkpoint and analyses unique Ge’ez words from AGE train/dev. Cache: 47,233 types, 4,024 unparsed, 42,955 with roots. Root/pattern inventories are ranked on training words. These predictions feed auxiliary-head experiments, grounded routing, and exploratory analyser-input experiments.
3. `../drfn_fix/build_clean_labels.py` reads the old predictions PLUS `../../geez-dataset-v1/data/unified/geez_morphology_rows.jsonl` and `../../geez-morphology-model/data/closed_class_words_without_root.tsv`. Curated forms override predicted roots/patterns; rootless/function words and names are handled explicitly; remaining predictions undergo a consonant check. The script reads the unified file without filtering its split column. External curated lexical annotations therefore contribute to dev labels too; these probes are not an independent evaluation against newly human-annotated gold. The exact source dataset commit/hash used at build time is not recorded in the output cache.
4. Outputs: 256 root classes or 1,024 root classes (including none), plus 17 pattern classes. Of 204,429 source training tokens, 64,417 (31.51%) receive in-bank root IDs in the 256 cache; 90,324 (44.18%) in the wide cache. Zero includes both genuinely rootless and out-of-inventory words. These figures are word-token coverage, not analyser accuracy or subword coverage.
5. Fixed training uses labels for auxiliary heads and, in GMM, routing, gate and value supervision. Amharic diagnostic target maps are derived from AGE training-pair co-occurrences and consonant signatures; they are not manually gold target annotations. Source labels are training/probe targets; ordinary GMM translation inference predicts its routes from hidden states rather than querying the ByT5 analyser.

## Evidence that completed runs actually used these caches

256 cache SHA256: `97b48e20e769df15365732e9320e866e676426049f94b8647d5d3a74ce94dcd8` matches saved metadata for all six fixed clean small-model runs (GMM, auxiliary-only and cmm_spec, seeds 11/22).

Wide SHA256: `dd6b2b9387ac37336767135b733db29e0e182094fb7b007c41a7299758857681` matches both GMM and auxiliary-only completed run metadata for seeds 11/22/33 across Colab and Kaggle. The local small-model, extra-seed and NLLB fix packs contain byte-identical wide caches. The Colab auxiliary seed-22 metadata also exists but has no completed score; the completed Kaggle replacement is used for that seed.

## What helped, and what is not isolated

The dataset supplied linguistic resources through analyser training and then direct label correction. GMM and auxiliary-only controls shared the same cleaned cache, so the wide +0.414 chrF contrast tests their different use of the same resource. It does not measure the dataset’s independent contribution. Small GMM root consistency and causal slot interventions support a learned/useful memory mechanism, but independent label-quality validation remains separate.

There is no single-factor comparison separating dataset corrections from the simultaneous gate/value/normalization fixes. Widening labels increased in-bank word-token coverage from 31.51% to 44.18%, not measured detection accuracy from 31.51% to 44.18%.

The new GPU A/B bundles use code a1ad2b6 / data 04f45a7. No link from those new checkpoints into the inspected completed thesis caches/runs was found. The TPU attempt likewise supplied no verified trained output to these thesis runs. The newer dataset/analyser can support a future controlled label-quality experiment, but this has not been demonstrated yet.

AGE supplies the thesis sentence translation pairs. The user’s Ge’ez morphology dataset supplies annotations/linguistic supervision through the chain above; these are distinct roles.
