# Genomic Prediction for Powdery Mildew Resistance in Wheat — Analysis Code

Code accompanying "Genomic Prediction for Powdery Mildew Resistance in
Wheat: A Thirty-Model Benchmark of Conventional, Machine-Learning, and
Deep-Learning Approaches" (WAMI panel, n = 286, ~20,996 SNPs, DS1–DS3).

This repository is organized to answer, directly and without ambiguity,
the reviewer request that motivated it: **does the code actually support
reproduction of every reported result?** Below, every table, supplementary
table, and figure in the manuscript is matched to the script that produces
it. Where no such script exists in this repository, that is stated
explicitly rather than left implicit — see **"Not included / known gaps"**
near the end.

**A note on numbering**: the manuscript's Supplementary Tables were
renumbered during revision to put every citation in ascending order at
first mention (S1–S12, as listed below). Some scripts' own internal file
names and code comments still use the pre-renumbering labels (for
example, `python/03_merge_and_build_table4.py` writes
`table_S5b_leakage_delta.csv`, which is the *old* label for what is now
**Table S4**). The mapping table below always gives the *current*
manuscript number; where a script's own output filename differs, that is
noted directly in the table.

## Repository structure

```
R/                     Main 30-model pipeline: genuine BGLR fits (6 models),
                        Figure 3
python/                Main 30-model pipeline: 13 conventional/ML models,
                        10 DL models, and the final Table 4 merge
R_reviewer_response/   Analyses added to address specific reviewer/editor
                        comments (year-fixed-effect LOYO, cross-year genetic
                        correlation, relatedness-stratified per-year CV, RF
                        importance for the marker-enrichment test, the PCA
                        scaling-convention diagnostic, and the enrichment
                        test itself)
data/                  Phenotype data (Data_3Reps.csv) and the GWAS
                        marker-position reference (mta_table_kaur2023.csv)
                        actually included in this repository; see "Data" below
                        for what is NOT included and why
output/                Empty; every script writes its results here
```

## Data

**Included in this repository:**
- `data/Data_3Reps.csv` — the raw, replicated phenotype data: 286
  genotypes × 3 years (DS1–DS3) × 3 field replicates = 2,574 rows, columns
  `Taxa, Rep, Environment, Score`. This is the source data from which
  per-genotype, per-year values were derived.
- `data/mta_table_kaur2023.csv` — chromosome and genetic-map (cM) position
  for the 113 marker–trait associations reported in the companion
  genome-wide association study of this panel and trait, transcribed from
  that paper's Table 3. Used by `R_reviewer_response/08_rf_gwas_enrichment_test.R`.

## Script-to-output mapping

### Main pipeline (`R/`, `python/`)

| Script | Produces | Manuscript reference |
|---|---|---|
| `R/00_smoke_test.R` | — | Confirms BGLR installs/runs correctly; not a results-producing script |
| `R/01_bglr_fold.R` | — | Core function library (`run_bglr_fold()`), sourced by the two scripts below; not run directly |
| `R/02_bglr_per_year_and_pooled_ordinary.R` | `BGLR_PerYear_5fold_summary.csv`, `BGLR_Pooled_ordinary_summary.csv` | Per-year results for BayesA/BayesB/BayesC/BayesCπ/BRR/BL (part of **Table S2**, full per-model per-year accuracy); pooled *ordinary* (non-grouped) CV for these 6 models, feeding the leakage-delta comparison in **Table S4** (genotype-leakage effect) |
| `R/03_bglr_pooled_grouped.R` | `BGLR_Pooled_grouped_summary.csv` | Pooled, genotype-grouped (leakage-safe) CV for the same 6 models — the Bayesian-alphabet rows of **Table 4** and its machine-readable companion **Table S3** |
| `python/01_benchmark_13_conventional_ml.py` | `genuine_pooled_summary.csv`, `genuine_per_year_5fold_all_folds.csv`, `genuine_leakage_delta_by_model.csv`, `genuine_paired_permutation_test.csv` | The 13 non-Bayesian, non-DL models' rows of **Table 4**, **Table S2**, **Table S4**; a representative-model paired-permutation comparison (**Table S5**, formal paired-permutation comparison — *note*: the script's own representative-model set, GBLUP/rrBLUP/RandomForest/XGBoost/RKHS, differs from the six models named in the manuscript text, GBLUP/rrBLUP/ElasticNet/BayesR/BayesC/RKHS; re-run with the manuscript's model set before treating this output as the final Table S5 source) |
| `python/02_benchmark_10_deep_learning.py` | DL-model equivalents of the above (`genuine_dl10_pooled_summary.csv`, etc.) | The 10 DL models' rows of **Table 4** and **Table S2** |
| `python/03_merge_and_build_table4.py` | `table4_grouped_final.csv`, `table_S5b_leakage_delta.csv` | Merges all three pipelines above into the final **Table 4** (BayesR added back as a disclosed proxy) and the pooled leakage-delta table (**Table S4**) |
| `R/figure3_per_year_accuracy.R` | `Figure3_per_year_accuracy.png` / `.pdf` | **Figure 3** (per-year accuracy, all 30 models). Uses values matching **Table S2**, hardcoded directly in the script rather than read from a CSV — if `Table S2`/`BGLR_PerYear_5fold_summary.csv` values are ever updated, this script's `DS1_vec`/`DS2_vec`/`DS3_vec` must be updated to match by hand |
