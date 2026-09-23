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

**Not included in this repository** (referenced by path in the scripts,
but too large / from an external repository):
- `data.txt` — the wide-format, one-row-per-genotype phenotype summary
  (columns `Taxa, DS1, DS2, DS3`) that most scripts actually read. This is
  a simple per-genotype, per-year summary of `Data_3Reps.csv`; regenerate
  it with a mean or mixed-model BLUE across the 3 replicates per
  genotype × year, or contact the corresponding author for the exact file
  used in the reported analyses.
- `Genotype.Numerical.txt` — the ~20,996-SNP numeric (0/1/2) marker matrix.
  Publicly available through the CIMMYT research data repository (see the
  manuscript's Data Availability Statement for the accession).

Every script's `PHENO_PATH` / `GENO_PATH` (or `PHENO_PATH`/`GENO_PATH` in
`R_reviewer_response/00_config.R`) must be edited to point at your local
copies of these two files before running.

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

### Reviewer-response analyses (`R_reviewer_response/`)

| Script | Produces | Manuscript reference |
|---|---|---|
| `00_config.R` | — | Shared paths and loader functions, sourced by every script below; not run directly |
| `01_loyo_year_fixed_and_genetic_corr.R` | `LOYO_year_fixed_effect_results.csv`, `cross_year_genetic_correlation.csv` | The year-adjusted LOYO re-analysis and the cross-year genetic-correlation estimate reported in the Discussion (Section 3) and reflected in the Abstract and Conclusions |
| `02_relatedness_stratified_cv_per_year.R` | `per_year_relatedness_grouped_CV.csv` | The relatedness-stratified per-year cross-validation check reported in Section 2.9, described there as an independent kernel-based check (not a direct row of Table S2, since it uses a different RKHS implementation — BGLR's native kernel regression on the GRM — than the scikit-learn `KernelRidge` used for the manuscript's tabulated "RKHS" model) |
| `03_generate_rf_importance.R` | `rf_importance_DS1.csv`, `rf_importance_DS2.csv`, `rf_importance_DS3.csv` | Per-SNP random forest importance scores, the direct input to the marker-enrichment test below. (Chromosome-level *sums* of these scores, not the per-SNP values themselves, underlie the chromosome-wise comparison in the Discussion and Figure 3's companion discussion in Section 2.5.) |
| `05_pca_discrepancy_diagnostic.R` | Console output only (three PCA variance-explained values) | The explanation, in the Discussion, of why this study's population-structure PCA (Figure 2A: PC1 = 10.3%, PC2 = 5.2%) differs from the companion GWAS study's PCA on the same genotypes and marker set (PC1 = 43.3%, PC2 = 18.8%) |
| `08_rf_gwas_enrichment_test.R` | Console output (observed/expected hits, fold-enrichment, p-value per year) | The formal marker-level enrichment test reported in the Discussion (DS2: 3.3-fold enrichment, p = 0.0002; DS1/DS3: not significant), replacing the earlier chromosome-level-only comparison |

## Not included / known gaps

Stated plainly rather than left for a reviewer to discover:

1. **Figure 2** (population structure / PCA plot) — no plotting script is
   included in this repository. `R_reviewer_response/05_pca_discrepancy_diagnostic.R`
   computes the underlying PCA variance-explained values but does not
   generate the figure itself.
2. **Figure 4** (marker-density and training-population-size sensitivity)
   — no script is included. The values discussed in Section 2.8 and
   Section 4.7 of the manuscript are not reproducible from this
   repository alone.
3. **Table 1, Table 2** (descriptive statistics, heritability, ANOVA) and
   **Table S1** (descriptive statistics corroborating Table 1) — these
   reproduce results from the companion genome-wide association study of
   this panel and trait and are not re-derived by any script here.
4. **Table S5** (formal paired-permutation comparison) — as noted in the
   mapping table above, `python/01_benchmark_13_conventional_ml.py`'s
   built-in permutation test uses a different five-model comparison set
   than the manuscript text describes; this needs a small edit
   (`rep_models = [...]` in that script) and a re-run before its output
   file matches the reported Table S5 exactly.
5. **Table S8** (ordinal-outcome vs. Gaussian-outcome sensitivity check),
   **Table S9** (wall-clock computational time), **Table S10** (RKHS
   kernel-bandwidth sensitivity), and **Table S11** (marker-density
   sensitivity across three repeated random partitions) — no generating
   scripts for any of these four are included in this repository.
6. **Item 2's exact leakage delta** — `02_relatedness_stratified_cv_per_year.R`
   reports relatedness-stratified per-year accuracy, but a like-for-like
   comparison against the random-split Table S2 values, for the same
   model, has not yet been computed here.
7. **Item 8's enrichment test position coverage** — `08_rf_gwas_enrichment_test.R`
   only tests the subset of SNPs with a genetic-map position (14,328 of
   20,996 total), since no complete SNP-to-position map for the full
   marker set is included in this repository.
8. **BayesR** has no native BGLR implementation and is retained throughout
   as a disclosed regularized-regression proxy (see `R/01_bglr_fold.R`'s
   header comment and `python/03_merge_and_build_table4.py`'s
   `BAYESR_PROXY_VALUES`).
9. **Leave-one-year-out cross-validation** (**Table S6a/b/c**) for the
   original 30-model comparison, and the six-model paired-permutation
   comparison (**Table S5**) — as distinct from the year-fixed-effect
   re-analysis in `R_reviewer_response/01_loyo_year_fixed_and_genetic_corr.R`
   — were not re-run with the genuine BGLR models; their Bayesian-alphabet
   rows still reflect the original proxy implementation. This is
   disclosed in-text (Section 2.9) and in the corresponding Supplementary
   Table notes.

## Requirements

**R** (tested with BGLR via CRAN):
```r
install.packages(c("BGLR", "sommer", "ggplot2", "patchwork"))
```

**Python:**
```bash
pip install scikit-learn pandas numpy scipy xgboost lightgbm tensorflow
```

## Running the full pipeline

```bash
# 1. Confirm BGLR works
Rscript R/00_smoke_test.R

# 2. Main pipeline: genuine BGLR (long-running: real MCMC, nIter=12000,
#    burnIn=2000 by default — hours, not minutes)
Rscript R/02_bglr_per_year_and_pooled_ordinary.R
Rscript R/03_bglr_pooled_grouped.R

# 3. Main pipeline: the other 24 models
python3 python/01_benchmark_13_conventional_ml.py
python3 python/02_benchmark_10_deep_learning.py

# 4. Merge into the final Table 4 / Table S4
python3 python/03_merge_and_build_table4.py

# 5. Figure 3 (paste into R/RStudio, or:)
Rscript R/figure3_per_year_accuracy.R

# 6. Reviewer-response analyses (each independent; edit the paths in
#    00_config.R first)
Rscript R_reviewer_response/01_loyo_year_fixed_and_genetic_corr.R
Rscript R_reviewer_response/02_relatedness_stratified_cv_per_year.R
Rscript R_reviewer_response/03_generate_rf_importance.R
Rscript R_reviewer_response/05_pca_discrepancy_diagnostic.R
Rscript R_reviewer_response/08_rf_gwas_enrichment_test.R   # after joining
                                                             # a full marker-
                                                             # position map
                                                             # onto the RF
                                                             # importance CSVs
```

All scripts write their output to `./output/` by default (edit the
`OUT_DIR` / `OUTPUT_DIR` / `SCRIPTS_DIR` constant at the top of each
script to change this).

## Notes on runtime

BGLR does real MCMC sampling per fold (`nIter=12000`, `burnIn=2000` by
default) — the two `R/02_...`/`R/03_...` scripts together are on the order
of hours, depending on your machine. Both save progress incrementally and
can be safely re-run after an interruption.

## License / citation

[Add license and citation information before publishing this repository.]
