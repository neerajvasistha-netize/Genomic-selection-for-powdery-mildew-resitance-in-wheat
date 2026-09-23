#!/usr/bin/env Rscript
# 02_bglr_per_year_and_pooled_ordinary.R
#
# Genuine, MCMC-sampled BGLR fits (BayesA, BayesB, BayesC, BayesCpi, BRR, BL)
# for the powdery-mildew genomic-prediction benchmark, covering:
#   (a) per-year five-fold CV, each of the three years separately
#   (b) pooled (combined-across-years) CV under ORDINARY, observation-level
#       random splitting -- five-fold, ten-fold, and five-fold repeated x3
#
# This replaces the earlier regularized-regression proxy implementation
# (sklearn Ridge/Lasso/ElasticNet/BayesianRidge) used for these seven
# "Bayesian-alphabet" models with real MCMC-sampled Bayesian regression.
# BayesR is intentionally excluded: BGLR has no native BayesR routine
# (a four-component mixture prior), so it remains a disclosed proxy
# elsewhere in the pipeline rather than being silently substituted here.
#
# The pooled analysis here uses ORDINARY (ungrouped) folds deliberately,
# to isolate the proxy -> genuine swap from the separate genotype-leakage
# fix. See 03_bglr_pooled_grouped.R for the leakage-safe (genotype-grouped)
# rerun of the same pooled analysis.
#
# REQUIREMENTS: R with the BGLR package installed (install.packages("BGLR"))
#
# USAGE: edit the CONFIG block below, then `Rscript 02_bglr_per_year_and_pooled_ordinary.R`
#        or Source this file in RStudio.
#
# RUNTIME: real MCMC per fold. 6 models x 5 folds x (3 years + 5 pooled
# schemes) = 240 BGLR fits. At nIter=12000/burnIn=2000 this can take
# several hours depending on your machine; incremental saving (below)
# means an interruption does not lose completed work.

source("bglr_fold.R")

# --------------------------------------------------------------------------
# CONFIG -- edit these paths for your machine
# --------------------------------------------------------------------------
PHENO_PATH  <- "./data/data.txt"                        # phenotype file: Taxa, DS1, DS2, DS3 (tab-separated)
GENO_PATH   <- "./data/Genotype.Numerical.txt"           # marker file: taxa + SNP columns, 0/1/2 coded (tab-separated)
OUT_DIR     <- "./output"
TRAITS      <- c("DS1", "DS2", "DS3")
N_ITER      <- 12000     # drop to ~3000 for a fast first pass; raise for final numbers
N_BURNIN    <- 2000      # drop to ~500 for a fast first pass
SEED        <- 42
BGLR_MODELS <- c("BayesA", "BayesB", "BayesC", "BayesCpi", "BRR", "BL")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
pheno <- read.table(PHENO_PATH, header = TRUE, sep = "\t")
geno  <- read.table(GENO_PATH,  header = TRUE, sep = "\t")

colnames(pheno)[1] <- "GID"
colnames(geno)[1]  <- "GID"
common <- intersect(as.character(pheno$GID), as.character(geno$GID))
cat("Matched", length(common), "genotypes between pheno and geno files.\n")

Xg <- as.matrix(geno[match(common, as.character(geno$GID)), -1])
Xg <- Xg[, apply(Xg, 2, function(col) var(col, na.rm = TRUE) > 1e-6)]  # drop monomorphic markers
Xg <- scale(Xg)
rownames(Xg) <- common
cat("Marker matrix after filtering:", nrow(Xg), "x", ncol(Xg), "\n")

# --------------------------------------------------------------------------
# PART A: PER-YEAR FIVE-FOLD CV
# --------------------------------------------------------------------------
per_year_summary <- data.frame()
per_year_detail  <- data.frame()

for (trait in TRAITS) {
  y_trait <- pheno[match(common, as.character(pheno$GID)), trait]
  keep <- !is.na(y_trait)
  X_t <- Xg[keep, ]
  y_t <- y_trait[keep]
  cat("\n===", trait, "-", length(y_t), "phenotyped lines ===\n")

  set.seed(SEED)
  folds <- sample(rep(1:5, length.out = nrow(X_t)))

  for (model_name in BGLR_MODELS) {
    fold_r <- c()
    for (k in 1:5) {
      tr <- which(folds != k); te <- which(folds == k)
      res <- run_bglr_fold(X_t[tr, ], y_t[tr], X_t[te, ], y_t[te],
                            model_name = model_name, nIter = N_ITER, burnIn = N_BURNIN)
      r <- cor(res$y_true, res$y_pred)
      fold_r <- c(fold_r, r)
      per_year_detail <- rbind(per_year_detail,
                                data.frame(Trait = trait, Model = model_name, Fold = k, r = r))
      cat(trait, model_name, "fold", k, "r =", round(r, 4), "\n")
    }
    per_year_summary <- rbind(per_year_summary,
                               data.frame(Trait = trait, Model = model_name,
                                          Mean_r = mean(fold_r), SD_r = sd(fold_r)))
    # incremental save after every model, so an interruption doesn't lose everything
    write.csv(per_year_summary, file.path(OUT_DIR, "BGLR_PerYear_5fold_summary_partial.csv"), row.names = FALSE)
    write.csv(per_year_detail,  file.path(OUT_DIR, "BGLR_PerYear_5fold_by_fold_partial.csv"), row.names = FALSE)
  }
}

cat("\n=== PER-YEAR SUMMARY (feeds manuscript Section 2.6 / Table S4) ===\n")
print(per_year_summary)
write.csv(per_year_summary, file.path(OUT_DIR, "BGLR_PerYear_5fold_summary.csv"), row.names = FALSE)
write.csv(per_year_detail,  file.path(OUT_DIR, "BGLR_PerYear_5fold_by_fold.csv"), row.names = FALSE)

# --------------------------------------------------------------------------
# PART B: POOLED ANALYSIS ACROSS ALL 3 YEARS -- ORDINARY (UNGROUPED) CV
# --------------------------------------------------------------------------
long_rows <- list()
for (trait in TRAITS) {
  y_trait <- pheno[match(common, as.character(pheno$GID)), trait]
  for (i in seq_along(common)) {
    if (!is.na(y_trait[i])) {
      long_rows[[length(long_rows) + 1]] <- data.frame(GID = common[i], Year = trait, y = y_trait[i])
    }
  }
}
long_df <- do.call(rbind, long_rows)
X_long <- Xg[long_df$GID, ]
y_long <- long_df$y
cat("\nPooled dataset:", nrow(X_long), "observations across", length(TRAITS), "years.\n")

pooled_summary <- data.frame()
pooled_detail  <- data.frame()

run_pooled_scheme <- function(scheme_label, n_splits, seed) {
  set.seed(seed)
  folds <- sample(rep(1:n_splits, length.out = nrow(X_long)))
  for (model_name in BGLR_MODELS) {
    fold_r <- c()
    for (k in 1:n_splits) {
      tr <- which(folds != k); te <- which(folds == k)
      res <- run_bglr_fold(X_long[tr, ], y_long[tr], X_long[te, ], y_long[te],
                            model_name = model_name, nIter = N_ITER, burnIn = N_BURNIN)
      r <- cor(res$y_true, res$y_pred)
      fold_r <- c(fold_r, r)
      pooled_detail <<- rbind(pooled_detail,
                               data.frame(Scheme = scheme_label, Model = model_name, Fold = k, r = r))
      cat(scheme_label, model_name, "fold", k, "r =", round(r, 4), "\n")
    }
    pooled_summary <<- rbind(pooled_summary,
                              data.frame(Scheme = scheme_label, Model = model_name,
                                         Mean_r = mean(fold_r), SD_r = sd(fold_r)))
    write.csv(pooled_summary, file.path(OUT_DIR, "BGLR_Pooled_ordinary_summary_partial.csv"), row.names = FALSE)
    write.csv(pooled_detail,  file.path(OUT_DIR, "BGLR_Pooled_ordinary_by_fold_partial.csv"), row.names = FALSE)
  }
}

cat("\n--- Pooled 5-fold (ordinary/ungrouped) ---\n")
run_pooled_scheme("pooled_5fold", 5, SEED)

cat("\n--- Pooled 10-fold (ordinary/ungrouped) ---\n")
run_pooled_scheme("pooled_10fold", 10, SEED)

cat("\n--- Pooled 5-fold repeated x3 (ordinary/ungrouped) ---\n")
for (s in c(1, 2, 3)) {
  run_pooled_scheme(paste0("pooled_5fold_rep", s), 5, s)
}

cat("\n=== POOLED SUMMARY, ORDINARY CV (feeds the leakage-delta comparison, Table S5b) ===\n")
print(pooled_summary)
write.csv(pooled_summary, file.path(OUT_DIR, "BGLR_Pooled_ordinary_summary.csv"), row.names = FALSE)
write.csv(pooled_detail,  file.path(OUT_DIR, "BGLR_Pooled_ordinary_by_fold.csv"), row.names = FALSE)

cat("\nDone. Output written to:", OUT_DIR, "\n")
cat("Next: run 03_bglr_pooled_grouped.R for the leakage-safe (genotype-grouped) pooled results,\n")
cat("then 04_merge_pooled_results.R to combine everything with the Python pipeline outputs.\n")
