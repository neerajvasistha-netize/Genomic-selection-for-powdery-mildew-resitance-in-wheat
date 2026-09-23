#!/usr/bin/env Rscript
# 03_bglr_pooled_grouped.R
#
# ISSUE #2 FIX (genotype leakage), BGLR half -- genuine BGLR numbers for
# the pooled, combined-across-years analysis under GENOTYPE-GROUPED
# cross-validation: all of a genotype's observations, across DS1/DS2/DS3,
# stay in the same fold, so no genotype's marker profile can appear in
# both training and validation via a different year's phenotype.
#
# This is the leakage-safe counterpart to 02_bglr_per_year_and_pooled_ordinary.R's
# Part B, and mirrors what 06_benchmark_13_conventional_ml.py and
# 07_benchmark_10_deep_learning.py do for the other 24 models via
# scikit-learn's GroupKFold. Produces the 5 schemes that make up the
# primary, leakage-safe Table 4: pooled_grouped_5fold, pooled_grouped_10fold,
# pooled_grouped_5fold_rep1/2/3.
#
# BUILT-IN RESUME: if this script is re-run after an interruption, it
# reads whatever's already in the partial-output files and skips any
# (scheme, model) combination that's already complete, rather than
# recomputing from zero. Safe to just re-run this same script after any
# crash/interruption.
#
# REQUIREMENTS: R with the BGLR package installed (install.packages("BGLR"))
#
# RUNTIME: 6 models x 5 folds x 5 schemes = 150 BGLR fits -- budget
# several hours. Incremental saving means an interruption never loses
# more than the model currently in progress.

source("bglr_fold.R")

# --------------------------------------------------------------------------
# CONFIG -- edit these paths for your machine
# --------------------------------------------------------------------------
PHENO_PATH  <- "./data/data.txt"
GENO_PATH   <- "./data/Genotype.Numerical.txt"
OUT_DIR     <- "./output"
TRAITS      <- c("DS1", "DS2", "DS3")
N_ITER      <- 12000
N_BURNIN    <- 2000
BGLR_MODELS <- c("BayesA", "BayesB", "BayesC", "BayesCpi", "BRR", "BL")

dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
SUMMARY_PARTIAL <- file.path(OUT_DIR, "BGLR_Pooled_grouped_summary_partial.csv")
BY_FOLD_PARTIAL <- file.path(OUT_DIR, "BGLR_Pooled_grouped_by_fold_partial.csv")

scheme_n_splits <- c(
  pooled_grouped_5fold = 5, pooled_grouped_10fold = 10,
  pooled_grouped_5fold_rep1 = 5, pooled_grouped_5fold_rep2 = 5, pooled_grouped_5fold_rep3 = 5
)

# --------------------------------------------------------------------------
# LOAD DATA
# --------------------------------------------------------------------------
pheno <- read.table(PHENO_PATH, header = TRUE, sep = "\t")
geno  <- read.table(GENO_PATH,  header = TRUE, sep = "\t")

colnames(pheno)[1] <- "GID"
colnames(geno)[1]  <- "GID"
common <- intersect(as.character(pheno$GID), as.character(geno$GID))
cat("Matched", length(common), "genotypes.\n")

Xg <- as.matrix(geno[match(common, as.character(geno$GID)), -1])
Xg <- Xg[, apply(Xg, 2, function(col) var(col, na.rm = TRUE) > 1e-6)]
Xg <- scale(Xg)
rownames(Xg) <- common

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
groups_long <- long_df$GID
cat("Pooled dataset:", nrow(X_long), "observations across", length(unique(groups_long)), "genotypes.\n")

# --------------------------------------------------------------------------
# RESUME: load whatever's already done from a previous (possibly
# interrupted) run of this same script
# --------------------------------------------------------------------------
pooled_summary <- data.frame()
pooled_detail  <- data.frame()
done_combos <- character(0)

if (file.exists(SUMMARY_PARTIAL) && file.exists(BY_FOLD_PARTIAL)) {
  pooled_summary <- read.csv(SUMMARY_PARTIAL)
  pooled_detail  <- read.csv(BY_FOLD_PARTIAL)
  fold_counts <- aggregate(Fold ~ Scheme + Model, pooled_detail, length)
  for (i in seq_len(nrow(fold_counts))) {
    sch <- fold_counts$Scheme[i]; mod <- fold_counts$Model[i]; n <- fold_counts$Fold[i]
    expected <- scheme_n_splits[[sch]]
    if (!is.null(expected) && n == expected) done_combos <- c(done_combos, paste(sch, mod))
  }
  cat("Resuming: found", length(done_combos), "already-completed (scheme, model) pairs.\n")
} else {
  cat("No partial output found -- starting fresh.\n")
}

# --------------------------------------------------------------------------
# GENOTYPE-GROUPED FOLD ASSIGNMENT (leakage-safe)
# Shuffle unique genotype IDs, assign folds round-robin, so every
# observation from a given genotype lands in the same fold.
# --------------------------------------------------------------------------
grouped_folds <- function(groups, n_splits, seed) {
  set.seed(seed)
  unique_groups <- unique(groups)
  shuffled <- sample(unique_groups)
  fold_of_group <- setNames(rep(1:n_splits, length.out = length(shuffled)), shuffled)
  fold_of_group[as.character(groups)]
}

run_grouped_scheme <- function(scheme_label, n_splits, seed) {
  folds <- grouped_folds(groups_long, n_splits, seed)
  for (model_name in BGLR_MODELS) {
    combo_key <- paste(scheme_label, model_name)
    if (combo_key %in% done_combos) {
      cat(scheme_label, model_name, "already done -- skipping\n")
      next
    }
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
    write.csv(pooled_summary, SUMMARY_PARTIAL, row.names = FALSE)
    write.csv(pooled_detail,  BY_FOLD_PARTIAL, row.names = FALSE)
  }
}

cat("\n--- Pooled grouped 5-fold ---\n")
run_grouped_scheme("pooled_grouped_5fold", 5, 42)

cat("\n--- Pooled grouped 10-fold ---\n")
run_grouped_scheme("pooled_grouped_10fold", 10, 42)

cat("\n--- Pooled grouped 5-fold repeated x3 ---\n")
for (s in c(1, 2, 3)) {
  run_grouped_scheme(paste0("pooled_grouped_5fold_rep", s), 5, s)
}

cat("\n=== GROUPED POOLED SUMMARY (genuine BGLR, leakage-safe -- feeds Table 4) ===\n")
print(pooled_summary)
write.csv(pooled_summary, file.path(OUT_DIR, "BGLR_Pooled_grouped_summary.csv"), row.names = FALSE)
write.csv(pooled_detail,  file.path(OUT_DIR, "BGLR_Pooled_grouped_by_fold.csv"), row.names = FALSE)
cat("\nDone. Next: run 04_merge_pooled_results.R.\n")
