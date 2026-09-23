## =============================================================================
## Item 2 (R1#6): Genotype-grouped / relatedness-stratified CV WITHIN each year,
## and the leakage delta vs. the original random-split per-year numbers.
## =============================================================================
source("/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/00_config.R")
library(BGLR)

pheno_wide <- load_pheno_wide()
M          <- load_geno_matrix()
check_id_overlap(pheno_wide, M, context = "02_relatedness_cv")
G          <- build_grm(M)

## --- Step 1: define relatedness-based family clusters ----------------------
## Hierarchical clustering on genomic distance (1 - scaled GRM). Start with
## k ~ 20-30 and inspect the dendrogram to confirm it matches the "elevated
## mutual relatedness" cluster already flagged in the manuscript (Section 2.2).
Gs <- G / max(diag(G))
dist_mat <- as.dist(1 - Gs)
hc <- hclust(dist_mat, method = "average")
k  <- 25                                    # <-- tune this; inspect plot(hc)
families <- cutree(hc, k = k)
family_df <- data.frame(Taxa = rownames(G), family = families, stringsAsFactors = FALSE)

pheno_wide <- merge(pheno_wide, family_df, by = "Taxa")

## --- Step 2: group-aware 5-fold CV within each year -------------------------
## All genotypes in the same family go to the SAME fold, so no family is
## split across train/test (the leave-family-out analogue of the
## genotype-grouped scheme already used for the pooled analysis).

group_kfold_ids <- function(family_vec, k = 5, seed = 1) {
  set.seed(seed)
  fam_levels <- unique(family_vec)
  fam_fold   <- sample(rep(1:k, length.out = length(fam_levels)))
  names(fam_fold) <- fam_levels
  fam_fold[as.character(family_vec)]
}

run_grouped_cv_one_year <- function(taxa, blue, family, G, k = 5, nIter = 12000, burnIn = 2000, tag = "") {
  fold_id <- group_kfold_ids(family, k = k)
  accs <- numeric(k)
  ids <- match(taxa, rownames(G))
  G_sub <- G[ids, ids]
  for (f in 1:k) {
    y <- blue
    y[fold_id == f] <- NA
    fit <- BGLR(y = y, ETA = list(mark = list(K = G_sub, model = "RKHS")),
                nIter = nIter, burnIn = burnIn, verbose = FALSE,
                saveAt = paste0("grouped_cv_", tag, "_fold", f, "_"))
    pred <- fit$yHat[fold_id == f]
    obs  <- blue[fold_id == f]
    accs[f] <- suppressWarnings(cor(pred, obs, use = "complete.obs"))
  }
  data.frame(fold = 1:k, r = accs)
}

## --- Step 3: run for each year -----------------------------------------------
years <- c("DS1", "DS2", "DS3")
grouped_results <- list()
for (yr in years) {
  grouped_results[[yr]] <- run_grouped_cv_one_year(
    taxa = pheno_wide$Taxa, blue = pheno_wide[[yr]], family = pheno_wide$family,
    G = G, tag = yr
  )
}

summary_tbl <- do.call(rbind, lapply(names(grouped_results), function(yr) {
  r <- grouped_results[[yr]]$r
  data.frame(year = yr, mean_r_grouped = mean(r), sd_r_grouped = sd(r))
}))
cat("\n=== Relatedness-grouped per-year CV accuracy ===\n")
print(summary_tbl)
write.csv(summary_tbl, sp("per_year_relatedness_grouped_CV.csv"), row.names = FALSE)

## --- Step 4: leakage delta -----------------------------------------------
## Paste in the ORIGINAL random-split per-year accuracies from Table S4
## (the numbers already in your manuscript) and compute
## delta = random - grouped, mirroring Table S6's format for the pooled
## analysis, so this can become a new "Table S6b" (per-year leakage delta).
original_random_split <- c(DS1 = NA, DS2 = NA, DS3 = NA)  # <-- fill in from Table S4
leakage_delta <- unname(original_random_split - summary_tbl$mean_r_grouped)
cat("\n=== Leakage delta (random - grouped) ===\n")
print(data.frame(year = years, leakage_delta = leakage_delta))
