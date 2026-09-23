## =============================================================================
## Item 1 (R1#4, R2#1): Re-run LOYO with year as a fixed effect,
## and estimate the cross-year genetic correlation.
## =============================================================================
source("/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/00_config.R")
library(BGLR)
library(sommer)   # for the multi-trait genetic-correlation model

pheno_wide <- load_pheno_wide()
M          <- load_geno_matrix()
check_id_overlap(pheno_wide, M, context = "01_loyo")
G          <- build_grm(M)
pheno_long <- pheno_wide_to_long(pheno_wide)

## =============================================================================
## PART A: LOYO with year fitted as a fixed effect
## =============================================================================
## Removes the average between-year mean difference (Table 1: DS1=5.3,
## DS2=5.1, DS3=5.3) from the genomic signal being fit, isolating whether the
## *within-year-adjusted* marker effects transfer better across years than
## the original (proxy-based) LOYO analysis suggested (DS1 r=0.014, DS2
## r=0.065, DS3 r=0.008; mean=0.009).

run_loyo_year_fixed <- function(pheno_long, G, held_out_year, nIter = 12000, burnIn = 2000) {
  train <- pheno_long[pheno_long$Year != held_out_year, ]
  test  <- pheno_long[pheno_long$Year == held_out_year, ]

  train$Year <- droplevels(train$Year)
  X_fixed <- model.matrix(~ Year, data = train)   # intercept + 1 year dummy

  train_ids <- match(train$Genotype, rownames(G))
  test_ids  <- match(test$Genotype,  rownames(G))

  y <- c(train$BLUE, rep(NA, nrow(test)))
  ids_all <- c(train_ids, test_ids)
  G_sub <- G[ids_all, ids_all]

  X_full <- rbind(X_fixed, matrix(0, nrow = nrow(test), ncol = ncol(X_fixed)))
  colnames(X_full) <- colnames(X_fixed)

  ETA <- list(
    year = list(X = X_full, model = "FIXED"),
    mark = list(K = G_sub, model = "RKHS")   # swap for GBLUP/BRR etc. as needed
  )

  fit <- BGLR(y = y, ETA = ETA, nIter = nIter, burnIn = burnIn,
              saveAt = paste0("loyo_yearfixed_", held_out_year, "_"), verbose = FALSE)

  pred_test <- fit$yHat[(nrow(train) + 1):length(y)]
  r <- suppressWarnings(cor(pred_test, test$BLUE, use = "complete.obs"))
  data.frame(held_out_year = held_out_year, r_year_adjusted = r, n_test = nrow(test))
}

results_year_fixed <- do.call(rbind, lapply(levels(pheno_long$Year), run_loyo_year_fixed,
                                             pheno_long = pheno_long, G = G))
cat("\n=== PART A: LOYO with year as a fixed effect ===\n")
print(results_year_fixed)
write.csv(results_year_fixed, sp("LOYO_year_fixed_effect_results.csv"), row.names = FALSE)

## =============================================================================
## PART B: Cross-year genetic correlation (the "positive control")
## =============================================================================
## data.txt is already wide (Taxa, DS1, DS2, DS3) -- fit a 3-trait genomic
## model directly, with an unstructured (us) genetic covariance across
## DS1/DS2/DS3. The off-diagonal, scaled to a correlation, IS the cross-year
## genetic correlation the reviewers asked for.

common_ids <- intersect(pheno_wide$Taxa, rownames(G))
if (length(common_ids) < nrow(pheno_wide)) {
  message(length(common_ids), " of ", nrow(pheno_wide),
          " genotypes in data.txt matched Genotype.Numerical.txt; proceeding with the intersection.")
}

mt_data <- pheno_wide[match(common_ids, pheno_wide$Taxa), ]
G_mt <- G[common_ids, common_ids]
colnames(mt_data)[colnames(mt_data) == "Taxa"] <- "id"

mt_fit <- mmer(
  cbind(DS1, DS2, DS3) ~ 1,
  random = ~ vsr(id, Gu = G_mt, Gtc = unsm(3)),
  rcov   = ~ vsr(units, Gtc = diag(3)),
  data   = mt_data,
  verbose = FALSE
)

Gcov <- mt_fit$sigma$`u:id`
Gcor <- cov2cor(Gcov)
cat("\n=== PART B: Cross-year GENETIC correlation matrix ===\n")
print(round(Gcor, 3))
write.csv(as.data.frame(Gcor), sp("cross_year_genetic_correlation.csv"))

## --- Interpretation for the Discussion paragraph ----------------------------
## Compare Gcor (genetic correlation, PART B) against the PHENOTYPIC
## correlation already reported in Section 2.2 (Figure 1A, r > 0.6 pairwise):
##   - Gcor high AND results_year_fixed / original LOYO r stays near zero
##       => near-zero LOYO reflects a prediction/estimation-power limitation
##          (few genotypes, weak per-marker signal), not a genuine absence of
##          shared genetic control across years.
##   - Gcor itself low despite high phenotypic correlation
##       => points to non-genetic (environmental/GxE) drivers of the
##          between-year phenotypic correlation instead.
##
## Note: DS1/DS2/DS3 in data.txt are integers (0-9) -- raw/rounded severity
## scores rather than a continuous BLUE. That's fine computationally, but
## worth squaring against Section 4.2's "year-specific BLUEs" wording before
## finalizing -- state explicitly whether data.txt IS the BLUE (rounded for
## storage) or the pre-BLUE raw score, if a separate BLUE step exists
## elsewhere in the pipeline.
