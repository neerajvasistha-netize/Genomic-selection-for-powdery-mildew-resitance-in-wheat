## =============================================================================
## Item 8 (R2#5): Marker-level enrichment test -- are the top RF-importance
## SNPs disproportionately located within the confidence intervals of the
## 113 GWAS-significant MTAs from Kaur et al. 2023?
## =============================================================================
## Prerequisite: run 03_generate_rf_importance.R first, then join a
## marker-position map onto its output (see that script's comments) so each
## rf_importance_DS*.csv has SNP, Chr, Pos_bp, Importance -- NOT just SNP and
## Importance, since Genotype.Numerical.txt carries no position metadata.
## =============================================================================

source("/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/00_config.R")

mta <- read.csv(sp("mta_table_kaur2023.csv"), stringsAsFactors = FALSE)

## 200 kb window used in the companion GWAS paper's candidate-gene search
## (100 kb each side of the MTA) -- reuse the same window here for consistency.
window_bp <- 100000
mta$start <- mta$Pos_bp - window_bp
mta$end   <- mta$Pos_bp + window_bp

## --- enrichment test for one year -------------------------------------------
enrichment_test <- function(rf_importance, mta, top_frac = 0.01, n_perm = 10000, seed = 1) {
  stopifnot(all(c("SNP", "Chr", "Pos_bp", "Importance") %in% colnames(rf_importance)))
  set.seed(seed)
  n_top <- ceiling(top_frac * nrow(rf_importance))
  top_snps <- rf_importance[order(-rf_importance$Importance), ][1:n_top, ]

  in_mta_window <- function(snps_df, mta) {
    hits <- vapply(seq_len(nrow(snps_df)), function(i) {
      chr_match <- mta[mta$Chr == snps_df$Chr[i], ]
      if (nrow(chr_match) == 0) return(FALSE)
      any(snps_df$Pos_bp[i] >= chr_match$start & snps_df$Pos_bp[i] <= chr_match$end)
    }, logical(1))
    sum(hits)
  }

  observed_hits <- in_mta_window(top_snps, mta)
  perm_hits <- replicate(n_perm, {
    rand_snps <- rf_importance[sample(nrow(rf_importance), n_top), ]
    in_mta_window(rand_snps, mta)
  })

  p_value <- (sum(perm_hits >= observed_hits) + 1) / (n_perm + 1)
  fold_enrichment <- if (mean(perm_hits) > 0) observed_hits / mean(perm_hits) else NA

  list(n_top_snps = n_top, observed_hits = observed_hits,
       expected_hits = mean(perm_hits), fold_enrichment = fold_enrichment, p_value = p_value)
}

## --- run for each year -------------------------------------------------------
years <- c("DS1", "DS2", "DS3")
results <- list()
for (yr in years) {
  f <- sp(paste0("rf_importance_", yr, ".csv"))
  if (!file.exists(f)) {
    message("Skipping ", yr, ": ", f, " not found -- run 03_generate_rf_importance.R first.")
    next
  }
  rf_imp <- read.csv(f, stringsAsFactors = FALSE)
  if (!all(c("Chr", "Pos_bp") %in% colnames(rf_imp))) {
    message("Skipping ", yr, ": ", f, " has no Chr/Pos_bp columns yet -- join your marker-position ",
            "map onto it first (see 03_generate_rf_importance.R's comments).")
    next
  }
  mta_yr <- mta[mta$Year == yr, ]   # test against that year's own MTAs
  results[[yr]] <- enrichment_test(rf_imp, mta_yr)
  cat("\n==", yr, "==\n")
  print(results[[yr]])
}

if (length(results) == 0) {
  cat("\nNo years had usable rf_importance_DS*.csv files with Chr/Pos_bp -- ",
      "nothing to report yet. See the messages above.\n")
}

## Also worth running once pooled across all three years' RF importance
## against the full 113-MTA table, since that is the comparison closest to
## the chromosome-level one already in the Discussion paragraph this test is
## meant to formalize.

## Interpretation for the Discussion paragraph: report observed vs. expected
## hits, fold-enrichment, and the permutation p-value for each year (and
## pooled). This resolves whether DS2's chromosome-level overlap (4A, 6B)
## reflects genuine marker-level convergence with the GWAS loci, or is fully
## explained by DS2 simply having far more MTAs (85 of 113) and therefore
## more genomic territory for RF-important SNPs to land in by chance -- which
## the permutation null controls for automatically.
