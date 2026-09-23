## =============================================================================
## Prerequisite for Item 8 (R2#5): generate PER-SNP random forest importance
## for each year. Nothing currently saved in your Supplementary Tables has
## this (only chromosome-level sums, behind Section 2.5 / Figure 3) -- this
## script produces the per-SNP file that 08_rf_gwas_enrichment_test.R needs.
## =============================================================================
source("/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/00_config.R")
library(randomForest)

pheno_wide <- load_pheno_wide()
M          <- load_geno_matrix()
check_id_overlap(pheno_wide, M, context = "03_rf_importance")

## Match the manuscript's RF spec (Table S1, row "RandomForest"):
## n_estimators=300, max_depth=None, sqrt(p) features per split, bootstrap.
## randomForest() defaults to floor(p/3) for regression mtry, so mtry is set
## explicitly to sqrt(ncol(M)) to match what's documented in Table S1.
fit_rf_one_year <- function(taxa, blue, M, ntree = 300, seed = 1) {
  set.seed(seed)
  ids <- match(taxa, rownames(M))
  X <- M[ids, ]
  rf <- randomForest(x = X, y = blue, ntree = ntree,
                      mtry = max(1, floor(sqrt(ncol(X)))),
                      importance = TRUE)
  imp <- importance(rf, type = 1)   # %IncMSE (permutation importance)
  data.frame(SNP = rownames(imp), Importance = imp[, 1], row.names = NULL)
}

years <- c("DS1", "DS2", "DS3")
for (yr in years) {
  cat("Fitting RF for", yr, "...\n")
  rf_imp <- fit_rf_one_year(pheno_wide$Taxa, pheno_wide[[yr]], M)
  out_file <- sp(paste0("rf_importance_", yr, ".csv"))
  write.csv(rf_imp, out_file, row.names = FALSE)
  cat("  wrote", out_file, "(", nrow(rf_imp), "SNPs )\n")
}

## --- Chromosome/position join --------------------------------------------
## rf_importance_DS*.csv above has SNP + Importance only -- NO Chr/Pos_bp,
## because Genotype.Numerical.txt's column headers are just marker names
## (e.g. BS00022677_51), with no physical-position metadata attached.
##
## 08_rf_gwas_enrichment_test.R needs Chr and Pos_bp per SNP to test window
## overlap with the 113 GWAS MTAs. You need a marker position map covering
## all ~20,996 SNPs (not just the 113 MTA markers in mta_table_kaur2023.csv,
## which only has positions for those). This map should already exist
## somewhere in your pipeline, since Section 4.3 of the manuscript states
## "marker positions referenced to hexaploid wheat chromosomes 1A-7D [76]"
## for this exact marker set -- likely the same source used for the
## companion GWAS study's Table 3 (probably the WAMI panel's Illumina 90K
## consensus map from Sukumaran et al. 2016 / the CIMMYT data repository
## https://data.cimmyt.org/file.xhtml?persistentId=hdl:11529/10714/2 already
## cited in your Data Availability Statement).
##
## Once you have that map as a CSV with columns SNP, Chr, Pos_bp, join it in
## like this before running 08:
##
##   marker_map <- read.csv(sp("your_marker_position_map.csv"), stringsAsFactors = FALSE)
##   for (yr in c("DS1","DS2","DS3")) {
##     rf_imp <- read.csv(sp(paste0("rf_importance_", yr, ".csv")), stringsAsFactors = FALSE)
##     rf_imp <- merge(rf_imp, marker_map, by = "SNP")
##     write.csv(rf_imp, sp(paste0("rf_importance_", yr, ".csv")), row.names = FALSE)
##   }
##
## As a sanity check that marker naming is consistent between your genotype
## file and the GWAS paper's Table 3 (both use the Illumina 90K array's
## naming convention), this line confirms overlap:
mta <- read.csv(sp("mta_table_kaur2023.csv"), stringsAsFactors = FALSE)
overlap <- intersect(colnames(M), mta$Marker)
cat("\n", length(overlap), " of the 113 GWAS MTA markers are present by name in ",
    "Genotype.Numerical.txt (out of ", ncol(M), " total SNPs) -- confirms consistent ",
    "marker naming, useful for validating your full marker-position map once joined.\n", sep = "")
