## =============================================================================
## Item 5 (R2#3): Diagnose the PCA variance-explained discrepancy.
## This study reports PC1=10.3%, PC2=5.2%; Kaur et al. 2023 reports
## PC1=43.3%, PC2=18.8%, for what is described as the same 286 genotypes and
## the same 21,132-SNP curated marker set.
## =============================================================================
## Leading explanation (already reflected in Section 4.3/4.4 of the
## tracked-changes manuscript): the companion GWAS study [13] retained
## HapMap-format genotype calls, while this study converted the same curated
## marker set to a numeric (0/1/2) allele-dosage matrix for the GP model
## implementations. That HapMap-to-numeric conversion commonly applies its
## own default missing-data handling and allele-coding convention. Before
## running the three PCA versions below, first pin down WHICH tool did the
## conversion, since its defaults explain the numbers more directly than
## guessing at scaling alone:
##   - GAPIT (GAPIT.HapMap2numeric): major allele = 0, minor = 2,
##     heterozygotes = 1; missing calls imputed to the SNP's mean dosage.
##   - TASSEL (Numericalization plugin): similar minor/major dosage coding;
##     missing handling depends on the imputation plugin used beforehand.
##   - PLINK (--recodeA): reference-allele dosage (not necessarily minor
##     allele); missing calls stay NA unless imputed separately.
##   - A custom script: whatever convention you wrote into it.
## =============================================================================
source("/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/00_config.R")

M <- load_geno_matrix()

## Version A: centered only, NOT scaled (unit-variance) -- what scikit-learn's
## PCA() does by default if fed raw 0/1/2 dosages.
pca_unscaled <- prcomp(M, center = TRUE, scale. = FALSE)
var_unscaled <- summary(pca_unscaled)$importance[2, 1:2] * 100

## Version B: centered AND scaled to unit variance per SNP -- closer to what
## GWAS software (TASSEL/GAPIT/rMVP) typically does for population-structure
## PCA, and to sklearn's PCA() if StandardScaler was applied first.
pca_scaled <- prcomp(M, center = TRUE, scale. = TRUE)
var_scaled <- summary(pca_scaled)$importance[2, 1:2] * 100

## Version C: allele-frequency-weighted (VanRaden-style) scaling -- what a
## GRM-based PCA effectively uses; another common GWAS-pipeline convention.
p <- colMeans(M) / 2
M_freq_scaled <- sweep(M, 2, 2 * p, "-")
M_freq_scaled <- sweep(M_freq_scaled, 2, sqrt(2 * p * (1 - p)), "/")
M_freq_scaled[!is.finite(M_freq_scaled)] <- 0   # guard against monomorphic SNPs
pca_freqscaled <- prcomp(M_freq_scaled, center = FALSE, scale. = FALSE)
var_freqscaled <- summary(pca_freqscaled)$importance[2, 1:2] * 100

cat("Unscaled (center only):     PC1 =", round(var_unscaled[1], 1),
    "%  PC2 =", round(var_unscaled[2], 1), "%\n")
cat("Unit-variance scaled:       PC1 =", round(var_scaled[1], 1),
    "%  PC2 =", round(var_scaled[2], 1), "%\n")
cat("Allele-frequency scaled:    PC1 =", round(var_freqscaled[1], 1),
    "%  PC2 =", round(var_freqscaled[2], 1), "%\n")
cat("\nThis study reported:        PC1 = 10.3 %  PC2 = 5.2 %\n")
cat("Kaur et al. 2023 reported:  PC1 = 43.3 %  PC2 = 18.8 %\n")

## Whichever of the three above lands closest to 10.3/5.2 is almost certainly
## what this study's PCA script actually did; whichever lands closest to
## 43.3/18.8 is almost certainly closest to what the companion GWAS paper's
## PCA did. Once confirmed, replace the hedged sentence in Section 4.4 of the
## tracked-changes manuscript with the specific, confirmed explanation.
