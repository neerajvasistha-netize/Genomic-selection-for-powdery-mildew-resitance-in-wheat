## =============================================================================
## 00_config.R -- shared paths and data-loading helpers.
## Every other script in this set does source("00_config.R") first, so the
## file paths and column names only need to be correct in ONE place.
## =============================================================================

PHENO_PATH <- "/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/data.txt"
GENO_PATH  <- "/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis/Genotype.Numerical.txt"

## Folder where these .R scripts and mta_table_kaur2023.csv live, and where
## every script's CSV output (rf_importance_DS*.csv, LOYO results, etc.)
## gets written. This is used instead of relying on R's current working
## directory (getwd()), which silently changes depending on how you launch a
## script (RStudio "Source", Rscript from Terminal, an interactive console
## session, etc.) and was the cause of the "cannot open file
## mta_table_kaur2023.csv" error. EDIT THIS if you saved the scripts
## somewhere other than the data folder:
SCRIPTS_DIR <- "/Users/neerajvasistha/Desktop/Destop 15 Sep_2026/Neeraj_Papers/raman_data/Final/Today_analysis"

## Convenience wrapper: every script below reads/writes its small CSVs via
## sp(filename) instead of a bare filename, so it always resolves against
## SCRIPTS_DIR no matter what the working directory is when you hit Source/Run.
sp <- function(filename) file.path(SCRIPTS_DIR, filename)

## Confirmed file formats (via 00_inspect_data_structure.R output):
##   data.txt                : WIDE, columns Taxa, DS1, DS2, DS3 (286 rows,
##                              one per genotype; DS1/DS2/DS3 integer 0-9)
##   Genotype.Numerical.txt  : first column "taxa" (lowercase), then ~20,996
##                              SNP columns (marker names e.g. BS00022677_51),
##                              values 0/1/2

load_pheno_wide <- function(path = PHENO_PATH) {
  pheno_wide <- read.table(path, header = TRUE, stringsAsFactors = FALSE)
  stopifnot(all(c("Taxa", "DS1", "DS2", "DS3") %in% colnames(pheno_wide)))
  pheno_wide
}

load_geno_matrix <- function(path = GENO_PATH) {
  M <- as.matrix(read.table(path, header = TRUE, row.names = 1))
  storage.mode(M) <- "numeric"
  M
}

build_grm <- function(M) {
  ## VanRaden-style GRM, same convention used elsewhere in the study.
  tcrossprod(scale(M, center = TRUE, scale = TRUE)) / ncol(M)
}

pheno_wide_to_long <- function(pheno_wide) {
  long <- do.call(rbind, lapply(c("DS1", "DS2", "DS3"), function(yr) {
    data.frame(Genotype = pheno_wide$Taxa, Year = yr, BLUE = pheno_wide[[yr]],
               stringsAsFactors = FALSE)
  }))
  long$Year <- factor(long$Year, levels = c("DS1", "DS2", "DS3"))
  long
}

check_id_overlap <- function(pheno_wide, M, context = "") {
  missing_in_geno <- setdiff(pheno_wide$Taxa, rownames(M))
  if (length(missing_in_geno) > 0) {
    stop(sprintf(
      "[%s] %d genotype ID(s) in data.txt have no match in Genotype.Numerical.txt (e.g. %s). Check for case/whitespace differences before proceeding.",
      context, length(missing_in_geno), paste(head(missing_in_geno, 5), collapse = ", ")
    ))
  }
  invisible(TRUE)
}
