# 00_smoke_test.R
# Confirms BGLR is installed and working on your machine before running
# the full pipeline. Source this file's 01_bglr_fold.R first, or run this
# script directly with Rscript.

source("01_bglr_fold.R")

set.seed(1)
X <- matrix(rnorm(50 * 200), 50, 200)
y <- X[, 1:5] %*% rnorm(5) + rnorm(50)

cat("Running a tiny BGLR fit (should take well under a minute)...\n")
out <- run_bglr_fold(X[1:40, ], y[1:40], X[41:50, ], y[41:50],
                      model_name = "BayesA", nIter = 3000, burnIn = 500)

print(out)
cat("\nCorrelation (y_true vs y_pred):", cor(out$y_true, out$y_pred), "\n")
cat("If that printed a real number, BGLR is working correctly.\n")
