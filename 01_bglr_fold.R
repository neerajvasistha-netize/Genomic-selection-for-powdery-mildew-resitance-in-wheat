#!/usr/bin/env Rscript
# bglr_fold.R
#
# Two ways to use this file:
#
#   (1) IN RSTUDIO: click Source (or Ctrl/Cmd+Shift+S) to load the function
#       run_bglr_fold() into your environment, then call it directly from
#       the console -- see the worked example at the bottom of this file
#       inside the `if (FALSE) { ... }` block (select those lines and
#       run them, or copy them into the console).
#
#   (2) FROM THE COMMAND LINE / the Python driver: this file is also a
#       runnable script --
#         Rscript bglr_fold.R X_train.csv y_train.csv X_test.csv y_test.csv \
#                              model_name nIter burnIn out_pred.csv
#       genuine_30model_benchmark.py calls it exactly this way via
#       subprocess, once per (model x fold). Nothing to change for that
#       path -- the CLI block at the bottom only runs when the file is
#       executed with Rscript, not when you Source it in RStudio.
#
# Fits ONE genuine BGLR model on ONE train/test fold.

suppressMessages(library(BGLR))

#' Fit a genuine BGLR model on one train/test split and return predictions
#'
#' @param X_train matrix, n_train x n_markers (already numeric/standardized)
#' @param y_train numeric vector, length n_train
#' @param X_test  matrix, n_test x n_markers
#' @param y_test  numeric vector, length n_test (used only for the returned
#'                data.frame's y_true column -- not seen by BGLR)
#' @param model_name one of "BayesA","BayesB","BayesC","BayesCpi","BRR","BL"
#'                (BayesCpi maps onto BGLR's "BayesC" -- see note below)
#' @param nIter   MCMC iterations (12000 for final numbers, ~3000 for a
#'                quick check)
#' @param burnIn  MCMC burn-in (2000 for final numbers, ~500 for a quick
#'                check)
#' @return data.frame with columns y_true, y_pred (one row per test point)
run_bglr_fold <- function(X_train, y_train, X_test, y_test,
                           model_name, nIter = 12000, burnIn = 2000) {

  X_train <- as.matrix(X_train)
  X_test  <- as.matrix(X_test)
  y_train <- as.numeric(y_train)
  y_test  <- as.numeric(y_test)

  # BGLR requires the response for ALL individuals (train+test), with test
  # phenotypes set to NA -- it predicts NA'd individuals from the training
  # data in a single joint model. This is the standard BGLR GS workflow.
  y_all <- c(y_train, rep(NA, length(y_test)))
  X_all <- rbind(X_train, X_test)
  test_idx <- (length(y_train) + 1):nrow(X_all)

  # Map our 7 Bayesian-alphabet names onto BGLR's actual supported model
  # strings. BayesCpi = BGLR's "BayesC" (pi is estimated, not fixed).
  bglr_model <- switch(model_name,
    "BayesA"   = "BayesA",
    "BayesB"   = "BayesB",
    "BayesC"   = "BayesC",
    "BayesCpi" = "BayesC",
    "BRR"      = "BRR",
    "BL"       = "BL",
    stop(paste("Unsupported model_name:", model_name,
               "-- BayesR has no native BGLR implementation; do not",
               "silently substitute another model for it (see the",
               "note at the bottom of this file)."))
  )

  ETA <- list(list(X = X_all, model = bglr_model))

  # tmp working directory for BGLR's internal files, cleaned up after
  wd <- tempfile("bglr_")
  dir.create(wd)
  old_wd <- getwd()
  setwd(wd)
  on.exit({ setwd(old_wd); unlink(wd, recursive = TRUE) }, add = TRUE)

  fm <- BGLR(y = y_all, ETA = ETA, nIter = nIter, burnIn = burnIn,
             verbose = FALSE, saveAt = "fold_")

  pred_test <- fm$yHat[test_idx]
  data.frame(y_true = y_test, y_pred = pred_test)
}

# --------------------------------------------------------------------------
# RSTUDIO USAGE: after Sourcing this file, call run_bglr_fold() directly,
# or use the standalone smoke_test_bglr.R and run_ds2_bglr.R scripts
# (just Source those -- no manual line selection needed).
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# CLI ENTRY POINT -- only runs when this file is executed with Rscript
# (i.e. NOT when Source()'d in RStudio). This is what
# genuine_30model_benchmark.py calls via subprocess.
# --------------------------------------------------------------------------
if (!interactive()) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) == 8) {
    X_train <- read.csv(args[1], header = FALSE)
    y_train <- read.csv(args[2], header = FALSE)[, 1]
    X_test  <- read.csv(args[3], header = FALSE)
    y_test  <- read.csv(args[4], header = FALSE)[, 1]
    model_name <- args[5]
    nIter   <- as.integer(args[6])
    burnIn  <- as.integer(args[7])
    out_path <- args[8]

    out <- run_bglr_fold(X_train, y_train, X_test, y_test, model_name, nIter, burnIn)
    write.csv(out, out_path, row.names = FALSE)
  }
}

# --- Note on BayesR ---
# BGLR has no "BayesR" model type. Genuine BayesR (Erbe et al. 2012, a
# 4-component mixture of marker-effect variance classes) requires a
# different tool -- e.g. the standalone BayesR software, or the `qgg` R
# package's `gbayes(method = "bayesR")`. If you need a real BayesR number,
# run one of those and merge its output in on the Python/R side; do not
# relabel a BRR/BayesC run as "BayesR" -- that reproduces exactly the
# naming problem the editor flagged in the first place.
