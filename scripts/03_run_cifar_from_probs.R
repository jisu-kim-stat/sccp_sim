source("R/methods.R")
source("R/sim.R")

library(reticulate)
# ----------------------------
# Load data from npz
## ----------------------------
load_cifar_npz <- function(npz_path){
  suppressPackageStartupMessages(library(reticulate))
  np <- reticulate::import("numpy", delay_load = TRUE)
  z  <- np$load(npz_path, allow_pickle = TRUE)
  
  p_sel <- z[["p_sel"]]
  y_sel <- z[["y_sel"]]
  p_cal <- z[["p_cal"]]
  y_cal <- z[["y_cal"]]
  p_tst <- z[["p_tst"]]
  y_tst <- z[["y_tst"]]
  
  # convert
  p_sel <- as.matrix(p_sel)
  p_cal <- as.matrix(p_cal)
  p_tst <- as.matrix(p_tst)
  
  y_sel <- as.integer(y_sel) + 1L
  y_cal <- as.integer(y_cal) + 1L
  y_tst <- as.integer(y_tst) + 1L
  
  K <- ncol(p_sel)
  stopifnot(ncol(p_cal) == K, ncol(p_tst) == K)
  
  # IMPORTANT: give class labels 1..K as colnames (your build_pred_set_* uses names)
  colnames(p_sel) <- as.character(seq_len(K))
  colnames(p_cal) <- as.character(seq_len(K))
  colnames(p_tst) <- as.character(seq_len(K))
  
  list(
    p_sel = p_sel, y_sel = y_sel,
    p_cal = p_cal, y_cal = y_cal,
    p_tst = p_tst, y_tst = y_tst,
    K = K
  )
}

 ----------------------------
# Run from npz file
## ----------------------------

run_once_from_npz <- function(npz_path, alpha = 0.05, Kc = 10){
  library(reticulate)
  np <- import("numpy")

  z <- np$load(npz_path, allow_pickle = TRUE)

  p_sel <- as.matrix(z[["p_sel"]])
  p_cal <- as.matrix(z[["p_cal"]])
  p_tst <- as.matrix(z[["p_tst"]])

  y_sel <- as.integer(z[["y_sel"]]) + 1L
  y_cal <- as.integer(z[["y_cal"]]) + 1L
  y_te  <- as.integer(z[["y_tst"]]) + 1L

  K <- ncol(p_sel)
  colnames(p_sel) <- colnames(p_cal) <- colnames(p_tst) <- as.character(seq_len(K))

  get_true_prob <- function(p_mat, y){
    idx <- match(as.character(y), colnames(p_mat))
    p_mat[cbind(seq_len(nrow(p_mat)), idx)]
  }

  s_sel <- 1 - get_true_prob(p_sel, y_sel)
  s_cal <- 1 - get_true_prob(p_cal, y_cal)

  qG_cal <- conformal_quantile(s_cal, alpha)

  pred_G <- lapply(seq_len(nrow(p_tst)), function(i){
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_global(pr, qG_cal)
  })
  met_G <- eval_metrics(pred_G, y_te, K)

  list(GCP = met_G)
}
