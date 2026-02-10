## scripts/cifar_from_npz_minimal.R

library(reticulate)
library(dplyr)
library(purrr)

# ---- utilities ----
conformal_quantile <- function(scores, alpha) {
  scores <- sort(as.numeric(scores))
  n <- length(scores)
  k <- ceiling((n + 1) * (1 - alpha))
  scores[min(max(k, 1), n)]
}

build_pred_set_global <- function(p_row, q) {
  labels <- as.integer(names(p_row))
  s <- 1 - p_row
  labels[s <= q]
}

eval_metrics <- function(pred_sets, y_true, K) {
  covered <- map2_lgl(pred_sets, y_true, ~ .y %in% .x)
  tibble(
    overall_cov = mean(covered),
    mean_set_size = mean(lengths(pred_sets))
  )
}

# ---- main function ----
run_once_from_npz <- function(npz_path, alpha = 0.05) {
  np <- reticulate::import("numpy", delay_load = TRUE)
  z  <- np$load(npz_path, allow_pickle = TRUE)

  p_cal <- as.matrix(z[["p_cal"]])
  p_tst <- as.matrix(z[["p_tst"]])
  y_cal <- as.integer(z[["y_cal"]]) + 1L
  y_tst <- as.integer(z[["y_tst"]]) + 1L

  K <- ncol(p_cal)
  colnames(p_cal) <- colnames(p_tst) <- as.character(seq_len(K))

  get_true_prob <- function(p, y) {
    idx <- match(as.character(y), colnames(p))
    p[cbind(seq_len(nrow(p)), idx)]
  }

  s_cal <- 1 - get_true_prob(p_cal, y_cal)
  qG    <- conformal_quantile(s_cal, alpha)

  pred_G <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_global(pr, qG)
  })

  eval_metrics(pred_G, y_tst, K)
}



