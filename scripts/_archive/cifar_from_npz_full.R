suppressPackageStartupMessages({
  library(reticulate)
  library(dplyr)
  library(purrr)
  library(tibble)
})

# ----------------------------
# Utilities
# ----------------------------
conformal_quantile <- function(scores, alpha) {
  scores <- sort(as.numeric(scores))
  n <- length(scores)
  if (n == 0) return(Inf)
  k <- ceiling((n + 1) * (1 - alpha))
  k <- max(min(k, n), 1)
  scores[k]
}

eval_metrics <- function(pred_sets, y_true, K) {
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  tibble(
    overall_cov = mean(covered),
    mean_set_size = mean(lengths(pred_sets)),
    median_set_size = median(lengths(pred_sets))
  )
}

# ----------------------------
# Tail / classwise
# ----------------------------
get_tail_classes <- function(y_ref, K, tail_frac = 0.2) {
  y_ref <- as.integer(y_ref)
  tab <- tabulate(y_ref, nbins = K)
  cls <- seq_len(K)
  ord <- order(tab, decreasing = FALSE)
  m <- max(1, ceiling(K * tail_frac))
  tail_cls <- cls[ord[1:m]]
  list(tail_cls = tail_cls, freq = tab)
}

eval_metrics_tail <- function(pred_sets, y_true, tail_cls) {
  y_true <- as.integer(y_true)
  is_tail <- y_true %in% tail_cls
  if (!any(is_tail)) {
    return(tibble(
      overall_cov_tail = NA_real_,
      mean_set_size_tail = NA_real_,
      median_set_size_tail = NA_real_,
      n_tail = 0L
    ))
  }
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  ss <- lengths(pred_sets)
  tibble(
    overall_cov_tail = mean(covered[is_tail]),
    mean_set_size_tail = mean(ss[is_tail]),
    median_set_size_tail = median(ss[is_tail]),
    n_tail = sum(is_tail)
  )
}

eval_classwise_cov <- function(pred_sets, y_true, K) {
  y_true <- as.integer(y_true)
  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))

  cov_by_class <- rep(NA_real_, K)
  n_by_class <- tabulate(y_true, nbins = K)

  for (k in seq_len(K)) {
    idx <- which(y_true == k)
    if (length(idx) > 0) cov_by_class[k] <- mean(covered[idx])
  }

  tibble(
    y = seq_len(K),
    n = as.integer(n_by_class),
    cov = as.numeric(cov_by_class)
  )
}

summarize_classwise <- function(df_classwise, tail_cls) {
  df_tail <- df_classwise %>% filter(.data$y %in% tail_cls, .data$n > 0)
  df_all  <- df_classwise %>% filter(.data$n > 0)

  tibble(
    worst_class_cov = if (nrow(df_all) > 0) min(df_all$cov, na.rm = TRUE) else NA_real_,
    var_class_cov   = if (nrow(df_all) > 1) var(df_all$cov, na.rm = TRUE) else NA_real_,
    tail_mean_cov   = if (nrow(df_tail) > 0) mean(df_tail$cov, na.rm = TRUE) else NA_real_,
    tail_worst_cov  = if (nrow(df_tail) > 0) min(df_tail$cov, na.rm = TRUE) else NA_real_
  )
}

# ----------------------------
# Clusterwise metrics
# ----------------------------
eval_clusterwise_metrics <- function(pred_sets, y_true, label_clusters, Kc_eff = NULL) {
  y_true <- as.integer(y_true)
  cl_true <- label_clusters[y_true]
  if (is.null(Kc_eff)) Kc_eff <- max(label_clusters)

  covered <- map2_lgl(pred_sets, y_true, ~ (.y %in% .x))
  ss <- lengths(pred_sets)

  cov_by_cluster <- rep(NA_real_, Kc_eff)
  size_by_cluster <- rep(NA_real_, Kc_eff)
  n_by_cluster <- tabulate(cl_true, nbins = Kc_eff)

  for (cc in seq_len(Kc_eff)) {
    idx <- which(cl_true == cc)
    if (length(idx) > 0) {
      cov_by_cluster[cc]  <- mean(covered[idx])
      size_by_cluster[cc] <- mean(ss[idx])
    }
  }

  tibble(
    cluster = seq_len(Kc_eff),
    n = as.integer(n_by_cluster),
    cov = as.numeric(cov_by_cluster),
    mean_set_size = as.numeric(size_by_cluster)
  )
}

summarize_clusterwise <- function(df_clusterwise) {
  df_ok <- df_clusterwise %>% filter(.data$n > 0)
  tibble(
    worst_cluster_cov = if (nrow(df_ok) > 0) min(df_ok$cov, na.rm = TRUE) else NA_real_,
    var_cluster_cov   = if (nrow(df_ok) > 1) var(df_ok$cov, na.rm = TRUE) else NA_real_,
    mean_cluster_size = if (nrow(df_ok) > 0) mean(df_ok$mean_set_size, na.rm = TRUE) else NA_real_,
    max_cluster_size  = if (nrow(df_ok) > 0) max(df_ok$mean_set_size, na.rm = TRUE) else NA_real_
  )
}

# ----------------------------
# Prediction set constructors
# ----------------------------
build_pred_set_global <- function(p_row, qG) {
  labels <- as.integer(names(p_row))
  s_vec  <- 1 - p_row
  labels[s_vec <= qG]
}

build_pred_set_cluster <- function(p_row, label_clusters, q_by_cluster) {
  labels <- as.integer(names(p_row))
  s_vec  <- 1 - p_row
  cl_ids <- label_clusters[labels]
  th_vec <- q_by_cluster[cl_ids]
  labels[s_vec <= th_vec]
}

# ============================================================
# CCCP-style clustering: quantile embedding + null + weighted k-means
# ============================================================
min_count_for_null <- function(alpha) {
  thr <- (1 / min(alpha, 0.1)) - 1
  floor(thr)
}

weighted_kmeans <- function(X, centers, weights, iter_max = 50, nstart = 5, seed = 1) {
  set.seed(seed)
  X <- as.matrix(X)
  n <- nrow(X); d <- ncol(X)
  w <- as.numeric(weights)
  w <- pmax(w, 0)

  best_tot <- Inf
  best_cluster <- NULL

  w_center <- function(Xc, wc) {
    if (length(wc) == 0 || sum(wc) <= 0) return(colMeans(Xc))
    colSums(Xc * wc) / sum(wc)
  }

  for (s in seq_len(nstart)) {
    if (sum(w) > 0) {
      idx <- sample.int(n, size = centers, replace = FALSE, prob = w / sum(w))
    } else {
      idx <- sample.int(n, size = centers, replace = FALSE)
    }
    mu <- X[idx, , drop = FALSE]
    cl <- rep(1L, n)

    for (it in seq_len(iter_max)) {
      d2 <- sapply(seq_len(centers), function(k) rowSums((X - matrix(mu[k, ], n, d, byrow = TRUE))^2))
      new_cl <- max.col(-d2)
      if (all(new_cl == cl)) break
      cl <- new_cl

      for (k in seq_len(centers)) {
        ik <- which(cl == k)
        if (length(ik) == 0) {
          if (sum(w) > 0) {
            ridx <- sample.int(n, size = 1, prob = w / sum(w))
          } else {
            ridx <- sample.int(n, size = 1)
          }
          mu[k, ] <- X[ridx, ]
        } else {
          mu[k, ] <- w_center(X[ik, , drop = FALSE], w[ik])
        }
      }
    }

    tot <- 0
    for (k in seq_len(centers)) {
      ik <- which(cl == k)
      if (length(ik) > 0) {
        dif <- X[ik, , drop = FALSE] - matrix(mu[k, ], length(ik), d, byrow = TRUE)
        tot <- tot + sum(w[ik] * rowSums(dif^2))
      }
    }

    if (tot < best_tot) {
      best_tot <- tot
      best_cluster <- cl
    }
  }

  list(cluster = best_cluster, tot_withinss = best_tot)
}

label_kmeans_cccp_style <- function(scores_by_label,
                                    Kc = 10,
                                    alpha = 0.05,
                                    quantiles = c(.5, .6, .7, .8, .9, 1 - alpha),
                                    seed = 1) {

  quantiles <- sort(unique(quantiles))
  K <- length(scores_by_label)

  n_y <- sapply(seq_len(K), function(k) {
    v <- scores_by_label[[as.character(k)]]
    if (is.null(v)) 0L else length(v)
  })

  n_min <- min_count_for_null(alpha)
  is_null <- n_y < n_min

  all_scores <- unlist(scores_by_label, use.names = FALSE)
  global_q <- if (length(all_scores) == 0) rep(0.5, length(quantiles)) else
    as.numeric(quantile(all_scores, probs = quantiles, na.rm = TRUE, type = 8))

  emb <- matrix(NA_real_, nrow = K, ncol = length(quantiles))
  for (k in seq_len(K)) {
    v <- scores_by_label[[as.character(k)]]
    if (is.null(v) || length(v) == 0) {
      emb[k, ] <- global_q
    } else if (!is_null[k]) {
      emb[k, ] <- as.numeric(quantile(v, probs = quantiles, na.rm = TRUE, type = 8))
    } else {
      emb[k, ] <- global_q
    }
  }

  idx_nonnull <- which(!is_null)
  if (length(idx_nonnull) == 0) {
    label_clusters <- rep(1L, K)
    return(list(label_clusters = label_clusters, Kc_eff = 1L, is_null = is_null, n_min = n_min))
  }

  X <- scale(emb[idx_nonnull, , drop = FALSE])
  w <- sqrt(pmax(n_y[idx_nonnull], 0))

  Kc_eff_nonnull <- min(Kc, length(idx_nonnull))
  km <- weighted_kmeans(X, centers = Kc_eff_nonnull, weights = w, iter_max = 50, nstart = 5, seed = seed)

  # non-null: 1..Kc_eff_nonnull, null: Kc_eff_nonnull+1
  label_clusters <- rep(Kc_eff_nonnull + 1L, K)
  label_clusters[idx_nonnull] <- km$cluster
  Kc_eff <- Kc_eff_nonnull + 1L

  list(label_clusters = label_clusters, Kc_eff = Kc_eff, is_null = is_null, n_min = n_min)
}

# ============================================================
# NEW: tau-shrinkage for cluster thresholds
# ============================================================
shrink_cluster_thresholds <- function(qC_cal, qG_cal, n_cluster_cal, tau) {
  # w_c = n_c/(n_c+tau)
  w <- n_cluster_cal / (n_cluster_cal + tau)
  w[!is.finite(w)] <- 0
  as.numeric(w * qC_cal + (1 - w) * qG_cal)
}

# ----------------------------
# Main: Run GCP / CCCP / SCCP(tau) from NPZ
# ----------------------------
run_cifar_from_npz <- function(npz_path,
                               alpha = 0.05,
                               Kc = 10,
                               tau = 50,
                               tail_frac = 0.2,
                               seed = 1,
                               quantiles = c(.5, .6, .7, .8, .9, 1 - alpha)) {

  np <- reticulate::import("numpy", delay_load = TRUE)
  z  <- np$load(npz_path, allow_pickle = TRUE)

  # NOTE: sel split은 이제 필요 없지만, 파일에 있으면 읽어도 무방
  p_cal <- as.matrix(z[["p_cal"]])
  p_tst <- as.matrix(z[["p_tst"]])

  y_cal <- as.integer(z[["y_cal"]]) + 1L
  y_tst <- as.integer(z[["y_tst"]]) + 1L

  K <- ncol(p_cal)
  colnames(p_cal) <- colnames(p_tst) <- as.character(seq_len(K))

  get_true_prob <- function(p_mat, y) {
    idx <- match(as.character(y), colnames(p_mat))
    p_mat[cbind(seq_len(nrow(p_mat)), idx)]
  }

  s_cal_true <- 1 - get_true_prob(p_cal, y_cal)

  tail_info <- get_tail_classes(y_ref = y_cal, K = K, tail_frac = tail_frac)
  tail_cls <- tail_info$tail_cls

  # 1) clustering (from calibration scores)
  scores_by_label_cal <- split(s_cal_true, y_cal)
  scores_by_label_cal <- scores_by_label_cal[as.character(seq_len(K))]
  for (k in seq_len(K)) if (is.null(scores_by_label_cal[[k]])) scores_by_label_cal[[k]] <- numeric(0)

  clu_obj <- label_kmeans_cccp_style(
    scores_by_label = scores_by_label_cal,
    Kc = Kc,
    alpha = alpha,
    quantiles = quantiles,
    seed = seed
  )
  label_clusters <- clu_obj$label_clusters
  Kc_eff <- clu_obj$Kc_eff
  null_id <- Kc_eff

  # 2) global threshold
  qG_cal <- conformal_quantile(s_cal_true, alpha)

  # 3) cluster thresholds on calibration + null forced to global
  cl_cal <- label_clusters[y_cal]
  n_cluster_cal <- tabulate(cl_cal, nbins = Kc_eff)

  qC_cal <- sapply(seq_len(Kc_eff), function(cc) {
    if (cc == null_id) return(qG_cal)
    idx <- which(cl_cal == cc)
    if (length(idx) == 0) return(qG_cal)
    conformal_quantile(s_cal_true[idx], alpha)
  })

  # 4) SCCP(tau): shrink cluster thresholds toward global using n_cluster_cal
  q_star <- qC_cal
  q_star[seq_len(Kc_eff)] <- shrink_cluster_thresholds(qC_cal, qG_cal, n_cluster_cal, tau = tau)
  q_star[null_id] <- qG_cal  # null은 무조건 global

  # ------------------------------------------------------------
  # Prediction sets + metrics
  # ------------------------------------------------------------

  # ---- GCP
  pred_GCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_global(pr, qG_cal)
  })

  met_GCP_overall <- eval_metrics(pred_GCP, y_tst, K)
  met_GCP_tail    <- eval_metrics_tail(pred_GCP, y_tst, tail_cls)
  cw_GCP  <- eval_classwise_cov(pred_GCP, y_tst, K)
  sum_GCP <- summarize_classwise(cw_GCP, tail_cls)
  clw_GCP   <- eval_clusterwise_metrics(pred_GCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_GCP <- summarize_clusterwise(clw_GCP)

  # ---- CCCP (cluster thresholds; null is global)
  pred_CCCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, qC_cal)
  })

  met_CCCP_overall <- eval_metrics(pred_CCCP, y_tst, K)
  met_CCCP_tail    <- eval_metrics_tail(pred_CCCP, y_tst, tail_cls)
  cw_CCCP  <- eval_classwise_cov(pred_CCCP, y_tst, K)
  sum_CCCP <- summarize_classwise(cw_CCCP, tail_cls)
  clw_CCCP   <- eval_clusterwise_metrics(pred_CCCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_CCCP <- summarize_clusterwise(clw_CCCP)

  # ---- SCCP(tau)
  pred_SCCP <- lapply(seq_len(nrow(p_tst)), function(i) {
    pr <- p_tst[i, ]; names(pr) <- colnames(p_tst)
    build_pred_set_cluster(pr, label_clusters, q_star)
  })

  met_SCCP_overall <- eval_metrics(pred_SCCP, y_tst, K)
  met_SCCP_tail    <- eval_metrics_tail(pred_SCCP, y_tst, tail_cls)
  cw_SCCP  <- eval_classwise_cov(pred_SCCP, y_tst, K)
  sum_SCCP <- summarize_classwise(cw_SCCP, tail_cls)
  clw_SCCP   <- eval_clusterwise_metrics(pred_SCCP, y_tst, label_clusters, Kc_eff = Kc_eff)
  clsum_SCCP <- summarize_clusterwise(clw_SCCP)

  # ---- Output tables
  overall_tbl <- bind_rows(
    met_GCP_overall  %>% mutate(method = "GCP"),
    met_CCCP_overall %>% mutate(method = "CCCP"),
    met_SCCP_overall %>% mutate(method = "SCCP_tau")
  )

  tail_tbl <- bind_rows(
    met_GCP_tail  %>% mutate(method = "GCP"),
    met_CCCP_tail %>% mutate(method = "CCCP"),
    met_SCCP_tail %>% mutate(method = "SCCP_tau")
  )

  classwise_summary_tbl <- bind_rows(
    sum_GCP  %>% mutate(method = "GCP"),
    sum_CCCP %>% mutate(method = "CCCP"),
    sum_SCCP %>% mutate(method = "SCCP_tau")
  )

  clusterwise_summary_tbl <- bind_rows(
    clsum_GCP  %>% mutate(method = "GCP"),
    clsum_CCCP %>% mutate(method = "CCCP"),
    clsum_SCCP %>% mutate(method = "SCCP_tau")
  )

  list(
    overall = overall_tbl,
    tail = tail_tbl,
    classwise_summary = classwise_summary_tbl,
    clusterwise_summary = clusterwise_summary_tbl,
    classwise = list(GCP = cw_GCP, CCCP = cw_CCCP, SCCP_tau = cw_SCCP),
    clusterwise = list(GCP = clw_GCP, CCCP = clw_CCCP, SCCP_tau = clw_SCCP),
    tail_info = list(tail_frac = tail_frac, tail_classes = tail_cls, freq_cal = tail_info$freq),
    label_clusters = label_clusters,
    Kc_eff = Kc_eff,
    qG_cal = qG_cal,
    qC_cal = qC_cal,
    q_star = q_star,
    n_cluster_cal = n_cluster_cal,
    tau = tau
  )
}
