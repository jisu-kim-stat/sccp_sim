# =========================================================
# R/sim.R
# - run_once(), run_many(), run_one_combo()
# - aggregation helpers (overall/clusterwise/classwise)
# =========================================================

# ----------------------------
# Helpers: collectors/aggregation
# ----------------------------

.pull_scalar_metrics <- function(res_list, tag) {
  purrr::map_dfr(res_list, function(r) {
    x <- tibble::as_tibble(r[[tag]])
    x %>%
      dplyr::select(overall_cov, mean_set_size, median_set_size,
                    cov_var_across_classes, worst_class_cov) %>%
      dplyr::slice(1)
  })
}

.pull_per_class_tbl_one_run <- function(run, method_key) {
  x <- run[[method_key]]
  if (is.null(x)) return(tibble::tibble())
  x <- tibble::as_tibble(x)

  if ("per_class" %in% names(x)) {
    pc <- tryCatch(x$per_class[[1]], error = function(e) NULL)
    if (is.null(pc)) return(tibble::tibble())
    tibble::as_tibble(pc) %>%
      dplyr::mutate(method = dplyr::case_when(
        method_key == "GCP"  ~ "Global",
        method_key == "CCCP" ~ "CC-CP",
        method_key == "SCCP" ~ "SCC-CP",
        TRUE ~ method_key
      ))
  } else if (all(c("class","cov") %in% names(x))) {
    x %>%
      dplyr::select(class, cov) %>%
      dplyr::mutate(method = dplyr::case_when(
        method_key == "GCP"  ~ "Global",
        method_key == "CCCP" ~ "CC-CP",
        method_key == "SCCP" ~ "SCC-CP",
        TRUE ~ method_key
      ))
  } else {
    tibble::tibble()
  }
}

.compute_clusterwise_cov_one_run <- function(run) {
  lc <- run$label_clusters
  if (is.null(lc)) return(tibble::tibble())
  methods <- c("GCP", "CCCP", "SCCP")

  per_class_all <- purrr::map_dfr(methods, ~ .pull_per_class_tbl_one_run(run, .x))
  if (nrow(per_class_all) == 0) return(tibble::tibble())

  map_tbl <- tibble::tibble(class = seq_along(lc), cluster = lc)

  per_class_all %>%
    dplyr::left_join(map_tbl, by = "class") %>%
    dplyr::group_by(method, cluster) %>%
    dplyr::summarise(cluster_cov = mean(cov, na.rm = TRUE), .groups = "drop")
}

# ----------------------------
# Filtering unseen classes 
# ----------------------------

filter_unseen_classes <- function(p_sel, p_cal, p_tst, y_sel, y_cal, y_te, verbose = TRUE) {
  valid_classes <- colnames(p_sel)

  keep_sel <- as.character(y_sel) %in% valid_classes
  keep_cal <- as.character(y_cal) %in% valid_classes
  keep_te  <- as.character(y_te)  %in% valid_classes

  if (verbose) {
    if (any(!keep_sel)) message("Dropping ", sum(!keep_sel), " selection samples with unseen classes.")
    if (any(!keep_cal)) message("Dropping ", sum(!keep_cal), " calibration samples with unseen classes.")
    if (any(!keep_te))  message("Dropping ", sum(!keep_te),  " test samples with unseen classes.")
  }

  y_sel2 <- y_sel[keep_sel]
  y_cal2 <- y_cal[keep_cal]
  y_te2  <- y_te[keep_te]

  p_sel2 <- p_sel[keep_sel, , drop = FALSE]
  p_cal2 <- p_cal[keep_cal, , drop = FALSE]
  p_tst2 <- p_tst[keep_te,  , drop = FALSE]

  colnames(p_sel2) <- colnames(p_cal2) <- colnames(p_tst2) <- valid_classes

  list(
    p_sel = p_sel2, p_cal = p_cal2, p_tst = p_tst2,
    y_sel = y_sel2, y_cal = y_cal2, y_te  = y_te2,
    valid_classes = valid_classes
  )
}

# ----------------------------
# One simulation run
# ----------------------------

run_once <- function(K = 50, d = 10, n_train = 4000, n_select = 2000,
                     n_calib = 2000, n_test = 4000,
                     prior = "zipf", noise = "hetero",
                     alpha = 0.05, Kc = 5, seed = 1) {

  set.seed(seed)

  Dtr  <- gen_data(n_train, K, d, prior, noise)
  Dsel <- gen_data(n_select, K, d, prior, noise)
  Dca  <- gen_data(n_calib, K, d, prior, noise)
  Dte  <- gen_data(n_test,  K, d, prior, noise)

  Xtr  <- as.data.frame(Dtr$X);  colnames(Xtr)  <- paste0("x", seq_len(ncol(Xtr)))
  Xsel <- as.data.frame(Dsel$X); colnames(Xsel) <- colnames(Xtr)
  Xca  <- as.data.frame(Dca$X);  colnames(Xca)  <- colnames(Xtr)
  Xte  <- as.data.frame(Dte$X);  colnames(Xte)  <- colnames(Xtr)

  y_tr  <- Dtr$y
  y_sel <- Dsel$y
  y_cal <- Dca$y
  y_te  <- Dte$y

  df_train <- data.frame(y = factor(y_tr), Xtr)
  mod <- nnet::multinom(y ~ ., data = df_train,
                        trace = FALSE, MaxNWts = 10000)

  p_sel <- as.matrix(predict(mod, newdata = Xsel, type = "probs"))
  p_cal <- as.matrix(predict(mod, newdata = Xca,  type = "probs"))
  p_tst <- as.matrix(predict(mod, newdata = Xte,  type = "probs"))

  # Filter unseen classes and align y/p
  filt <- filter_unseen_classes(p_sel, p_cal, p_tst, y_sel, y_cal, y_te, verbose = TRUE)

  p_sel <- filt$p_sel; p_cal <- filt$p_cal; p_tst <- filt$p_tst
  y_sel <- filt$y_sel; y_cal <- filt$y_cal; y_te  <- filt$y_te

  # Scores for selection/calibration (true-label scores)
  s_sel_true <- 1 - get_true_prob(p_sel, y_sel)
  s_cal_true <- 1 - get_true_prob(p_cal, y_cal)

  # ----- GCP -----
  out_gcp <- fit_GCP(p_cal, y_cal, p_tst, y_te, K, alpha)
  metrics_GCP   <- out_gcp$metrics
  classwise_GCP <- out_gcp$classwise
  qG_cal <- out_gcp$qG_cal

  # ----- CCCP -----
  out_cccp <- fit_CCCP(s_cal_true, y_cal, p_tst, y_te, K, alpha, Kc, m_min = 100)
  metrics_CCCP   <- out_cccp$metrics
  classwise_CCCP <- out_cccp$classwise
  label_clusters <- out_cccp$label_clusters
  qC_cal_safe    <- out_cccp$qC_cal

  # ----- SCCCP -----
  out_sccp <- fit_SCCCP(
    p_sel = p_sel, y_sel = y_sel, s_sel_true = s_sel_true,
    p_tst = p_tst, y_te = y_te, K = K, alpha = alpha,
    label_clusters = label_clusters,
    qC_cal_safe = qC_cal_safe, qG_cal = qG_cal,
    grid = seq(0, 1, by = 0.1)
  )
  metrics_SCCP   <- out_sccp$metrics
  classwise_SCCP <- out_sccp$classwise
  lambda_hat     <- out_sccp$lambda_hat
  q_star         <- out_sccp$q_star

  list(
    GCP   = c(metrics_GCP,  list(classwise = classwise_GCP)),
    CCCP  = c(metrics_CCCP, list(classwise = classwise_CCCP)),
    SCCP  = c(metrics_SCCP, list(classwise = classwise_SCCP)),
    lambda_hat = lambda_hat,
    qG_cal = qG_cal, qC_cal = qC_cal_safe, q_star = q_star,
    label_clusters = label_clusters
  )
}

# ----------------------------
# Multiple repetitions for one combo
# ----------------------------

run_many <- function(R = 10,
                     K = 50, d = 5,
                     n_train = 6000, n_select = 4000,
                     n_calib = 4000, n_test = 6000,
                     prior = "zipf", noise = "hetero",
                     alpha = 0.05, Kc = 6) {

  seeds <- sample(1:9999, R, replace = FALSE)
  res_list <- purrr::map(seeds, ~ run_once(
    K=K, d=d, n_train=n_train, n_select=n_select, n_calib=n_calib, n_test=n_test,
    prior=prior, noise=noise, alpha=alpha, Kc=Kc, seed=.x
  ))

  agg_scalar <- dplyr::bind_rows(
    .pull_scalar_metrics(res_list, "GCP")  %>% dplyr::mutate(method = "Global"),
    .pull_scalar_metrics(res_list, "CCCP") %>% dplyr::mutate(method = "CC-CP"),
    .pull_scalar_metrics(res_list, "SCCP") %>% dplyr::mutate(method = "SCC-CP")
  )

  summary_overall <- agg_scalar %>%
    dplyr::group_by(method) %>%
    dplyr::summarise(
      overall_cov = mean(overall_cov, na.rm = TRUE),
      mean_set_size = mean(mean_set_size, na.rm = TRUE),
      median_set_size = mean(median_set_size, na.rm = TRUE),
      cov_var_across_classes = mean(cov_var_across_classes, na.rm = TRUE),
      worst_class_cov = mean(worst_class_cov, na.rm = TRUE),
      .groups = "drop"
    )

  clusterwise_all <- purrr::map_dfr(seq_along(res_list), function(i) {
    .compute_clusterwise_cov_one_run(res_list[[i]]) %>%
      dplyr::mutate(rep = i)
  })

  summary_clusterwise <- clusterwise_all %>%
    dplyr::group_by(method, cluster) %>%
    dplyr::summarise(
      mean_cluster_cov = mean(cluster_cov, na.rm = TRUE),
      .groups = "drop"
    )

  classwise_all <- purrr::map_dfr(seq_along(res_list), function(i) {
    r <- res_list[[i]]
    dplyr::bind_rows(
      r$GCP$classwise  %>% dplyr::mutate(method = "Global"),
      r$CCCP$classwise %>% dplyr::mutate(method = "CC-CP"),
      r$SCCP$classwise %>% dplyr::mutate(method = "SCC-CP")
    ) %>% dplyr::mutate(rep = i)
  })

  summary_classwise <- classwise_all %>%
    dplyr::group_by(method, class) %>%
    dplyr::summarise(
      mean_class_cov  = mean(class_cov, na.rm = TRUE),
      mean_class_size = mean(class_size, na.rm = TRUE),
      .groups = "drop"
    )

  list(
    overall     = summary_overall,
    clusterwise = summary_clusterwise,
    classwise   = summary_classwise,
    runs        = res_list
  )
}

# ----------------------------
# One combo wrapper 
# ----------------------------

run_one_combo <- function(K, Kc, d, noise, prior, seed,
                          alpha = 0.05,
                          n_train = 4000, n_select = 2000, n_calib = 2000, n_test = 4000) {
  set.seed(seed)
  out <- run_many(
    R         = 1,
    K         = K,
    d         = d,
    n_train   = n_train,
    n_select  = n_select,
    n_calib   = n_calib,
    n_test    = n_test,
    prior     = prior,
    noise     = noise,
    alpha     = alpha,
    Kc        = Kc
  )

  out$overall     <- out$overall     %>% mutate(K = K, Kc = Kc, d = d, noise = noise, prior = prior, seed = seed)
  out$clusterwise <- out$clusterwise %>% mutate(K = K, Kc = Kc, d = d, noise = noise, prior = prior, seed = seed)
  out$classwise   <- out$classwise   %>% mutate(K = K, Kc = Kc, d = d, noise = noise, prior = prior, seed = seed)
  out
}
