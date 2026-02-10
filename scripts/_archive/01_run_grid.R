# =========================================================
# scripts/01_run_grid.R
# - Run grid simulations
# - Save results to out/*.rds
# =========================================================

source("R/methods.R")
source("R/sim.R")

ensure_dirs(c("out", "fig", "table"))

# ---- Grid  ----
grid <- tidyr::expand_grid(
  K    = c(30, 50),
  Kc   = c(3, 5, 7, 10),
  d    = 10,
  prior = c("balanced","zipf"),
  noise = c("homog", "hetero")
) %>%
  filter(!(K == 30 & Kc == 10))

n_reps <- 50
seeds  <- 1:n_reps
alpha  <- 0.05

total_tasks <- nrow(grid) * n_reps
pb <- progress_bar$new(
  total = total_tasks,
  format = "[:bar] :percent ETA: :eta (K=:K Kc=:Kc d=:d noise=:noise prior=:prior seed=:seed)"
)

all_runs <- list()
for (i in seq_len(nrow(grid))) {
  g <- grid[i, ]
  for (s in seeds) {
    pb$tick(tokens = list(
      K    = g$K,
      Kc   = g$Kc,
      d    = g$d,
      noise = g$noise,
      prior = g$prior,
      seed  = s
    ))

    result <- run_one_combo(
      K     = g$K,
      Kc    = g$Kc,
      d     = g$d,
      noise = g$noise,
      prior = g$prior,
      seed  = s,
      alpha = alpha,
      n_train = 4000, n_select = 2000, n_calib = 2000, n_test = 4000
    )
    all_runs <- append(all_runs, list(result))
  }
}

# ---- Final aggregation (overall / clusterwise / classwise) ----
final_overall <- all_runs %>%
  purrr::map(~ .x$overall) %>%
  purrr::compact() %>%
  list_rbind()

final_clusterwise <- all_runs %>%
  purrr::map(~ .x$clusterwise) %>%
  purrr::compact() %>%
  list_rbind()

final_classwise <- all_runs %>%
  purrr::map(~ .x$classwise) %>%
  purrr::compact() %>%
  list_rbind()

cat("# rows\n")
cat("final_overall    :", nrow(final_overall), "\n")
cat("final_clusterwise:", nrow(final_clusterwise), "\n")
cat("final_classwise  :", nrow(final_classwise), "\n")

# ---- Lambda df (for histogram) ----
lambda_df <- purrr::map_dfr(all_runs, function(res_combo) {
  purrr::map_dfr(res_combo$runs, function(run) {
    tibble(
      K        = run$K,
      Kc       = length(unique(run$label_clusters)),
      cluster  = seq_along(run$lambda_hat),
      lambda   = run$lambda_hat
    )
  })
}) %>%
  filter(!is.na(lambda))

# ---- Save to out/ ----
saveRDS(final_overall,     file = "out/final_overall.rds")
saveRDS(final_clusterwise, file = "out/final_clusterwise.rds")
saveRDS(final_classwise,   file = "out/final_classwise.rds")
saveRDS(lambda_df,         file = "out/lambda_df.rds")

# (Optional) raw all_runs 저장은 용량이 큼
# saveRDS(all_runs, file = "out/all_runs.rds")
