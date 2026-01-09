# =========================================================
# scripts/02_plot_all.R
# - Read out/*.rds
# - Plot figures + make tables only
# =========================================================

source("R/methods.R")  # packages + ensure_dirs + theme helpers if needed

ensure_dirs(c("fig/overall", "fig/clusterwise", "fig/lambda", "table"))

final_overall     <- readRDS("out/final_overall.rds")
final_clusterwise <- readRDS("out/final_clusterwise.rds")
final_classwise   <- readRDS("out/final_classwise.rds")
lambda_df         <- readRDS("out/lambda_df.rds")

method_levels  <- c("Global","CC-CP","SCC-CP")

theme_paper <- function(base_size = 11){
  theme_bw(base_size = base_size) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      strip.background = element_rect(fill = "white", colour = "black", linewidth = 0.5),
      strip.text = element_text(face = "bold"),
      legend.position = "top",
      legend.title = element_blank(),
      legend.key = element_blank(),
      panel.spacing = unit(6, "pt")
    )
}

col_vals   <- c("Global"="black","CC-CP"="grey35","SCC-CP"="grey10")
lty_vals   <- c("Global"="solid","CC-CP"="dashed","SCC-CP"="dotdash")
shape_vals <- c("Global"=16, "CC-CP"=1, "SCC-CP"=17)

y_scale_cov <- scale_y_continuous(
  labels = percent_format(accuracy = 1),
  limits = c(0.8, 1.0),
  breaks = seq(0.8, 1.0, by = 0.05)
)

Ks_to_plot <- sort(unique(final_overall$K))
d_vals     <- sort(unique(final_overall$d))
noise_vals <- sort(unique(final_overall$noise))

# --------------------------------------------------
# 1) Overall plot
# --------------------------------------------------
for (noise0 in noise_vals) {
  for (d0 in d_vals) {
    for (K0 in Ks_to_plot) {

      df_overall <- final_overall %>%
        filter(prior == "zipf", #balanced or zipf
               noise == noise0,
               d == d0,
               K == K0)

      if (nrow(df_overall) == 0) next

      plot_overall_dK <- df_overall %>%
        mutate(
          method   = factor(method, levels = method_levels),
          Kc_label = factor(paste0("Kc=", Kc),
                            levels = c("Kc=3","Kc=5","Kc=7","Kc=10"))
        ) %>%
        group_by(method, Kc_label) %>%
        summarise(mean_cov = mean(overall_cov, na.rm = TRUE),
                  .groups  = "drop") %>%
        ggplot(aes(x = method, y = mean_cov,
                   colour = method, shape = method)) +
        geom_point(size = 2.3, fill = "white", stroke = 0.8) +
        geom_hline(yintercept = 0.95, linetype = "dotted", linewidth = 0.6) +
        facet_wrap(~ Kc_label, nrow = 1) +
        y_scale_cov +
        scale_colour_manual(values = col_vals) +
        scale_shape_manual(values  = shape_vals) +
        labs(x = "Method", y = "Coverage") +
        theme_paper()

      ggsave(
        filename = sprintf("fig/overall/overall_zipf_%s_d%d_K%d.png",
                           noise0, d0, K0),
        plot     = plot_overall_dK,
        width    = 7, height = 3.5, dpi = 300, bg = "white"
      )
    }
  }
}

# --------------------------------------------------
# 2) Clusterwise plot 
# --------------------------------------------------
for (noise0 in noise_vals) {
  for (d0 in d_vals) {
    for (K0 in Ks_to_plot) {

      df_cluster <- final_clusterwise %>%
        filter(prior == "zipf",
               noise == noise0,
               d == d0,
               K == K0)

      if (nrow(df_cluster) == 0) next

      plot_cluster_dK <- df_cluster %>%
        mutate(
          method   = factor(method, levels = method_levels),
          cluster  = as.factor(cluster),
          Kc_label = factor(paste0("Kc=", Kc),
                            levels = c("Kc=3","Kc=5","Kc=7","Kc=10"))
        ) %>%
        group_by(method, Kc_label, cluster) %>%
        summarise(mean_cov = mean(mean_cluster_cov, na.rm = TRUE),
                  .groups = "drop") %>%
        ggplot(aes(x = cluster, y = mean_cov,
                   group = method, colour = method,
                   linetype = method)) +
        geom_line(linewidth = 0.7) +
        geom_point(aes(shape = method), size = 1.8, stroke = 0.8) +
        geom_hline(yintercept = 0.95, linetype = "dotted", linewidth = 0.6) +
        facet_wrap(~ Kc_label, nrow = 1) +
        scale_colour_manual(values = col_vals) +
        scale_linetype_manual(values= lty_vals) +
        scale_shape_manual(values= shape_vals) +
        y_scale_cov +
        labs(x = "Cluster", y = "Coverage") +
        theme_paper()

      ggsave(
        filename = sprintf("fig/clusterwise/cluster_zipf_%s_d%d_K%d.png",
                           noise0, d0, K0),
        plot     = plot_cluster_dK,
        width    = 7, height = 3.5, dpi = 300, bg = "white"
      )
    }
  }
}

# --------------------------------------------------
# 3) Coverage Report table (zipf + hetero)
# --------------------------------------------------
coverage_report <- final_classwise %>%
  filter(prior == "zipf", noise == "hetero") %>%
  group_by(d, K, Kc, method) %>%
  summarise(
    prop_over95 = mean(mean_class_cov >= 0.95, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(prop_percent = percent(prop_over95, accuracy = 0.01)) %>%
  select(K, Kc, method, prop_percent)

coverage_table <- coverage_report %>%
  mutate(method = factor(method, levels = c("Global", "CC-CP", "SCC-CP"))) %>%
  arrange(K, Kc, method) %>%
  pivot_wider(names_from = method, values_from = prop_percent) %>%
  select(K, Kc, `Global`, `CC-CP`, `SCC-CP`)

xt <- xtable(
  coverage_table,
  caption = "Fraction of classes whose coverage exceeded the 0.95 target.",
  label   = "tab:over95-zipf-hetero",
  digits  = 3
)

print(
  xt,
  file = "table/1125_classwise_zipf_hetero_pivot.tex",
  include.rownames = FALSE,
  booktabs = TRUE,
  sanitize.text.function = identity
)

# --------------------------------------------------
# 4) Lambda histogram (all)
# --------------------------------------------------
p_lambda_hist <- ggplot(lambda_df, aes(x = lambda)) +
  geom_histogram(
    binwidth = 0.1,
    boundary = 0,
    closed   = "left",
    color    = "black",
    fill     = "grey30"
  ) +
  scale_x_continuous(
    limits = c(0, 1),
    breaks = seq(0, 1, by = 0.2)
  ) +
  labs(x = expression(lambda), y = "Count") +
  theme_bw(base_size = 14) +
  theme(
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank()
  )

ggsave(
  filename = "fig/lambda/lambda_hist_zipf_hetero.png",
  plot     = p_lambda_hist,
  width    = 5.5,
  height   = 4.2,
  dpi      = 300
)
