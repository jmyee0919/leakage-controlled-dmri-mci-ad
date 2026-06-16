# =========================
# Figure 5
# Integrated cross-dataset summary
# =========================

pkgs <- c("ggplot2", "dplyr", "readr", "patchwork", "scales")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = "https://cloud.r-project.org")
  }
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(patchwork)
  library(scales)
})

# -------------------------
# paths
# -------------------------
base_dir <- getwd()

syn_path  <- file.path(base_dir, "paper_results_synthetic.csv")
adni_path <- file.path(base_dir, "paper_results_adni.csv")

out_png <- file.path(base_dir, "Figure5_final.png")
out_pdf <- file.path(base_dir, "Figure5_final.pdf")

# -------------------------
# colors / theme
# -------------------------
col_main <- "#DC0000FF"   # red
col_grid <- "grey55"

theme_fig <- theme_classic(base_size = 13) +
  theme(
    plot.title = element_text(face = "bold", size = 15),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    strip.text = element_text(face = "bold", size = 12),
    strip.background = element_blank(),
    legend.position = "none",
    plot.margin = margin(10, 10, 10, 10)
  )

# -------------------------
# read data
# -------------------------
syn <- read_csv(syn_path, show_col_types = FALSE)
adni <- read_csv(adni_path, show_col_types = FALSE)

# -------------------------
# prepare synthetic summary
# -------------------------
syn_wide <- syn %>%
  select(
    window, comparison_set,
    roc_auc_mean, pr_ap_mean, brier_mean,
    delta_auc_mean,
    delong_p_range_low, delong_p_range_high
  ) %>%
  mutate(window = factor(window, levels = c("3-18", "6-24", "12-30")))

syn_snap <- syn_wide %>% filter(comparison_set == "Snapshot only")
syn_delta <- syn_wide %>% filter(comparison_set == "Snapshot+Change")

syn_gain <- syn_snap %>%
  select(window,
         snap_roc   = roc_auc_mean,
         snap_pr    = pr_ap_mean,
         snap_brier = brier_mean) %>%
  left_join(
    syn_delta %>%
      select(window,
             delta_roc   = roc_auc_mean,
             delta_pr    = pr_ap_mean,
             delta_brier = brier_mean,
             delta_auc_mean,
             delong_p_range_low,
             delong_p_range_high),
    by = "window"
  ) %>%
  mutate(
    source = "Synthetic",
    setting = as.character(window),
    roc_gain = delta_roc - snap_roc,
    pr_gain = delta_pr - snap_pr,
    brier_improve = snap_brier - delta_brier,
    p_worst = pmax(delong_p_range_low, delong_p_range_high, na.rm = TRUE),
    p_label = paste0("all p<", scientific(p_worst, digits = 2))
  ) %>%
  select(source, setting, roc_gain, pr_gain, brier_improve, delta_auc_mean, p_label)

# -------------------------
# prepare ADNI summary
# -------------------------
adni2 <- adni %>%
  mutate(
    setting = case_when(
      cohort == "all"      & calibration == "sigmoid"  ~ "All\nSigmoid",
      cohort == "filtered" & calibration == "sigmoid"  ~ "Filtered\nSigmoid",
      cohort == "filtered" & calibration == "isotonic" ~ "Filtered\nIsotonic",
      cohort == "all"      & calibration == "isotonic" ~ "All\nIsotonic",
      TRUE ~ paste(cohort, calibration)
    ),
    setting = factor(setting,
                     levels = c("All\nSigmoid",
                                "Filtered\nSigmoid",
                                "Filtered\nIsotonic",
                                "All\nIsotonic"))
  )

adni_snap <- adni2 %>% filter(comparison_set == "Snapshot only")
adni_delta <- adni2 %>% filter(comparison_set == "Snapshot+Change")

adni_gain <- adni_snap %>%
  select(setting,
         snap_roc   = roc_auc,
         snap_pr    = pr_ap,
         snap_brier = brier) %>%
  left_join(
    adni_delta %>%
      select(setting,
             delta_roc   = roc_auc,
             delta_pr    = pr_ap,
             delta_brier = brier,
             delta_auc_mean = delta_auc_vs_snapshot,
             delong_p_value),
    by = "setting"
  ) %>%
  mutate(
    source = "ADNI",
    roc_gain = delta_roc - snap_roc,
    pr_gain = delta_pr - snap_pr,
    brier_improve = snap_brier - delta_brier,
    p_label = paste0("p=", formatC(delong_p_value, format = "f", digits = 4))
  ) %>%
  select(source, setting, roc_gain, pr_gain, brier_improve, delta_auc_mean, p_label)

# combined
all_gain <- bind_rows(
  syn_gain %>% mutate(source = factor(source, levels = c("Synthetic", "ADNI"))),
  adni_gain %>% mutate(source = factor(source, levels = c("Synthetic", "ADNI")))
)

# -------------------------
# helper plot function
# -------------------------
make_gain_plot <- function(df, ycol, ylab, title_text) {
  ggplot(df, aes(x = setting, y = .data[[ycol]])) +
    geom_col(fill = col_main, width = 0.72) +
    geom_hline(yintercept = 0, linetype = "dashed", color = col_grid) +
    facet_wrap(~source, scales = "free_x", nrow = 1) +
    scale_y_continuous(labels = number_format(accuracy = 0.001),
                       expand = expansion(mult = c(0, 0.08))) +
    labs(title = title_text, x = NULL, y = ylab) +
    theme_fig +
    theme(axis.text.x = element_text(size = 10))
}

# -------------------------
# Panel A
# -------------------------
pA <- make_gain_plot(
  all_gain,
  ycol = "roc_gain",
  ylab = expression(Delta*"ROC AUC"),
  title_text = "ROC AUC gain"
)

# -------------------------
# Panel B
# -------------------------
pB <- make_gain_plot(
  all_gain,
  ycol = "pr_gain",
  ylab = expression(Delta*"PR-AUC"),
  title_text = "PR-AUC gain"
)

# -------------------------
# Panel C
# -------------------------
pC <- make_gain_plot(
  all_gain,
  ycol = "brier_improve",
  ylab = "Brier improvement",
  title_text = "Brier improvement"
)

# -------------------------
# Panel D
# -------------------------
pD <- ggplot(all_gain, aes(x = setting, y = delta_auc_mean)) +
  geom_point(size = 3, color = col_main) +
  geom_line(aes(group = 1), linewidth = 0.8, color = col_main) +
  geom_text(aes(label = p_label), vjust = -0.7, size = 3.7) +
  geom_hline(yintercept = 0, linetype = "dashed", color = col_grid) +
  facet_wrap(~source, scales = "free_x", nrow = 1) +
  scale_y_continuous(labels = number_format(accuracy = 0.001),
                     expand = expansion(mult = c(0, 0.14))) +
  labs(
    title = expression("Paired "*Delta*"AUC (DeLong)"),
    x = NULL,
    y = expression(Delta*"AUC")
  ) +
  theme_fig +
  theme(axis.text.x = element_text(size = 10))

# -------------------------
# combine (ABCD)
# -------------------------
fig5 <- (pA + pB) / (pC + pD) +
  plot_annotation(tag_levels = "A")

# -------------------------
# save
# -------------------------
ggsave(out_png, fig5, width = 15, height = 10.5, dpi = 320, bg = "white")
ggsave(out_pdf, fig5, width = 15, height = 10.5, bg = "white")

cat("Saved:\n")
cat(" - ", out_png, "\n", sep = "")
cat(" - ", out_pdf, "\n", sep = "")