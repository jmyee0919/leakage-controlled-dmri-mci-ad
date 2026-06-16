# =========================================================
# Figure 2 FINAL (minimal figure-only style)
# Uses ONLY:
#   - master_summary.csv
#   - master_delong.csv
#
# Put this script in the SAME folder as those two csv files.
# Run:
#   Rscript figure2_master_summary_minimal.R
# =========================================================

pkgs <- c("ggplot2", "dplyr", "readr", "patchwork", "tibble", "stringr", "scales")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) {
  install.packages(to_install, repos = "https://cloud.r-project.org")
}

library(ggplot2)
library(dplyr)
library(readr)
library(patchwork)
library(tibble)
library(stringr)
library(scales)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  getwd()
}

base_dir <- get_script_dir()
cat("Using base_dir:", base_dir, "\n")

summary_file <- file.path(base_dir, "master_summary.csv")
delong_file  <- file.path(base_dir, "master_delong.csv")

if (!file.exists(summary_file)) stop("Could not find master_summary.csv in the same folder as this script.")
if (!file.exists(delong_file)) stop("Could not find master_delong.csv in the same folder as this script.")

out_pdf <- file.path(base_dir, "Figure2_synthetic_robustness_minimal.pdf")
out_png <- file.path(base_dir, "Figure2_synthetic_robustness_minimal.png")

col_base <- "#00A087FF"
col_full <- "#DC0000FF"

model_colors <- c(
  "Snapshot only" = col_base,
  "Snapshot + DeltaMRI" = col_full
)

standardize_model_name <- function(x) {
  x_low <- tolower(as.character(x))
  if (str_detect(x_low, "snapshot") && !str_detect(x_low, "change|delta|\\+|full|all")) return("Snapshot only")
  if (str_detect(x_low, "change|delta|\\+|full|all")) return("Snapshot + DeltaMRI")
  as.character(x)
}

theme_fig <- function() {
  theme_classic(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10, color = "black"),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      plot.tag = element_text(face = "bold", size = 18)
    )
}

sum_df <- read_csv(summary_file, show_col_types = FALSE)
del_df <- read_csv(delong_file, show_col_types = FALSE)

names(sum_df) <- tolower(names(sum_df))
names(del_df) <- tolower(names(del_df))

set_col <- if ("set" %in% names(sum_df)) "set" else if ("model" %in% names(sum_df)) "model" else names(sum_df)[1]

sum_df <- sum_df %>%
  mutate(
    set_clean = sapply(.data[[set_col]], standardize_model_name),
    roc_auc = as.numeric(roc_auc),
    pr_ap = as.numeric(pr_ap),
    brier = as.numeric(brier)
  )

window_levels <- c("3-18", "6-24", "12-30")

sum_win <- sum_df %>%
  group_by(window, set_clean) %>%
  summarise(
    roc_mean = mean(roc_auc, na.rm = TRUE),
    roc_sd   = sd(roc_auc, na.rm = TRUE),
    pr_mean  = mean(pr_ap, na.rm = TRUE),
    pr_sd    = sd(pr_ap, na.rm = TRUE),
    brier_mean = mean(brier, na.rm = TRUE),
    brier_sd   = sd(brier, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(
    window = factor(window, levels = window_levels),
    set_clean = factor(set_clean, levels = c("Snapshot only", "Snapshot + DeltaMRI"))
  )

delta_col <- if ("deltaauc" %in% names(del_df)) "deltaauc" else if ("delta_auc" %in% names(del_df)) "delta_auc" else NA
p_col <- if ("p_value" %in% names(del_df)) "p_value" else if ("p" %in% names(del_df)) "p" else NA

del_win <- del_df %>%
  mutate(
    delta_auc = as.numeric(.data[[delta_col]]),
    p_value_num = as.numeric(.data[[p_col]])
  ) %>%
  group_by(window) %>%
  summarise(
    dauc_mean = mean(delta_auc, na.rm = TRUE),
    dauc_sd   = sd(delta_auc, na.rm = TRUE),
    p_max     = max(p_value_num, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  mutate(window = factor(window, levels = window_levels))

common_pos <- position_dodge(width = 0.18)

p_roc <- ggplot(sum_win, aes(x = window, y = roc_mean, color = set_clean, group = set_clean)) +
  geom_line(linewidth = 1.1, position = common_pos) +
  geom_point(size = 3.2, position = common_pos) +
  geom_errorbar(aes(ymin = roc_mean - roc_sd, ymax = roc_mean + roc_sd),
                width = 0.07, linewidth = 1.0, position = common_pos) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(limits = c(0.84, 1.01), labels = label_number(accuracy = 0.01)) +
  labs(title = "ROC AUC across windows", x = "Delta-window (months)", y = "Mean ROC AUC +/- SD", color = NULL) +
  theme_fig()

p_pr <- ggplot(sum_win, aes(x = window, y = pr_mean, color = set_clean, group = set_clean)) +
  geom_line(linewidth = 1.1, position = common_pos) +
  geom_point(size = 3.2, position = common_pos) +
  geom_errorbar(aes(ymin = pr_mean - pr_sd, ymax = pr_mean + pr_sd),
                width = 0.07, linewidth = 1.0, position = common_pos) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(limits = c(0.74, 1.01), labels = label_number(accuracy = 0.01)) +
  labs(title = "PR-AUC across windows", x = "Delta-window (months)", y = "Mean PR-AUC +/- SD", color = NULL) +
  theme_fig()

p_brier <- ggplot(sum_win, aes(x = window, y = brier_mean, color = set_clean, group = set_clean)) +
  geom_line(linewidth = 1.1, position = common_pos) +
  geom_point(size = 3.2, position = common_pos) +
  geom_errorbar(aes(ymin = pmax(0, brier_mean - brier_sd), ymax = brier_mean + brier_sd),
                width = 0.07, linewidth = 1.0, position = common_pos) +
  scale_color_manual(values = model_colors) +
  scale_y_continuous(limits = c(0.02, 0.16), labels = label_number(accuracy = 0.01)) +
  labs(title = "Brier score across windows", x = "Delta-window (months)", y = "Mean Brier score +/- SD", color = NULL) +
  theme_fig()

label_df <- del_win %>%
  mutate(
    label = paste0("p = ", format(p_max, scientific = TRUE, digits = 2)),
    y_lab = dauc_mean + dauc_sd + c(0.005, 0.004, 0.004)
  )

p_dauc <- ggplot(del_win, aes(x = window, y = dauc_mean, group = 1)) +
  geom_line(color = col_full, linewidth = 1.1) +
  geom_point(color = col_full, size = 3.4) +
  geom_errorbar(aes(ymin = dauc_mean - dauc_sd, ymax = dauc_mean + dauc_sd),
                width = 0.07, linewidth = 1.0, color = col_full) +
  geom_text(data = label_df, aes(x = window, y = y_lab, label = label), size = 3.7, color = "black") +
  scale_y_continuous(limits = c(0.06, 0.14), labels = label_number(accuracy = 0.01)) +
  labs(title = "dAUC across windows", x = "Delta-window (months)", y = "Mean dAUC +/- SD") +
  theme_fig()

fig2 <- (p_roc + p_pr) / (p_brier + p_dauc) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

ggsave(out_pdf, fig2, width = 14, height = 10)
ggsave(out_png, fig2, width = 14, height = 10, dpi = 600)

cat("Saved:\n")
cat(" - ", out_pdf, "\n", sep = "")
cat(" - ", out_png, "\n", sep = "")

print(fig2)
