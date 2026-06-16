# Figure3_LOCKED_FINAL.R
# ADNI Figure 3 final:
# - Curves are drawn from adni_filtered_primary_predictions.csv
# - Annotation numbers are LOCKED to paper_results_adni.csv
#   using dataset=ADNI, cohort=filtered, calibration=sigmoid,
#   comparison_set = Snapshot only / Snapshot+Change

req_pkgs <- c('readr','dplyr','ggplot2','scales','cowplot')
for (p in req_pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) {
    install.packages(p, repos = 'https://cloud.r-project.org')
  }
}

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(scales)
  library(cowplot)
})

find_existing <- function(candidates) {
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) return(NA_character_)
  hit[[1]]
}

base_dir <- getwd()
cat('Using base_dir: ', base_dir, '\n', sep = '')

pred_file <- find_existing(c(
  file.path(base_dir, 'adni_filtered_primary_predictions.csv'),
  file.path(base_dir, 'filtered_primary_predictions.csv'),
  file.path(base_dir, 'predictions_adni_filtered_primary.csv')
))
if (is.na(pred_file)) stop('Prediction CSV not found in this folder.')

results_file <- find_existing(c(
  file.path(base_dir, 'paper_results_adni.csv'),
  file.path(base_dir, 'adni_results_primary.csv')
))
if (is.na(results_file)) stop('paper_results_adni.csv not found in this folder.')

pred <- read_csv(pred_file, show_col_types = FALSE)
names(pred) <- tolower(names(pred))

truth_col <- intersect(c('y_true','label','outcome','y','target'), names(pred))
base_col  <- intersect(c('p_snapshot','snapshot_prob','prob_snapshot_only','pred_snapshot','snapshot_only'), names(pred))
full_col  <- intersect(c('p_snapshot_change','p_delta','delta_prob','prob_snapshot_change','pred_snapshot_delta','snapshot_change'), names(pred))
subj_col  <- intersect(c('subjectid','subject_id','rid','id'), names(pred))

if (length(truth_col) == 0 || length(base_col) == 0 || length(full_col) == 0) {
  stop('Prediction CSV must contain truth + snapshot + snapshot-change probability columns.')
}

truth_col <- truth_col[[1]]
base_col  <- base_col[[1]]
full_col  <- full_col[[1]]
subj_col  <- if (length(subj_col) == 0) NA_character_ else subj_col[[1]]

pred <- pred %>%
  transmute(
    subject = if (is.na(subj_col)) row_number() else as.character(.data[[subj_col]]),
    y = as.integer(.data[[truth_col]]),
    p_base = as.numeric(.data[[base_col]]),
    p_full = as.numeric(.data[[full_col]])
  ) %>%
  filter(!is.na(y), !is.na(p_base), !is.na(p_full))

if (any(duplicated(pred$subject))) {
  pred <- pred %>%
    group_by(subject) %>%
    summarise(
      y = max(y, na.rm = TRUE),
      p_base = mean(p_base, na.rm = TRUE),
      p_full = mean(p_full, na.rm = TRUE),
      .groups = 'drop'
    )
  cat('Collapsed duplicated subject rows to subject-level predictions: n = ', nrow(pred), '\n', sep = '')
}

res <- read_csv(results_file, show_col_types = FALSE)
names(res) <- tolower(names(res))

res_primary <- res %>%
  filter(
    tolower(dataset) == 'adni',
    tolower(cohort) == 'filtered',
    tolower(calibration) == 'sigmoid'
  )

row_base <- res_primary %>% filter(tolower(comparison_set) == 'snapshot only')
row_full <- res_primary %>% filter(tolower(comparison_set) == 'snapshot+change')

if (nrow(row_base) != 1 || nrow(row_full) != 1) {
  stop('Could not uniquely identify filtered/sigmoid Snapshot only and Snapshot+Change rows in paper_results_adni.csv')
}

fixed_auc_base <- row_base$roc_auc[[1]]
fixed_auc_full <- row_full$roc_auc[[1]]
fixed_ap_base  <- row_base$pr_ap[[1]]
fixed_ap_full  <- row_full$pr_ap[[1]]
fixed_dauc     <- row_full$delta_auc_vs_snapshot[[1]]
fixed_p        <- row_full$delong_p_value[[1]]
fixed_z        <- row_full$delong_z[[1]]
fixed_n        <- row_full$n_test[[1]]

cat('LOCKED paper values:\n')
cat('  ROC AUC: ', fixed_auc_base, ' vs ', fixed_auc_full, '\n', sep = '')
cat('  PR AP:   ', fixed_ap_base, ' vs ', fixed_ap_full, '\n', sep = '')
cat('  dAUC:    ', fixed_dauc, ', p = ', fixed_p, ', z = ', fixed_z, ', n = ', fixed_n, '\n', sep = '')

roc_curve_df <- function(y, p) {
  o <- order(-p)
  y <- y[o]
  P <- sum(y == 1)
  N <- sum(y == 0)
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  tibble(fpr = c(0, fp / N, 1), tpr = c(0, tp / P, 1))
}

pr_curve_df <- function(y, p) {
  o <- order(-p)
  y <- y[o]
  tp <- cumsum(y == 1)
  fp <- cumsum(y == 0)
  precision <- tp / pmax(tp + fp, 1)
  recall <- tp / sum(y == 1)
  tibble(recall = c(0, recall, 1), precision = c(1, precision, precision[length(precision)]))
}

calibration_df <- function(y, p, bins = 10) {
  p <- pmin(pmax(p, 1e-8), 1 - 1e-8)
  brks <- quantile(p, probs = seq(0, 1, length.out = bins + 1), na.rm = TRUE)
  brks <- unique(brks)
  grp <- cut(p, breaks = brks, include.lowest = TRUE, labels = FALSE)
  tibble(y = y, p = p, grp = grp) %>%
    group_by(grp) %>%
    summarise(mean_pred = mean(p), obs_rate = mean(y), .groups = 'drop') %>%
    filter(!is.na(grp))
}

net_benefit_df <- function(y, p, thresholds = seq(0.03, 0.30, by = 0.005)) {
  n <- length(y)
  prevalence <- mean(y)
  tibble(threshold = thresholds) %>%
    rowwise() %>%
    mutate(
      tp = sum(p >= threshold & y == 1),
      fp = sum(p >= threshold & y == 0),
      nb_model = tp / n - fp / n * (threshold / (1 - threshold)),
      nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold)),
      nb_none = 0
    ) %>%
    ungroup()
}

fmt3 <- function(x) formatC(x, format = 'f', digits = 3)
fmt4 <- function(x) formatC(x, format = 'f', digits = 4)
fmt_p <- function(x) {
  if (is.na(x)) return('NA')
  if (x < 1e-3) return(format(x, scientific = TRUE, digits = 3))
  formatC(x, format = 'f', digits = 6)
}

roc_base <- roc_curve_df(pred$y, pred$p_base)
roc_full <- roc_curve_df(pred$y, pred$p_full)
pr_base  <- pr_curve_df(pred$y, pred$p_base)
pr_full  <- pr_curve_df(pred$y, pred$p_full)
cal_base <- calibration_df(pred$y, pred$p_base, bins = 10)
cal_full <- calibration_df(pred$y, pred$p_full, bins = 10)
nb_base  <- net_benefit_df(pred$y, pred$p_base)
nb_full  <- net_benefit_df(pred$y, pred$p_full)

col_base <- '#0AA087FF'
col_full <- '#DC0000FF'
col_none <- '#7F7F7F'
col_all  <- '#000000'

fig_theme <- theme_classic(base_size = 12) +
  theme(
    plot.title = element_text(face = 'bold', size = 15, hjust = 0),
    axis.title = element_text(face = 'bold'),
    axis.text = element_text(color = 'black'),
    legend.title = element_blank(),
    legend.position = 'bottom',
    legend.direction = 'horizontal',
    legend.box = 'horizontal',
    legend.text = element_text(size = 11),
    plot.margin = margin(8, 8, 8, 8),
    plot.tag = element_text(face = 'bold', size = 16)
  )

p1 <- ggplot() +
  geom_line(data = roc_base, aes(fpr, tpr, color = 'Snapshot only'), linewidth = 1.2) +
  geom_line(data = roc_full, aes(fpr, tpr, color = 'Snapshot + DeltaMRI'), linewidth = 1.2) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = col_none, linewidth = 0.8) +
  annotate('text', x = 0.985, y = 0.08,
           label = paste0('AUC: ', fmt3(fixed_auc_base), ' vs ', fmt3(fixed_auc_full)),
           hjust = 1, size = 4.8) +
  scale_color_manual(values = c('Snapshot only' = col_base, 'Snapshot + DeltaMRI' = col_full)) +
  scale_x_continuous(limits = c(-0.02, 1.05), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.02, 1.05), expand = c(0, 0),
                     breaks = c(0,0.25,0.5,0.75,1.0),
                     labels = number_format(accuracy = 0.01)) +
  labs(tag = 'A', title = 'ROC', x = 'False Positive Rate', y = 'True Positive Rate') +
  fig_theme

p2 <- ggplot() +
  geom_line(data = pr_base, aes(recall, precision, color = 'Snapshot only'), linewidth = 1.2) +
  geom_line(data = pr_full, aes(recall, precision, color = 'Snapshot + DeltaMRI'), linewidth = 1.2) +
  annotate('text', x = 0.985, y = 0.08,
           label = paste0('AP: ', fmt3(fixed_ap_base), ' vs ', fmt3(fixed_ap_full)),
           hjust = 1, size = 4.8) +
  scale_color_manual(values = c('Snapshot only' = col_base, 'Snapshot + DeltaMRI' = col_full)) +
  scale_x_continuous(limits = c(-0.02, 1.05), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.05, 1.05), expand = c(0, 0),
                     breaks = c(0,0.25,0.5,0.75,1.0),
                     labels = number_format(accuracy = 0.01)) +
  labs(tag = 'B', title = 'Precision-Recall', x = 'Recall', y = 'Precision') +
  fig_theme

cal_plot_df <- bind_rows(
  cal_base %>% mutate(model = 'Snapshot only'),
  cal_full %>% mutate(model = 'Snapshot + DeltaMRI')
)

p3 <- ggplot(cal_plot_df, aes(mean_pred, obs_rate, color = model)) +
  geom_abline(slope = 1, intercept = 0, linetype = 2, color = col_none, linewidth = 0.8) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.8) +
  scale_color_manual(values = c('Snapshot only' = col_base, 'Snapshot + DeltaMRI' = col_full)) +
  scale_x_continuous(limits = c(-0.01, max(cal_plot_df$mean_pred) * 1.05), expand = c(0, 0)) +
  scale_y_continuous(limits = c(-0.02, 0.65), expand = c(0, 0),
                     breaks = c(0,0.2,0.4,0.6),
                     labels = number_format(accuracy = 0.01)) +
  labs(tag = 'C', title = 'Calibration', x = 'Mean predicted probability', y = 'Observed event rate') +
  fig_theme

nb_plot_df <- bind_rows(
  nb_base %>% transmute(threshold, net_benefit = nb_model, model = 'Snapshot only'),
  nb_full %>% transmute(threshold, net_benefit = nb_model, model = 'Snapshot + DeltaMRI'),
  nb_base %>% transmute(threshold, net_benefit = nb_all, model = 'Treat all'),
  nb_base %>% transmute(threshold, net_benefit = nb_none, model = 'Treat none')
)

p4 <- ggplot(nb_plot_df, aes(threshold, net_benefit, color = model, linetype = model)) +
  geom_line(linewidth = 1.1) +
  annotate('text', x = 0.295, y = max(nb_plot_df$net_benefit, na.rm = TRUE) * 0.90,
           label = paste0('dAUC = ', fmt4(fixed_dauc), ', p = ', fmt_p(fixed_p)),
           hjust = 1, size = 4.8) +
  scale_color_manual(values = c(
    'Snapshot only' = col_base,
    'Snapshot + DeltaMRI' = col_full,
    'Treat all' = col_all,
    'Treat none' = col_none
  )) +
  scale_linetype_manual(values = c(
    'Snapshot only' = 'solid',
    'Snapshot + DeltaMRI' = 'solid',
    'Treat all' = 'dashed',
    'Treat none' = 'dashed'
  )) +
  scale_x_continuous(limits = c(0.03, 0.30), breaks = c(0.10, 0.20, 0.30), expand = c(0, 0)) +
  labs(tag = 'D', title = 'Decision Curve Analysis', x = 'Threshold probability', y = 'Net benefit') +
  fig_theme

legend_plot <- ggplot(
  data.frame(x = 1:4, y = 1, model = c('Snapshot only','Snapshot + DeltaMRI','Treat all','Treat none')),
  aes(x, y, color = model, linetype = model)
) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(values = c(
    'Snapshot only' = col_base,
    'Snapshot + DeltaMRI' = col_full,
    'Treat all' = col_all,
    'Treat none' = col_none
  )) +
  scale_linetype_manual(values = c(
    'Snapshot only' = 'solid',
    'Snapshot + DeltaMRI' = 'solid',
    'Treat all' = 'dashed',
    'Treat none' = 'dashed'
  )) +
  theme_void() +
  theme(legend.position = 'bottom', legend.title = element_blank(), legend.text = element_text(size = 11))

shared_legend <- cowplot::get_legend(legend_plot)

panel_grid <- cowplot::plot_grid(
  p1 + theme(legend.position = 'none'),
  p2 + theme(legend.position = 'none'),
  p3 + theme(legend.position = 'none'),
  p4 + theme(legend.position = 'none'),
  ncol = 2,
  align = 'hv',
  axis = 'tblr'
)

final_plot <- cowplot::plot_grid(
  panel_grid,
  shared_legend,
  ncol = 1,
  rel_heights = c(1, 0.08)
)

out_png <- file.path(base_dir, 'Figure3_locked_final.png')
out_pdf <- file.path(base_dir, 'Figure3_locked_final.pdf')

ggsave(out_png, final_plot, width = 14, height = 10, dpi = 320, bg = 'white')
ggsave(out_pdf, final_plot, width = 14, height = 10, bg = 'white')

cat('Saved:\n- ', out_png, '\n- ', out_pdf, '\n', sep = '')
