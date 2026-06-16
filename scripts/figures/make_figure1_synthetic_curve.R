
# =========================================================
# Figure 1 FINAL (drop-in script)
# Put this file in the same folder as:
#   A_s1(3-18).csv
#   A_m1(3-18).csv
# Then run:
#   Rscript figure1_final_dropin.R
# =========================================================

# ---------------------------
# 0) Packages
# ---------------------------
pkgs <- c("ggplot2", "dplyr", "readr", "tibble", "patchwork", "pROC", "PRROC", "scales")
to_install <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(to_install) > 0) install.packages(to_install, repos = "https://cloud.r-project.org")

library(ggplot2)
library(dplyr)
library(readr)
library(tibble)
library(patchwork)
library(pROC)
library(PRROC)
library(scales)

# ---------------------------
# 1) Robust working directory
# ---------------------------
get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  if (!is.null(sys.frames()[[1]]$ofile)) {
    return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  }
  return(getwd())
}

base_dir <- get_script_dir()
cat("Using base_dir:", base_dir, "\n")

# ---------------------------
# 2) File paths
# ---------------------------
snapshot_file <- file.path(base_dir, "A_s1(3-18).csv")
delta_file    <- file.path(base_dir, "A_m1(3-18).csv")

if (!file.exists(snapshot_file) || !file.exists(delta_file)) {
  stop(
    paste0(
      "Could not find required files in the same folder as this R script:\n",
      " - A_s1(3-18).csv\n",
      " - A_m1(3-18).csv\n",
      "Current folder: ", base_dir
    )
  )
}

# output files
out_pdf   <- file.path(base_dir, "Figure1_final.pdf")
out_png   <- file.path(base_dir, "Figure1_final.png")
out_mets  <- file.path(base_dir, "Figure1_final_metrics.csv")
out_del   <- file.path(base_dir, "Figure1_final_delong.csv")

# ---------------------------
# 3) Colors requested by user
# ---------------------------
col_base <- "#00A087FF"  # teal
col_full <- "#DC0000FF"  # red
col_gray <- "#8A8A8A"

# ---------------------------
# 4) Helper functions
# ---------------------------
fmt_num <- function(x, digits = 3) sprintf(paste0("%.", digits, "f"), x)

fmt_p <- function(p) {
  if (is.na(p)) return("NA")
  if (p < 0.001) return(format(p, scientific = TRUE, digits = 2))
  sprintf("%.3f", p)
}

pick_first_existing <- function(df, candidates, required = TRUE, label = "column") {
  hit <- candidates[candidates %in% names(df)]
  if (length(hit) > 0) return(hit[1])
  if (required) stop(paste0("Could not find ", label, ". Tried: ", paste(candidates, collapse = ", ")))
  return(NA_character_)
}

safe_num <- function(x) {
  out <- suppressWarnings(as.numeric(x))
  out
}

compute_relchange <- function(base, follow, interval_months) {
  interval_months <- ifelse(is.na(interval_months) | interval_months <= 0, 0.1, interval_months)
  ((follow - base) / base) / (interval_months / 12)
}

calibration_stats <- function(y, prob) {
  eps <- 1e-6
  p_clip <- pmin(pmax(prob, eps), 1 - eps)
  lp <- qlogis(p_clip)
  fit <- glm(y ~ lp, family = binomial())
  tibble(
    cal_intercept = unname(coef(fit)[1]),
    cal_slope     = unname(coef(fit)[2])
  )
}

metrics_from_pred <- function(y, prob, model_name) {
  roc_obj <- pROC::roc(response = y, predictor = prob, quiet = TRUE)
  ci_auc  <- as.numeric(pROC::ci.auc(roc_obj))
  pr_obj <- PRROC::pr.curve(
    scores.class0 = prob[y == 1],
    scores.class1 = prob[y == 0],
    curve = TRUE
  )
  brier <- mean((prob - y)^2)
  cal <- calibration_stats(y, prob)
  tibble(
    model = model_name,
    roc_auc = as.numeric(pROC::auc(roc_obj)),
    roc_auc_ci_low = ci_auc[1],
    roc_auc_ci_high = ci_auc[3],
    pr_ap = pr_obj$auc.integral,
    brier = brier,
    cal_slope = cal$cal_slope,
    cal_intercept = cal$cal_intercept,
    n_test = length(y)
  )
}

roc_df_from_pred <- function(y, prob, model_name) {
  roc_obj <- pROC::roc(response = y, predictor = prob, quiet = TRUE)
  tibble(
    fpr = 1 - roc_obj$specificities,
    tpr = roc_obj$sensitivities,
    model = model_name
  )
}

pr_df_from_pred <- function(y, prob, model_name) {
  pr_obj <- PRROC::pr.curve(
    scores.class0 = prob[y == 1],
    scores.class1 = prob[y == 0],
    curve = TRUE
  )
  tibble(
    recall = pr_obj$curve[, 1],
    precision = pr_obj$curve[, 2],
    model = model_name
  )
}

calibration_df_from_pred <- function(y, prob, n_bins = 10, model_name) {
  tibble(y = y, prob = prob) %>%
    mutate(bin = ntile(prob, n_bins)) %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob),
      obs_rate = mean(y),
      .groups = "drop"
    ) %>%
    mutate(model = model_name)
}

dca_df_from_pred <- function(y, prob, thresholds, model_name) {
  n <- length(y)
  bind_rows(lapply(thresholds, function(pt) {
    pred_pos <- prob >= pt
    tp <- sum(pred_pos & y == 1)
    fp <- sum(pred_pos & y == 0)
    nb <- (tp / n) - (fp / n) * (pt / (1 - pt))
    tibble(threshold = pt, net_benefit = nb, model = model_name)
  }))
}

treat_all_df <- function(y, thresholds) {
  prev <- mean(y)
  tibble(
    threshold = thresholds,
    net_benefit = prev - (1 - prev) * (threshold / (1 - threshold)),
    model = "Treat all"
  )
}

treat_none_df <- function(thresholds) {
  tibble(
    threshold = thresholds,
    net_benefit = 0,
    model = "Treat none"
  )
}

theme_fig <- function() {
  theme_classic(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      plot.tag = element_text(face = "bold", size = 18),
      plot.caption = element_text(size = 10, hjust = 0)
    )
}

# ---------------------------
# 5) Load data
# ---------------------------
snap <- read_csv(snapshot_file, show_col_types = FALSE)
delt <- read_csv(delta_file, show_col_types = FALSE)

names(snap) <- tolower(names(snap))
names(delt) <- tolower(names(delt))

# subject id
sid_snap <- pick_first_existing(snap, c("subjectid", "subject_id", "rid"), label = "snapshot subject id")
sid_delt <- pick_first_existing(delt, c("subjectid", "subject_id", "rid"), label = "delta subject id")

snap[[sid_snap]] <- as.character(snap[[sid_snap]])
delt[[sid_delt]] <- as.character(delt[[sid_delt]])

names(snap)[names(snap) == sid_snap] <- "subjectid"
names(delt)[names(delt) == sid_delt] <- "subjectid"

# outcome
y_snap_col <- pick_first_existing(snap, c("converter", "y", "label"), label = "snapshot outcome")
y_delt_col <- pick_first_existing(delt, c("converter", "y", "label"), required = FALSE, label = "delta outcome")

# interval
interval_col <- pick_first_existing(delt, c("interval_months", "months", "interval"), label = "interval months")

# region detection
# snapshot columns
hippo_snap_col <- pick_first_existing(snap, c("hippo_base", "st29sv", "st29sv_base", "hippocampus", "hippo"), label = "snapshot hippocampus")
vent_snap_col  <- pick_first_existing(snap, c("vent_base", "st37sv", "st37sv_base", "ventricle", "vent"), label = "snapshot ventricle")
entorh_snap_col <- pick_first_existing(snap, c("entorh_base", "st149sv", "st149sv_base", "entorhinal", "entorh"), required = FALSE, label = "snapshot entorhinal")

# delta direct feature names if already present
hippo_rel_col  <- pick_first_existing(delt, c("hippo_relchange_emp", "st29sv_relchange_emp"), required = FALSE, label = "hippo relchange")
vent_rel_col   <- pick_first_existing(delt, c("vent_relchange_emp", "st37sv_relchange_emp"), required = FALSE, label = "vent relchange")
entorh_rel_col <- pick_first_existing(delt, c("entorh_relchange_emp", "st149sv_relchange_emp"), required = FALSE, label = "entorh relchange")

# if not direct, derive from base/follow
if (is.na(hippo_rel_col)) {
  hippo_base_col <- pick_first_existing(delt, c("hippo_base", "st29sv_base"), label = "delta hippo base")
  hippo_follow_col <- pick_first_existing(delt, c("hippo_follow", "st29sv_follow"), label = "delta hippo follow")
}
if (is.na(vent_rel_col)) {
  vent_base_col <- pick_first_existing(delt, c("vent_base", "st37sv_base"), label = "delta vent base")
  vent_follow_col <- pick_first_existing(delt, c("vent_follow", "st37sv_follow"), label = "delta vent follow")
}
if (is.na(entorh_rel_col) && !is.na(entorh_snap_col)) {
  entorh_base_col <- pick_first_existing(delt, c("entorh_base", "st149sv_base"), required = FALSE, label = "delta entorh base")
  entorh_follow_col <- pick_first_existing(delt, c("entorh_follow", "st149sv_follow"), required = FALSE, label = "delta entorh follow")
}

# missing indicators if already present
hippo_isna_col  <- pick_first_existing(delt, c("hippo_relchange_emp_isna", "st29sv_relchange_emp_isna"), required = FALSE, label = "hippo isna")
vent_isna_col   <- pick_first_existing(delt, c("vent_relchange_emp_isna", "st37sv_relchange_emp_isna"), required = FALSE, label = "vent isna")
entorh_isna_col <- pick_first_existing(delt, c("entorh_relchange_emp_isna", "st149sv_relchange_emp_isna"), required = FALSE, label = "entorh isna")

# ---------------------------
# 6) Build merged analysis table
# ---------------------------
snap_small <- snap %>%
  transmute(
    subjectid = as.character(subjectid),
    y_snap = as.integer(!!as.name(y_snap_col)),
    hippo_base = safe_num(!!as.name(hippo_snap_col)),
    vent_base  = safe_num(!!as.name(vent_snap_col)),
    entorh_base = if (!is.na(entorh_snap_col)) safe_num(!!as.name(entorh_snap_col)) else NA_real_
  )

delt_small <- delt %>%
  mutate(interval_m = safe_num(!!as.name(interval_col))) %>%
  transmute(
    subjectid = as.character(subjectid),
    y_delta = if (!is.na(y_delt_col)) as.integer(!!as.name(y_delt_col)) else NA_integer_,
    interval_m = interval_m,

    hippo_relchange_emp = if (!is.na(hippo_rel_col)) {
      safe_num(!!as.name(hippo_rel_col))
    } else {
      compute_relchange(
        safe_num(!!as.name(hippo_base_col)),
        safe_num(!!as.name(hippo_follow_col)),
        interval_m
      )
    },

    vent_relchange_emp = if (!is.na(vent_rel_col)) {
      safe_num(!!as.name(vent_rel_col))
    } else {
      compute_relchange(
        safe_num(!!as.name(vent_base_col)),
        safe_num(!!as.name(vent_follow_col)),
        interval_m
      )
    },

    entorh_relchange_emp = if (!is.na(entorh_snap_col)) {
      if (!is.na(entorh_rel_col)) {
        safe_num(!!as.name(entorh_rel_col))
      } else if (exists("entorh_base_col") && exists("entorh_follow_col") &&
                 !is.na(entorh_base_col) && !is.na(entorh_follow_col)) {
        compute_relchange(
          safe_num(!!as.name(entorh_base_col)),
          safe_num(!!as.name(entorh_follow_col)),
          interval_m
        )
      } else {
        NA_real_
      }
    } else {
      NA_real_
    },

    hippo_relchange_emp_isna = if (!is.na(hippo_isna_col)) {
      as.integer(!!as.name(hippo_isna_col))
    } else {
      as.integer(is.na(hippo_relchange_emp))
    },

    vent_relchange_emp_isna = if (!is.na(vent_isna_col)) {
      as.integer(!!as.name(vent_isna_col))
    } else {
      as.integer(is.na(vent_relchange_emp))
    },

    entorh_relchange_emp_isna = if (!is.na(entorh_snap_col)) {
      if (!is.na(entorh_isna_col)) {
        as.integer(!!as.name(entorh_isna_col))
      } else {
        as.integer(is.na(entorh_relchange_emp))
      }
    } else {
      NA_integer_
    }
  )

dat <- snap_small %>%
  inner_join(delt_small, by = "subjectid") %>%
  mutate(
    y = ifelse(!is.na(y_delta), y_delta, y_snap),
    hippo_relchange_emp = ifelse(is.na(hippo_relchange_emp), 0, hippo_relchange_emp),
    vent_relchange_emp = ifelse(is.na(vent_relchange_emp), 0, vent_relchange_emp),
    entorh_relchange_emp = ifelse(is.na(entorh_relchange_emp), 0, entorh_relchange_emp)
  )

cat("Merged rows:", nrow(dat), "\n")
cat("Positive rate:", mean(dat$y), "\n")

# ---------------------------
# 7) Stratified split
# ---------------------------
set.seed(42)
idx_pos <- which(dat$y == 1)
idx_neg <- which(dat$y == 0)

n_pos_test <- floor(length(idx_pos) * 0.30)
n_neg_test <- floor(length(idx_neg) * 0.30)

test_idx <- c(sample(idx_pos, n_pos_test), sample(idx_neg, n_neg_test))
train_dat <- dat[-test_idx, , drop = FALSE]
test_dat  <- dat[test_idx,  , drop = FALSE]

# ---------------------------
# 8) Feature sets (auto includes entorh if present)
# ---------------------------
feats_base <- c("hippo_base", "vent_base")
if ("entorh_base" %in% names(train_dat) && !all(is.na(train_dat$entorh_base))) {
  feats_base <- c(feats_base, "entorh_base")
}

feats_full <- c(feats_base, "hippo_relchange_emp", "vent_relchange_emp",
                "hippo_relchange_emp_isna", "vent_relchange_emp_isna")

if ("entorh_base" %in% feats_base) {
  feats_full <- c(feats_full, "entorh_relchange_emp", "entorh_relchange_emp_isna")
}

# ---------------------------
# 9) Weighted logistic + train scaling
# ---------------------------
fit_predict_logistic <- function(train, test, features) {
  x_train <- train[, features, drop = FALSE]
  x_test  <- test[, features, drop = FALSE]

  mu <- sapply(x_train, mean, na.rm = TRUE)
  sdv <- sapply(x_train, sd, na.rm = TRUE)
  sdv[is.na(sdv) | sdv == 0] <- 1

  x_train_sc <- sweep(sweep(as.matrix(x_train), 2, mu, "-"), 2, sdv, "/")
  x_test_sc  <- sweep(sweep(as.matrix(x_test),  2, mu, "-"), 2, sdv, "/")

  df_train <- data.frame(y = train$y, x_train_sc)
  colnames(df_train) <- c("y", features)

  n_all <- nrow(df_train)
  n_pos <- sum(df_train$y == 1)
  n_neg <- sum(df_train$y == 0)

  w_pos <- n_all / (2 * n_pos)
  w_neg <- n_all / (2 * n_neg)
  weights_vec <- ifelse(df_train$y == 1, w_pos, w_neg)

  fit <- glm(
    reformulate(features, response = "y"),
    data = df_train,
    family = binomial(),
    weights = weights_vec
  )

  df_test <- data.frame(x_test_sc)
  colnames(df_test) <- features

  prob <- predict(fit, newdata = df_test, type = "response")
  tibble(y = test$y, prob = as.numeric(prob))
}

pred_base <- fit_predict_logistic(train_dat, test_dat, feats_base) %>% rename(prob_base = prob)
pred_full <- fit_predict_logistic(train_dat, test_dat, feats_full) %>% rename(prob_full = prob)

pred <- tibble(
  y = pred_base$y,
  prob_base = pred_base$prob_base,
  prob_full = pred_full$prob_full
)

# ---------------------------
# 10) Metrics
# ---------------------------
metrics_tbl <- bind_rows(
  metrics_from_pred(pred$y, pred$prob_base, "Snapshot only"),
  metrics_from_pred(pred$y, pred$prob_full, "Snapshot + ΔMRI")
)

roc_base <- pROC::roc(pred$y, pred$prob_base, quiet = TRUE)
roc_full <- pROC::roc(pred$y, pred$prob_full, quiet = TRUE)
delong <- pROC::roc.test(roc_base, roc_full, paired = TRUE, method = "delong")

delong_tbl <- tibble(
  comparison = "Snapshot + ΔMRI vs Snapshot only",
  delta_auc = as.numeric(pROC::auc(roc_full) - pROC::auc(roc_base)),
  z = as.numeric(delong$statistic),
  p_value = as.numeric(delong$p.value),
  n_test = nrow(pred)
)

# ---------------------------
# 11) Curves
# ---------------------------
roc_df <- bind_rows(
  roc_df_from_pred(pred$y, pred$prob_base, "Snapshot only"),
  roc_df_from_pred(pred$y, pred$prob_full, "Snapshot + ΔMRI")
)

pr_df <- bind_rows(
  pr_df_from_pred(pred$y, pred$prob_base, "Snapshot only"),
  pr_df_from_pred(pred$y, pred$prob_full, "Snapshot + ΔMRI")
)

cal_df <- bind_rows(
  calibration_df_from_pred(pred$y, pred$prob_base, n_bins = 10, model_name = "Snapshot only"),
  calibration_df_from_pred(pred$y, pred$prob_full, n_bins = 10, model_name = "Snapshot + ΔMRI")
)

thresholds_dca <- seq(0.05, 0.30, by = 0.005)
dca_df <- bind_rows(
  dca_df_from_pred(pred$y, pred$prob_base, thresholds_dca, "Snapshot only"),
  dca_df_from_pred(pred$y, pred$prob_full, thresholds_dca, "Snapshot + ΔMRI"),
  treat_all_df(pred$y, thresholds_dca),
  treat_none_df(thresholds_dca)
)

# y limits for DCA
dca_min <- min(dca_df$net_benefit[dca_df$model %in% c("Snapshot only", "Snapshot + ΔMRI", "Treat none")], na.rm = TRUE)
dca_max <- max(dca_df$net_benefit[dca_df$model %in% c("Snapshot only", "Snapshot + ΔMRI", "Treat none")], na.rm = TRUE)
dca_y_low  <- floor((dca_min - 0.01) * 100) / 100
dca_y_high <- ceiling((dca_max + 0.01) * 100) / 100

# labels
auc_base <- metrics_tbl$roc_auc[metrics_tbl$model == "Snapshot only"]
auc_full <- metrics_tbl$roc_auc[metrics_tbl$model == "Snapshot + ΔMRI"]
ap_base  <- metrics_tbl$pr_ap[metrics_tbl$model == "Snapshot only"]
ap_full  <- metrics_tbl$pr_ap[metrics_tbl$model == "Snapshot + ΔMRI"]

roc_labels <- c(
  "Snapshot only"   = paste0("Snapshot only (AUC = ", fmt_num(auc_base), ")"),
  "Snapshot + ΔMRI" = paste0("Snapshot + ΔMRI (AUC = ", fmt_num(auc_full), ")")
)

pr_labels <- c(
  "Snapshot only"   = paste0("Snapshot only (AP = ", fmt_num(ap_base), ")"),
  "Snapshot + ΔMRI" = paste0("Snapshot + ΔMRI (AP = ", fmt_num(ap_full), ")")
)

# ---------------------------
# 12) Theme
# ---------------------------
theme_set(
  theme_classic(base_size = 13) +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 10),
      axis.title = element_text(face = "bold", size = 12),
      axis.text = element_text(size = 10),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = 10),
      plot.tag = element_text(face = "bold", size = 18),
      plot.caption = element_text(size = 10, hjust = 0)
    )
)

# ---------------------------
# 13) Panels
# ---------------------------
p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr, color = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = col_gray, linewidth = 0.8) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(
    values = c("Snapshot only" = col_base, "Snapshot + ΔMRI" = col_full),
    breaks = c("Snapshot only", "Snapshot + ΔMRI"),
    labels = roc_labels
  ) +
  scale_x_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  labs(title = "ROC", x = "False Positive Rate", y = "True Positive Rate")

p_pr <- ggplot(pr_df, aes(x = recall, y = precision, color = model)) +
  geom_line(linewidth = 1.2) +
  scale_color_manual(
    values = c("Snapshot only" = col_base, "Snapshot + ΔMRI" = col_full),
    breaks = c("Snapshot only", "Snapshot + ΔMRI"),
    labels = pr_labels
  ) +
  scale_x_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  labs(title = "Precision–Recall", x = "Recall", y = "Precision")

p_cal <- ggplot(cal_df, aes(x = mean_pred, y = obs_rate, color = model)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = col_gray, linewidth = 0.8) +
  geom_line(linewidth = 1.0) +
  geom_point(size = 2.3) +
  scale_color_manual(values = c("Snapshot only" = col_base, "Snapshot + ΔMRI" = col_full)) +
  scale_x_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  scale_y_continuous(limits = c(0, 1), labels = label_number(accuracy = 0.1)) +
  labs(title = "Calibration", x = "Mean predicted probability", y = "Observed event rate")

p_dca <- ggplot(dca_df, aes(x = threshold, y = net_benefit, color = model, linetype = model)) +
  annotate("rect", xmin = 0.10, xmax = 0.30, ymin = -Inf, ymax = Inf, fill = "#F3F3F3", alpha = 0.5) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(
    values = c(
      "Snapshot only" = col_base,
      "Snapshot + ΔMRI" = col_full,
      "Treat all" = "black",
      "Treat none" = col_gray
    )
  ) +
  scale_linetype_manual(
    values = c(
      "Snapshot only" = "solid",
      "Snapshot + ΔMRI" = "solid",
      "Treat all" = "dashed",
      "Treat none" = "dashed"
    )
  ) +
  coord_cartesian(xlim = c(0.05, 0.30), ylim = c(dca_y_low, dca_y_high)) +
  labs(title = "Decision Curve Analysis", x = "Threshold probability", y = "Net benefit")

# ---------------------------
# 14) Combine
# ---------------------------
caption_txt <- paste0(
  "Synthetic positive-control cohort, representative 3–18-month setting (70/30 hold-out split). ",
  "Paired DeLong: ΔAUC = ", fmt_num(delong_tbl$delta_auc),
  ", z = ", fmt_num(delong_tbl$z),
  ", p = ", fmt_p(delong_tbl$p_value), "."
)

fig1 <- (p_roc + p_pr) / (p_cal + p_dca) +
  plot_annotation(
    title = "Figure 1. Synthetic positive-control performance (representative 3–18-month setting)",
    subtitle = "Comparison of Snapshot only versus Snapshot + ΔMRI",
    caption = caption_txt,
    tag_levels = "A"
  )

# ---------------------------
# 15) Save
# ---------------------------
ggsave(out_pdf, plot = fig1, width = 12, height = 10, device = cairo_pdf)
ggsave(out_png, plot = fig1, width = 12, height = 10, dpi = 600)

write_csv(metrics_tbl, out_mets)
write_csv(delong_tbl, out_del)

cat("\nSaved:\n")
cat(" -", out_pdf, "\n")
cat(" -", out_png, "\n")
cat(" -", out_mets, "\n")
cat(" -", out_del, "\n\n")

print(metrics_tbl)
print(delong_tbl)
print(fig1)
