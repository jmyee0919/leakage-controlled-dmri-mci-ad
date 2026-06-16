# Figure 4 final polish + ABCD panel tags
# ADNI sensitivity across analysis settings
# Run with: Rscript Figure4_polished_final_ABCD.R
# Output: Figure4_final.png and Figure4_final.pdf in the current folder

pkgs <- c("ggplot2", "dplyr", "patchwork", "scales")
for (p in pkgs) {
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(patchwork)
  library(scales)
})

# -----------------------------
# Final locked values
# -----------------------------
settings <- c("All\nSigmoid", "Filtered\nSigmoid", "Filtered\nIsotonic", "All\nIsotonic")

# Colors aligned with the project palette
col_snapshot <- "#0AA087FF"   # teal
col_delta    <- "#DC0000FF"   # red

# Panel A: ROC AUC + 95% CI
roc_df <- tibble::tibble(
  setting = factor(rep(settings, each = 2), levels = settings),
  Set = rep(c("Snapshot only", "Snapshot + DeltaMRI"), times = 4),
  mean = c(
    0.8119, 0.8315,   # All Sigmoid
    0.8445, 0.8710,   # Filtered Sigmoid
    0.8423, 0.8690,   # Filtered Isotonic
    0.8083, 0.8275    # All Isotonic
  ),
  lower = c(
    0.7807505709965901, 0.7954371276458857,
    0.8127785958211371, 0.8428129509379509,
    0.8101696834315892, 0.8385155084440042,
    0.7754835911502953, 0.7938136931890617
  ),
  upper = c(
    0.8430622220507089, 0.8626395124570652,
    0.8776307045199973, 0.8999919437570769,
    0.8750800304878048, 0.8978708299926110,
    0.8417586579430547, 0.8592437847877334
  )
)

# Panel B: PR-AUC + 95% CI
pr_df <- tibble::tibble(
  setting = factor(rep(settings, each = 2), levels = settings),
  Set = rep(c("Snapshot only", "Snapshot + DeltaMRI"), times = 4),
  mean = c(
    0.4085, 0.4412,   # All Sigmoid
    0.5358, 0.5517,   # Filtered Sigmoid
    0.5212, 0.5359,   # Filtered Isotonic
    0.4035, 0.4294    # All Isotonic
  ),
  lower = c(
    0.3345454767077358, 0.3669041118673638,
    0.44752588964430773, 0.458447758128475,
    0.42885457199449656, 0.4440658636399708,
    0.3304238697592197, 0.35054874547258436
  ),
  upper = c(
    0.47926094721434137, 0.5311156206471417,
    0.6211732795171880, 0.6391570889161924,
    0.6046923105780035, 0.6229966371299425,
    0.4843705537667649, 0.5182348269561850
  )
)

# Panel C: Brier score
brier_df <- tibble::tibble(
  setting = factor(rep(settings, each = 2), levels = settings),
  Set = rep(c("Snapshot only", "Snapshot + DeltaMRI"), times = 4),
  mean = c(
    0.1033, 0.1010,
    0.1023, 0.0983,
    0.1021, 0.0989,
    0.1048, 0.1015
  )
)

# Panel D: Delta AUC + p values
dauc_df <- tibble::tibble(
  setting = factor(settings, levels = settings),
  dAUC = c(0.0197, 0.0266, 0.0266, 0.0192),
  p_label = c("p = 0.0731", "p = 0.0025", "p = 0.0045", "p = 0.1156")
)

# -----------------------------
# Theme
# -----------------------------
base_theme <- theme_classic(base_size = 14) +
  theme(
    plot.title = element_text(face = "bold", size = 16, hjust = 0),
    axis.title = element_text(face = "bold"),
    axis.text = element_text(color = "black"),
    legend.title = element_blank(),
    legend.text = element_text(size = 12),
    legend.position = "bottom",
    legend.box = "horizontal",
    panel.border = element_blank(),
    plot.margin = margin(8, 8, 8, 8)
  )

color_scale <- scale_color_manual(values = c(
  "Snapshot only" = col_snapshot,
  "Snapshot + DeltaMRI" = col_delta
))

# -----------------------------
# Panel A
# -----------------------------
pA <- ggplot(roc_df, aes(x = setting, y = mean, color = Set, group = Set)) +
  geom_line(linewidth = 1.0, position = position_dodge(width = 0.12)) +
  geom_point(size = 3.2, position = position_dodge(width = 0.12)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.08, linewidth = 0.8,
                position = position_dodge(width = 0.12)) +
  color_scale +
  scale_y_continuous(limits = c(0.78, 0.91), labels = number_format(accuracy = 0.01)) +
  labs(title = "ROC AUC across ADNI settings", x = NULL, y = "ROC AUC") +
  base_theme

# -----------------------------
# Panel B
# -----------------------------
pB <- ggplot(pr_df, aes(x = setting, y = mean, color = Set, group = Set)) +
  geom_line(linewidth = 1.0, position = position_dodge(width = 0.12)) +
  geom_point(size = 3.2, position = position_dodge(width = 0.12)) +
  geom_errorbar(aes(ymin = lower, ymax = upper), width = 0.08, linewidth = 0.8,
                position = position_dodge(width = 0.12)) +
  color_scale +
  scale_y_continuous(limits = c(0.33, 0.67), labels = number_format(accuracy = 0.01)) +
  labs(title = "PR-AUC across ADNI settings", x = NULL, y = "PR-AUC") +
  base_theme

# -----------------------------
# Panel C
# -----------------------------
pC <- ggplot(brier_df, aes(x = setting, y = mean, color = Set, group = Set)) +
  geom_line(linewidth = 1.0, position = position_dodge(width = 0.12)) +
  geom_point(size = 3.2, position = position_dodge(width = 0.12)) +
  color_scale +
  scale_y_continuous(limits = c(0.0975, 0.1055), labels = number_format(accuracy = 0.001)) +
  labs(title = "Brier score across ADNI settings", x = NULL, y = "Brier score") +
  base_theme

# -----------------------------
# Panel D
# -----------------------------
pD <- ggplot(dauc_df, aes(x = setting, y = dAUC, group = 1)) +
  geom_line(color = col_delta, linewidth = 1.0) +
  geom_point(color = col_delta, size = 3.2) +
  geom_text(aes(label = p_label, y = dAUC + 0.0012), size = 4.5, color = "black") +
  scale_y_continuous(limits = c(0.015, 0.0315), labels = number_format(accuracy = 0.001)) +
  labs(title = "Delta AUC across ADNI settings", x = NULL, y = expression(Delta*"AUC")) +
  base_theme +
  theme(legend.position = "none")

# -----------------------------
# Assemble with one shared legend + panel tags
# -----------------------------
fig4 <- ((pA | pB) / (pC | pD)) +
  plot_layout(guides = "collect") +
  plot_annotation(tag_levels = "A") &
  theme(
    legend.position = "bottom",
    plot.tag = element_text(face = "bold", size = 16),
    plot.tag.position = c(0.02, 0.98)
  )

# Save
out_png <- file.path(getwd(), "Figure4_final.png")
out_pdf <- file.path(getwd(), "Figure4_final.pdf")

ggsave(out_png, fig4, width = 14, height = 10, dpi = 300, bg = "white")
ggsave(out_pdf, fig4, width = 14, height = 10, bg = "white")

cat("Saved:\n")
cat("- ", out_png, "\n", sep = "")
cat("- ", out_pdf, "\n", sep = "")