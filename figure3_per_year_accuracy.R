# figure3_per_year_accuracy.R
#
# Regenerates Figure 3 with the genuine, revised per-year results (six
# Bayesian-alphabet models now reflect real BGLR fits; BayesR remains the
# original disclosed proxy; all other 23 models unchanged from the
# original submission). Data matches Supplementary Table S4 exactly.
#
# Figure 3. Five-fold cross-validation prediction accuracy (Pearson r)
# of thirty genomic prediction models -- ten conventional (red), ten
# machine-learning (blue), and ten deep-learning (green) -- for powdery
# mildew severity in (A) DS1, (B) DS2, and (C) DS3.
#
# REQUIREMENTS: ggplot2, patchwork
#   install.packages(c("ggplot2", "patchwork"))
#
# USAGE: paste this whole script into R / RStudio and run it. Saves
# Figure3_per_year_accuracy.png (and .pdf) to your working directory.

library(ggplot2)
library(patchwork)

# --------------------------------------------------------------------------
# DATA -- exactly matches Supplementary Table S4 (this revision).
# Five parallel vectors, all length 30, aligned by position.
# --------------------------------------------------------------------------
Model_vec <- c("BayesR", "BayesC", "BayesB", "BayesA", "BL", "ElasticNet", "BayesCpi", "BRR", "RandomForest", "LightGBM", "rrBLUP", "GBLUP", "SVR", "ResNet", "GBM", "DNN", "DualCNN", "XGBoost", "CNN", "Autoencoder", "RNN", "LSTM", "GRU", "Transformer", "KRR", "DecisionTree", "MLP", "PLS", "KNN", "RKHS")

Class_vec <- c("Conventional", "Conventional", "Conventional", "Conventional", "Conventional", "ML", "Conventional", "Conventional", "ML", "ML", "Conventional", "Conventional", "ML", "DL", "ML", "DL", "DL", "ML", "DL", "DL", "DL", "DL", "DL", "DL", "ML", "ML", "DL", "ML", "ML", "Conventional")

DS1_vec <- c(0.524, 0.54, 0.545, 0.543, 0.537, 0.505, 0.543, 0.54, 0.335, 0.384, 0.52, 0.495, 0.4, 0.432, 0.4, 0.42, 0.433, 0.384, 0.441, 0.401, 0.475, 0.466, 0.473, 0.352, 0.339, 0.323, 0.405, 0.33, 0.424, 0.383)

DS2_vec <- c(0.565, 0.701, 0.702, 0.702, 0.702, 0.643, 0.702, 0.701, 0.438, 0.594, 0.649, 0.617, 0.434, 0.399, 0.342, 0.339, 0.368, 0.594, 0.473, 0.372, 0.339, 0.424, 0.442, 0.453, 0.425, 0.45, 0.374, 0.402, 0.367, 0.391)

DS3_vec <- c(0.455, 0.472, 0.473, 0.464, 0.476, 0.521, 0.471, 0.472, 0.369, 0.411, 0.463, 0.343, 0.339, 0.385, 0.395, 0.419, 0.435, 0.411, 0.381, 0.42, 0.416, 0.437, 0.42, 0.387, 0.387, 0.449, 0.467, 0.478, 0.349, 0.467)

df <- data.frame(
  Model = Model_vec,
  Class = factor(Class_vec, levels = c("Conventional", "ML", "DL")),
  DS1 = DS1_vec,
  DS2 = DS2_vec,
  DS3 = DS3_vec,
  stringsAsFactors = FALSE
)

stopifnot(nrow(df) == 30, !any(is.na(df$DS1)), !any(is.na(df$DS2)), !any(is.na(df$DS3)))

class_colors <- c("Conventional" = "#D7191C", "ML" = "#2C7BB6", "DL" = "#1A9641")  # red, blue, green

make_panel <- function(data, year_col, panel_label) {
  pdat <- data.frame(Model = data$Model, Class = data$Class, r = data[[year_col]])
  pdat <- pdat[order(pdat$r, decreasing = TRUE), ]
  pdat$Model <- factor(pdat$Model, levels = rev(pdat$Model))  # top-to-bottom = highest-to-lowest

  ggplot(pdat, aes(x = Model, y = r, fill = Class)) +
    geom_col(width = 0.75) +
    geom_hline(yintercept = 0, linewidth = 0.3) +
    coord_flip() +
    scale_fill_manual(values = class_colors, drop = FALSE, name = "Model class") +
    labs(title = panel_label, x = NULL, y = "Prediction accuracy (Pearson r)") +
    theme_minimal(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 11),
      axis.text.y = element_text(size = 7.5),
      panel.grid.minor = element_blank()
    )
}

pA <- make_panel(df, "DS1", "A. DS1")
pB <- make_panel(df, "DS2", "B. DS2")
pC <- make_panel(df, "DS3", "C. DS3")

# patchwork collects and merges the three identical legends into one,
# placed at the bottom -- no extra package (e.g. cowplot) needed.
final_plot <- (pA | pB | pC) +
  plot_layout(guides = "collect") &
  theme(legend.position = "bottom")

print(final_plot)

ggsave("Figure3_per_year_accuracy.png", final_plot, width = 14, height = 7, dpi = 300)
ggsave("Figure3_per_year_accuracy.pdf", final_plot, width = 14, height = 7)

cat("Saved Figure3_per_year_accuracy.png and .pdf to", getwd(), "\n")
