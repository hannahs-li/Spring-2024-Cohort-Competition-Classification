# Classification Prediction Problem ----
# Analysis of trained models with ks recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load fits/tunings for all ks models ----
list.files(
  here("results/"),
  "_ks",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)

# build metrics table for all models with ks recipe ----
roc_auc <- c("0.86098",
             "0.95784",
             "0.95985",
             "0.90399",
             "0.91360")
models <- c("Logistic Regression", 
            "Boosted Tree (XGBoost)",
            "Boosted Tree (Light GBM)",
            "SVM Poly",
            "SVM Rbf")
recipes <- c("Kitchen Sink")
runtimes <- c(15.877, 1044.745, 1486.823, 2139.584, 1555) # Runtime in seconds

# combine metrics to build table 
metrics_ks <- data.frame(
  models = models,
  recipes = "Kitchen Sink",
  roc_auc = roc_auc,
  runtimes = runtimes
) |> 
  arrange(roc_auc)

table_metrics_ks <- metrics_ks |> 
  kbl(label = "Kitchen Sink Model Metrics") |> 
  kable_styling() 

# save 
save(metrics_ks, file = here("results/metrics_ks.rda"))
save(table_metrics_ks, file = here("figures_tables/table_metrics_ks.rda"))