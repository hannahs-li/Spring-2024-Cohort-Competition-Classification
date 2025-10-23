# Classification Prediction Problem ----
# Analysis of trained models with fe recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load fits/tunings for all fe models ----
list.files(
  here("results/"),
  "_fe",
  full.names = TRUE) |> 
  map(load, envir = .GlobalEnv)

# build metrics table for all models with fe recipe ----
roc_auc <- c("0.87164",
             "0.95834",
             "0.96238",
             "0.90708",
             "0.91386")
models <- c("Logistic Regression", 
            "Boosted Tree (XGBoost)",
            "Boosted Tree (Light GBM)",
            "SVM Poly",
            "SVM Rbf")
recipes <- c("Feature Engineering")
runtimes <- c(20.653, 843.896, 2013.138, 890.435, 969.727) # Runtime in seconds

# combine metrics 
metrics_fe <- data.frame(
  models = models,
  recipes = "Feature Engineering",
  roc_auc = roc_auc,
  runtimes = runtimes
) |> 
  arrange(roc_auc)

table_metrics_fe <- metrics_fe |> 
  kbl(label = "Feature Engineered Model Metrics") |> 
  kable_styling() 

# save 
save(metrics_fe, file = here("results/metrics_fe.rda"))
save(table_metrics_fe, file = here("figures_tables/table_metrics_fe.rda"))





