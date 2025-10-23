# Classification Prediction Problem ----
# Analysis of all trained models ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(kableExtra)

# handle common conflicts
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))
load(here("data/test.rda"))
###############################################################################
# Model metrics ----
###############################################################################
# load model metrics 
load(here("results/metrics_fe.rda"))
load(here("results/metrics_ks.rda"))

# make combine metric table ----
all_metrics <- metrics_fe |> full_join(metrics_ks) |> 
  arrange(desc(roc_auc))

table_all_metrics <- all_metrics |> 
  kbl() |> 
  kable_styling()

# save ---- 
save(table_all_metrics, file = here("figures_tables/table_all_metrics.rda"))

###############################################################################
# Final Model Parameters ----
###############################################################################
# best parameters for BT LGBM ----
load(here("results/tune_lgbm_bt_fe.rda"))

# build plot 
parameters_lbt <- autoplot(tune_lgbm_bt_fe) 

# build table 
table_lbt_parameters <- show_best(tune_lgbm_bt_fe, metric = "roc_auc") |> slice_head(n = 1) |> 
  select(-.estimator, -n, -.metric, -mean, -std_err) |> 
  kbl() |> kable_styling()

# save 
save(parameters_lbt, file = here("figures_tables/parameters_lbt.rda"))
save(table_lbt_parameters, file = here("figures_tables/table_lbt_parameters.rda"))

# best parameters for SVM poly  ----
load(here("results/tune_svm_poly_fe.rda"))

# build plot 
# parameters_svmp <- autoplot(tune_svm_poly_fe) 

# build table 
table_svmp_parameters <- show_best(tune_svm_poly_fe, metric = "roc_auc") |> slice_head(n = 1) |> 
  select(-.estimator, -n, -.metric, -mean, -std_err) |> 
  kbl() |> kable_styling()

# save 
save(parameters_svmp, file = here("figures_tables/parameters_svmp.rda"))
save(table_svmp_parameters, file = here("figures_tables/table_svmp_parameters.rda"))




