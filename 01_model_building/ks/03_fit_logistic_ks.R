# Regression Prediction Problem ----
# Define and fit lgbm bt model on ks recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)

# handle common conflicts
tidymodels_prefer()

# set parallel processing
num_cores <- parallel::detectCores(logical = TRUE)
registerDoMC(cores = num_cores - 1)
cl <- makePSOCKcluster(4)

# load training data
load(here("data/folds.rda"))
load(here("data/train.rda"))
load(here("data/test.rda"))

# load pre-processing/feature engineering/recipe
load(here("recipes/recipe_ks.rda"))

# model specs ----
log_spec <- logistic_reg() |>
  set_mode("classification") |>
  set_engine("glm")

# workflow ----
log_wflow <-
  workflow() |>
  add_model(log_spec) |>
  add_recipe(recipe_ks)

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("Logistic: KS") # start clock

# tuning code in here
set.seed(1234)
fit_log_ks <- fit_resamples(
  log_wflow,
  resamples = folds,
  metrics = metric_set(roc_auc),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_log_ks <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(fit_log_ks, file = here("results/fit_log_ks.rda"))
save(tictoc_log_ks, file = here("results/tictoc_log_ks.rda"))

# collect metrics ----
# final wflow 
final_wflow <- fit_log_ks |> 
  extract_workflow(fit_log_ks) |>  
  finalize_workflow(select_best(fit_log_ks, metric = "roc_auc"))

# train final model 
set.seed(1234)
final_fit_log_ks <- fit(final_wflow, train)

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_log_ks <- bind_cols(test, predict(final_fit_log_ks, test, type = "prob")) |> 
  mutate(predicted = .pred_TRUE) |> 
  select(id, predicted)

# save submission test ----
write_csv(submission_test_log_ks, file = here("submissions/submission_test_log_ks.csv"))

# roc_auc is 0.86098 
# Logistic: KS: 5.029 elapsed 