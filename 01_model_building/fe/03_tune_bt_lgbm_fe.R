# Regression Prediction Problem ----
# Define and fit lgbm bt model on fe recipe ----

# load packages ----
library(tidyverse)
library(tidymodels)
library(here)
library(doMC)
library(tictoc)
library(bonsai)

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
load(here("recipes/recipe_fe.rda"))

# model specs ----
bt_spec <- boost_tree(
  mode = "classification",
  mtry = tune(),
  learn_rate = tune(),
  min_n = tune(),
  trees = 1000,
  tree_depth = tune()
) |>
  set_engine("lightgbm")

# workflow ----
bt_wflow <- workflow() |> 
  add_model(bt_spec) |> 
  add_recipe(recipe_fe)

# hyperparameter tuning values ----
bt_params <- extract_parameter_set_dials(bt_spec) |> 
  update(mtry = mtry(c(18, 25))) |> 
  update(learn_rate = learn_rate(c(-1.6, -1.4))) |> 
  update(min_n = min_n(c(2, 10))) |> 
  update(tree_depth = tree_depth(c(6, 7)))

bt_grid <- grid_regular(bt_params, levels = c(3, 2, 2, 8))

# tune/fit workflow/model ----
tic.clearlog() # clear log
tic("BT LGBM: FE") # start clock

# tuning code in here
tune_bt_lgbm_fe <- tune_grid(
  bt_wflow, 
  resamples = folds,
  grid = bt_grid,
  metrics = metric_set(roc_auc),
  control = stacks::control_stack_resamples()
)

toc(log = TRUE)

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_bt_lgbm_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_bt_lgbm_fe, file = here("results/tune_bt_lgbm_fe.rda"))
save(tictoc_bt_lgbm_fe, file = here("results/tictoc_bt_lgbm_fe.rda"))

# load(here("results/tune_lgbm_bt_fe.rda"))
# autoplot(tune_lgbm_bt_fe)
# show_best(tune_lgbm_bt_fe, metric = "roc_auc")

# collect metrics ----
# extract final workflow 
final_wflow <- tune_bt_lgbm_fe |> 
  extract_workflow(tune_bt_lgbm_fe) |>  
  finalize_workflow(select_best(tune_bt_lgbm_fe, metric = "roc_auc"))

# train final model 
set.seed(1234)
final_tune_bt_lgbm_fe <- fit(final_wflow, train)

# compare to test submission  ----
load(here("data/test.rda"))
submission_test_bt_lgbm_fe <- bind_cols(test, predict(final_tune_bt_lgbm_fe, test, type = "prob")) |> 
  mutate(predicted = .pred_TRUE) |> 
  select(id, predicted)

# save submission test ----
write_csv(submission_test_bt_lgbm_fe, file = here("submissions/submission_test_bt_lgbm_fe.csv"))
