# Classification Prediction Problem ----
# Define and fit svm poly model on fe recipe

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
load(here("recipes/recipe_fe.rda"))

# model specifications ----
svm_poly_spec <- svm_poly(
  cost = tune(),
  degree = tune(),
  scale_factor = tune()
) |>
  set_engine("kernlab") |>
  set_mode("classification")

# define workflows ----
svm_poly_wflow <- workflow() |>
  add_model(svm_poly_spec) |>
  add_recipe(recipe_fe)

# hyperparameter tuning values ----
svm_poly_params <- hardhat::extract_parameter_set_dials(svm_poly_spec)

# define grid
svm_poly_grid <- grid_latin_hypercube(svm_poly_params, size = 53)

# fit workflow/model ----
tic("SVM Poly: FE") # start clock

tune_svm_poly_fe <- 
  svm_poly_wflow |>
  tune_grid(
    resamples = folds,
    metrics = metric_set(roc_auc),
    grid = svm_poly_grid,
    control = stacks::control_stack_resamples()
  )

toc(log = TRUE) # stop clock

# Extract runtime info
time_log <- tic.log(format = FALSE)

tictoc_svm_poly_fe <- tibble(
  model = time_log[[1]]$msg,
  start_time = time_log[[1]]$tic,
  end_time = time_log[[1]]$toc,
  runtime = end_time - start_time
)

# write out results (fitted/trained workflows & runtime info) ----
save(tune_svm_poly_fe, file = here("results/tune_svm_poly_fe.rda"))
save(tictoc_svm_poly_fe, file = here("results/tictoc_svm_poly_fe.rda"))

# collect metrics ---- 
# extract final wflow 
final_wflow <- tune_svm_poly_fe |> 
  extract_workflow(tune_svm_poly_fe) |>  
  finalize_workflow(select_best(tune_svm_poly_fe, metric = "roc_auc"))

# train final model 
set.seed(1234)
final_tune_svm_poly_fe <- fit(final_wflow, train)

submission_test_svm_poly_fe <- bind_cols(test, predict(final_tune_svm_poly_fe, test, type = "prob")) |> 
  mutate(predicted = .pred_TRUE) |> 
  select(id, predicted)

# save submission test ----
write_csv(submission_test_svm_poly_fe, file = here("submissions/submission_test_svm_poly_fe.csv"))
