# Classification Prediction Problem ----
# Setup preprocessing/recipes/feature engineering (kitchen sink) ---

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts
tidymodels_prefer()

# load training data ----
load(here("data/train.rda"))

# ks recipe ----
# build recipe 
recipe_ks <- recipe(host_is_superhost ~ ., data = train) |> 
  step_rm(id) |> 
  step_impute_median(all_numeric_predictors()) |> 
  step_impute_mode(all_nominal_predictors()) |>
  step_other(threshold = 0.05) |>
  step_dummy(all_nominal_predictors(), one_hot = TRUE) |>
  step_nzv(all_numeric_predictors()) |>
  step_normalize(all_numeric_predictors())

# check recipe 
recipe_ks |> 
  prep() |> 
  bake(new_data = NULL) |> 
  glimpse()

# save recipe ----
save(recipe_ks, file = here("recipes/recipe_ks.rda"))