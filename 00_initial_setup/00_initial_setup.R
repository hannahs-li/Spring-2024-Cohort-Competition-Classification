## Classification Prediction Problem ----
## Clean, split, and fold data ----

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)


# handle common conflicts 
tidymodels_prefer()

# load data ----
train_classification <- read_csv("data/train_classification.csv",
                                 col_types = cols(id = col_character()))

train <- train_classification |> 
  janitor::clean_names() |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(across(where(is.logical), as.factor)) |> 
  mutate_at(c('host_since', 'first_review', 'last_review'), as.Date, format = '%Y-%m-%d') |> 
  mutate(host_since = as.numeric(format(host_since,'%Y')),
         first_review = as.numeric(format(first_review,'%Y')),
         last_review = as.numeric(format(last_review,'%Y'))) |> 
  mutate(id = as.character(id)) |> 
  mutate(host_response_rate = as.numeric(host_response_rate)) |> 
  mutate(host_acceptance_rate = as.numeric(host_acceptance_rate)) |> 
  mutate(minimum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_nights_avg_ntm = pmin(minimum_maximum_nights, 1125))

# convert bathrooms_text to bathrooms 
train <- train |> 
  # select(bathrooms_text) |> 
  mutate(bathrooms_text = as.character(bathrooms_text)) |> 
  mutate(bathrooms = gsub("\\bbaths?\\b", "", bathrooms_text)) |> 
  mutate(bathrooms_v2 = gsub("\\bprivate\\b", "", bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bHalf-\\b", "0.5", bathrooms_v2)) |> 
  mutate(bathrooms_v2 = gsub("\\bShared half-\\b", "0.25", bathrooms_v2)) |> 
  mutate(bathrooms_v2 = gsub("\\bPrivate half-\\b", "0.5", bathrooms_v2)) |> 
  mutate(shared_bathrooms = grepl('shared', bathrooms_v2)) |> 
  mutate(shared_bathrooms = factor(shared_bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bshared\\b", "", bathrooms_v2)) |> 
  mutate(true_bathrooms = as.numeric(bathrooms_v2)) |> 
  select(-c(bathrooms_text, bathrooms_v2, bathrooms)) 


# clean test ----
test_classification <- read_csv("data/test_classification.csv",
                                col_types = cols(id = col_character()))

test <- test_classification  |> 
  janitor::clean_names() |> 
  mutate(across(where(is.character), as.factor)) |> 
  mutate(across(where(is.logical), as.factor)) |> 
  mutate_at(c('host_since', 'first_review', 'last_review'), as.Date, format = '%Y-%m-%d') |> 
  mutate(host_since = as.numeric(format(host_since,'%Y')),
         first_review = as.numeric(format(first_review,'%Y')),
         last_review = as.numeric(format(last_review,'%Y'))) |> 
  mutate(id = as.character(id)) |> 
  mutate(host_response_rate = as.numeric(host_response_rate)) |> 
  mutate(host_acceptance_rate = as.numeric(host_acceptance_rate)) |> 
  mutate(minimum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_maximum_nights = pmin(minimum_maximum_nights, 1125)) |> 
  mutate(maximum_nights_avg_ntm = pmin(minimum_maximum_nights, 1125))

# convert bathrooms_text to bathrooms 
test <- test |> 
  # select(bathrooms_text) |> 
  mutate(bathrooms_text = as.character(bathrooms_text)) |> 
  mutate(bathrooms = gsub("\\bbaths?\\b", "", bathrooms_text)) |> 
  mutate(bathrooms_v2 = gsub("\\bprivate\\b", "", bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bHalf-\\b", "0.5", bathrooms_v2)) |> 
  mutate(bathrooms_v2 = gsub("\\bShared half-\\b", "0.25", bathrooms_v2)) |> 
  mutate(bathrooms_v2 = gsub("\\bPrivate half-\\b", "0.5", bathrooms_v2)) |> 
  mutate(shared_bathrooms = grepl('shared', bathrooms_v2)) |> 
  mutate(shared_bathrooms = factor(shared_bathrooms)) |> 
  mutate(bathrooms_v2 = gsub("\\bshared\\b", "", bathrooms_v2)) |> 
  mutate(true_bathrooms = as.numeric(bathrooms_v2)) |> 
  select(-c(bathrooms_text, bathrooms_v2, bathrooms)) 

# initial split and fold ----
# set seed 
set.seed(1234)

# fold
folds <- train |> 
  vfold_cv(v = 5, repeats = 3, strata = host_is_superhost)

# save ----
save(folds, file = here("data/folds.rda"))
save(train, file = here("data/train.rda"))
save(test, file = here("data/test.rda"))
