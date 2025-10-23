## Classification Prediction Problem ----
## EDA of target variable ----

# load packages ---- 
library(tidyverse)
library(tidymodels)
library(here)

# handle common conflicts 
tidymodels_prefer()

# load data ----
load(here("data/train.rda"))

# check missingness 
train  |> 
  skimr::skim_without_charts(host_is_superhost) # no missingness 

# inspect target variable (host_is_superhost) ----
host_distribution <- train  |> 
  ggplot(aes(host_is_superhost)) + 
  geom_bar()

# save ----
save(host_distribution, file = here("figures_tables/host_distribution.rda"))
