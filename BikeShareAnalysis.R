install.packages("tidytreatment", dependencies = TRUE, type = "source")
library(rlang)
library(tidymodels)
library(tidyverse)
library(vroom)
library(lubridate)
library(rpart)
library(ranger)
library(stacks)
library(agua)
library(h2o)
library(dbarts)
library(BART)
library(tidytreatment)
library(dplyr)
library(tidybayes)
library(ggplot2)

bikedata <- vroom("/Users/braiden/STAT348/R project/train.csv")
testdata <- vroom("/Users/braiden/STAT348/R project/test.csv")

bikedata$weather[bikedata$weather == 4] <- 3

bikedata <- bikedata[,c(-10,-11)]


Bikedatarecipe <- recipe(count ~., bikedata) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_date(datetime, features="dow", label = TRUE) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_rm()

prepped_recipe <- prep(Bikedatarecipe) # Sets up the preprocessing using myDataS12

bikedata <- bake(prepped_recipe, new_data= bikedata)



Biketestdatarecipe <- recipe(testdata) %>%
  step_date(datetime, features="dow", label = TRUE)

testprepped_recipe <- prep(Biketestdatarecipe,retain = TRUE) # Sets up the preprocessing using myDataS12

testdata <- bake(testprepped_recipe, new_data= testdata)



my_model <- linear_reg() %>%
  set_engine("lm")

bike_workflow <- workflow() %>%
  add_recipe(Bikedatarecipe) %>%
  add_model(my_model) %>%
  fit(data = bikedata)

bike_predictions <- predict(bike_workflow,
                            new_data = testdata)




bike_predictions$count[bike_predictions$count < 0] <- 0

bike_predictions[,2] <- bike_predictions[,1]
bike_predictions[,1] <- testdata$datetime

colnames(bike_predictions)[1] <- "datetime"
colnames(bike_predictions)[2] <- "count" 

vroom_write(bike_predictions,delim = ",", "treepredictions.csv")

mutate(datetime=as.character(format(datetime)))

bike_predictions$datetime <- as_datetime(bike_predictions$datetime, tz = "America/New_York")
bike_predictions$datetime <- format(bike_predictions$datetime, format = "%Y-%m-%d %H:%M:%S")

write.csv(bike_predictions, "try 5.csv", row.names = FALSE)




preg_model <- linear_reg(penalty=.75, mixture=.34) %>% #Set model and tuning
  set_engine("glmnet")


preg_wf <- workflow() %>%
add_recipe(Bikedatarecipe) %>%
add_model(preg_model) %>%
fit(data=bikedata)


preg_model <- linear_reg(penalty = tune(),
                         mixture = tune()) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
add_recipe(prepped_recipe) %>%
add_model(preg_model)

#### This is where the thing goes
tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 3)

folds <- vfold_cv(bikedata, v = 2, repeats=1)

CV_results <- preg_wf %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(rmse, mae, rsq))


show_notes(.Last.tune.result)



my_mod <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model6
  set_engine("rpart") %>% # Engine = What R function to use7
  set_mode("regression")

tree_wf <- workflow() %>%
  add_recipe(Bikedatarecipe) %>%
  add_model(my_mod) %>%
  fit(data=bikedata)

bike_predictions <- predict(tree_wf,
                            new_data = testdata)

test_preds <- predict(tree_wf, new_data = testdata) %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) #needed for right format to Kaggle
## Write prediction file to CSV
vroom_write(x=test_preds, file="foresttest.csv", delim=",") 

my_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>% 
                      set_engine("ranger") %>% 
                      set_mode("regression")
##############################
##############################
##############################
##############################
##############################
##############################
bikedata <- select(bikedata, -casual, -registered) %>%
  mutate(count = log(count))

bike_recipe <- recipe(count~.,data = bikedata) %>%
  step_mutate(weather = ifelse(weather==4,3,weather)) %>%
  step_mutate(weather = factor(weather, levels = 1:3)) %>%
  step_mutate(season = factor(season, levels = 1:4)) %>%
  step_mutate(holiday = factor(holiday,levels=c(0,1))) %>%
  step_mutate(workingday = factor(workingday,levels=c(0,1))) %>%
  step_time(datetime, features = "hour") %>%
  step_rm(datetime) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_numeric_predictors())

prepped_recipe <- prep(bike_recipe)  

bake(prepped_recipe,new_data = bikedata)
bake(prepped_recipe,new_data = testdata)




folds <- vfold_cv(bikedata, v = 10)




untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()


lin_model <- linear_reg() %>%
  set_engine("lm")

linreg_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(lin_model)

linreg_folds_fit <- linreg_wf %>%
  fit_resamples(resamples = folds,
                control = tunedModel)


reg_tree <- decision_tree(tree_depth = tune(),
                        cost_complexity = tune(),
                        min_n=tune()) %>% #Type of model6
  set_engine("rpart") %>% # Engine = What R function to use7
  set_mode("regression")

regtree_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(reg_tree)


regTree_tunegrid <- grid_regular(tree_depth(),
                                 cost_complexity(),
                                 min_n(),
                                 levels = 5)

tree_folds_fit <- regtree_wf %>%
  tune_grid(resamples = folds,
            grid = regTree_tunegrid,
            metrics = metric_set(rmse),
            control = untunedModel)


bike_stack <- stacks() %>%
  add_candidates(linreg_folds_fit) %>%
  add_candidates(tree_folds_fit)


fitted_bike_stack <- bike_stack %>%
  blend_predictions() %>%
  fit_members()

collect_parameters(fitted_bike_stack, "tree_folds_fit")

stacked_predictions <- predict(fitted_bike_stack,new_data = testdata) %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

vroom_write(x=stacked_predictions, file="stackedpreds.csv", delim=",") 


library(parsnip)
#maybe use bart


automodel <- auto_ml(bikedata) %>%  
  set_engine("h2o") %>% 
  set_mode("regression") %>% 
  translate()

auto_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(automodel) %>%
  fit(data=bikedata)

stacked_predictions <- predict(auto_wf,new_data = testdata) %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime)))

bikedata<-as.data.frame(bikedata)












bartmodel <- parsnip::bart(mode = "regression",
                           engine = "dbarts",
                           trees = 20)
  
  
bart_wf <- workflow() %>%
  add_recipe(bike_recipe) %>%
  add_model(bartmodel) %>%
  fit(data=bikedata)

bart_predictions <- predict(bart_wf,new_data = testdata) %>%
  mutate(.pred=exp(.pred)) %>%
  bind_cols(., testdata) %>% #Bind predictions with test data
  select(datetime, .pred) %>% #Just keep datetime and predictions
  rename(count=.pred) %>% #rename pred to count (for submission to Kaggle)
  mutate(count=pmax(0, count)) %>% #pointwise max of (0, prediction)
  mutate(datetime=as.character(format(datetime))) 


vroom_write(x=bart_predictions, file="bartpreds.csv", delim=",")
