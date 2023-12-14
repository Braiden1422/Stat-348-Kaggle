#install.packages("naivebayes")
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
library(dplyr)
library(tidybayes)
library(ggplot2)
library(tidybayes)
library(embed)
library(discrim)
library(naivebayes)
#step_other(all_nom_pred, thresh = .001)
#penalty first mixture (between 0 and 1) second

amazontest <- vroom("/Users/braiden/STAT348/R project/amazon-employee-access-challenge/test.csv")
amazontrain <- vroom("/Users/braiden/STAT348/R project/amazon-employee-access-challenge/train.csv")

amazontrain$ACTION <- as.factor(amazontrain$ACTION)


##########my_recipe <- recipe(ACTION ~., data= amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) 



my_recipe <- recipe(ACTION~., data= amazontrain) %>%
step_mutate_at(all_numeric_predictors(), fn = factor) %>% 
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION))
  #step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%# turn all numeric features into factors5
# combines categorical values that occur <5% into an "other" value6# dummy variable encoding7
  
#Maybe take out dummy or lencode mixed



prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazontrain)


my_mod <- logistic_reg() %>% #Type of model3
  set_engine("glm")



amazon_workflow <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(my_mod) %>%
fit(data = amazontrain) # Fit the workflow9

amazon_predictions <- predict(amazon_workflow,
                              new_data=amazontest,
                              type= "prob")


log_preds <- amazon_predictions %>%
  bind_cols(.,amazontest) %>%
  select(id,.pred_1) %>%
  rename(Action = .pred_1)


vroom_write(file = "log_preds.csv",
            x = log_preds,
            delim = ",")


#########
#########
#########


my_forest_mod <- rand_forest(mtry = tune(),
            min_n=tune(),
            trees=500) %>%
          set_engine("ranger") %>%
          set_mode("classification")


amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_mod) %>%
  fit(data = amazontrain)

tuning_grid <- grid_regular(mtry(range = c(1,ncol(amazontrain)-1)),
                           min_n(),
                           levels = 2)


folds <- vfold_cv(amazontrain, v = 5, repeats=1)

CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc)) #Or leave metrics NULL


bestTune <- CV_results %>%
  select_best("roc_auc")

forest_workflow <- amazon_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = amazontrain)

forest_preds <- forest_workflow %>%
  predict(new_data = amazontest, type = "prob")

forest_predictions <- forest_preds %>%
  bind_cols(.,amazontest) %>%
  select(id,.pred_1) %>%
  rename(Action = .pred_1)

vroom_write(file = "forest_preds.csv",
            x = forest_predictions,
            delim = ",")


##################
##################
##################
my_recipe <- recipe(ACTION~., data = amazontrain) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .0001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())
  
prep <- prep(my_recipe)
baked <- bake(prep, new_data = amazontrain)
  
nb_model <- naive_Bayes(Laplace=tune(), smoothness=tune()) %>%
set_mode("classification") %>%
set_engine("naivebayes")

amazon_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                           smoothness(),
                           levels = 2)


folds <- vfold_cv(amazontrain, v = 5, repeats = 2)

CV_results <- amazon_workflow %>%
tune_grid(resamples=folds,
          grid=tuning_grid,
          metrics=metric_set(roc_auc))

besttune <- CV_results%>%
  select_best("roc_auc")

final_workflow <- amazon_workflow %>%
  finalize_workflow(besttune) %>%
  fit(data = amazontrain)

nb_preds <- final_workflow %>%
  predict(new_data = amazontest, type = "prob")

nb_predictions <- nb_preds %>%
  bind_cols(.,amazontest) %>%
  select(id,.pred_1) %>%
  rename(Action = .pred_1)

vroom_write(file = "nb_preds.csv",
            x = nb_predictions,
            delim = ",")

######################
######################
######################
######################


## knn model
knn_model <- nearest_neighbor(neighbors=10) %>% # set or tune
  set_mode("classification") %>%
set_engine("kknn")

knn_wf <- workflow() %>%
add_recipe(my_recipe) %>%
add_model(knn_model)

final_workflow <- knn_wf %>%
  fit(data = amazontrain)


knn_preds <- final_workflow %>%
  predict(new_data = amazontest, type = "prob")

knn_predictions <- knn_preds %>%
  bind_cols(.,amazontest) %>%
  select(id,.pred_1) %>%
  rename(Action = .pred_1)

vroom_write(file = "knn_preds.csv",
            x = knn_predictions,
            delim = ",")





