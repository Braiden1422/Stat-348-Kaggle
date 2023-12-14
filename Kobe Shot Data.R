install.packages("VBsparsePCA")
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
library(kernlab)
library(themis)
library(skimr)
library(keras)
library(timetk)
library(Hmisc)
library(VBsparsePCA)
library(embed)
library(bonsai)
library(lightgbm)
library(xgboost)


####. Goal of under .60205 log loss ###
# Thus far the BEST MODEL is a
#Random Forest classifier with the following parameters
#{'bootstrap': True, 'criterion': 'gini', 'max_depth': 8, 'max_features': 20, 'n_estimators': 100} which gives out a log loss of
#~ 0.612


##########################
###########################
##########################

Kobe <- vroom("/Users/braiden/STAT348/R project/kobe-bryant-shot-selection/data.csv")



#adding a distance variable
dist <- sqrt((Kobe$loc_x/10)^2 + (Kobe$loc_y/10)^2) 
Kobe$shot_distance <- dist

#adding an angle variable
loc_x_zero <- Kobe$loc_x == 0
Kobe['angle'] <- rep(0,nrow(Kobe))
Kobe$angle[!loc_x_zero] <- atan(Kobe$loc_y[!loc_x_zero] / Kobe$loc_x[!loc_x_zero])
Kobe$angle[loc_x_zero] <- pi / 2
Kobe$time_remaining = (Kobe$minutes_remaining*60)+Kobe$seconds_remaining

#Creating a home/away variable
Kobe$matchup <- ifelse(grepl("@", Kobe$matchup), 0,1)


Kobe['season'] <- substr(str_split_fixed(Kobe$season, '-',2)[,2],2,2)
Kobe$season <- as.factor(Kobe$season)

Kobe$period <- as.factor(Kobe$period)

Kobe <- Kobe %>%
  select(-c('team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id', 'game_date','shot_zone_area',
            'loc_x', 'loc_y', -"matchup", -"period", -"shot_zone_area"))

KobeTrain <- Kobe %>%
  filter(shot_made_flag == 1 | shot_made_flag == 0) %>%
  mutate(shot_made_flag = as.factor(shot_made_flag)) 

KobeTest <- Kobe %>%
  filter(is.na(shot_made_flag)) %>%
  select(-shot_made_flag)

##########################
##########################
##########################


KobeTrain %>%
  mutate(make = as.numeric(shot_made_flag)) %>%
  keep(is.numeric) %>%
  cor() %>%
  corrplot::corrplot()

ggplot(data = KobeTrain) +
  geom_point(mapping = aes(x = lon, y = lat), color = 'navy')

ggplot(data = KobeTrain) +
  geom_point(mapping = aes(x = loc_x, y = loc_y), color = 'maroon')






##########################
##########################
##########################
kobe_recipe <- recipe(shot_made_flag ~ ., data = KobeTrain) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_pca_sparse_bayes() 
 # step_mutate(as.factor(minutes_remaining, period, playoffs, season))

baked<- bake(prep(kobe_recipe), new_data = KobeTrain)

log_loss <- function(actual, predicted) {
  eps <- 1e-15  # Small value to avoid log(0)
  predicted <- pmax(pmin(predicted, 1 - eps), eps)  # Ensure predicted values are within [eps, 1 - eps]
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}


##########################
##########################
##########################

forest_model <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")

kobe_workflow <- workflow() %>%
  add_recipe(kobe_recipe) %>%
  add_model(forest_model)

tuning_grid <- grid_regular(mtry(range = c(1,ncol(KobeTrain)-1)),
                            min_n(),
                            levels = 2)

folds <- vfold_cv(KobeTrain, v = 5, repeats = 2)

CV_results <- kobe_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

besttune <- CV_results%>%
  select_best("roc_auc")

forest_workflow <- kobe_workflow %>%
  finalize_workflow(besttune) %>%
  fit(data = KobeTrain)

forest_preds <- forest_workflow %>%
  predict(new_data = KobeTest, type = "class")

forest_predictions <- forest_preds %>%
  bind_cols(.,KobeTest) %>%
  select(shot_id,.pred_class) %>%
  rename(shot_made_flag = .pred_class)

vroom_write(file = "kobe_preds.csv",
            x = forest_predictions,
            delim = ",")



##########################
##########################
##########################
# tuned trees = 2000 depth =1 learn rate = .01


boosted_recipe <- recipe(shot_made_flag ~ ., data = KobeTrain) %>% 
  step_dummy(all_nominal_predictors())

boosted_model <- boost_tree(tree_depth= 1, #Determined by random store-item combos
                            trees= 2000,
                            learn_rate= .01) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model)

folds <- vfold_cv(KobeTrain, v = 5, repeats = 2)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3)

CV_resultsB <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))


 besttuneB <- CV_resultsB%>%
  select_best("roc_auc")

boost_workflow <- boost_wf %>%
  finalize_workflow(besttuneB) %>%
  fit(data = KobeTrain)

boost_wf <- boost_wf %>%
  fit(data = KobeTrain)

boost_preds <- boost_wf %>%
  predict(new_data = KobeTest, type = "prob")

boost_predictions <- boost_preds %>%
  bind_cols(.,KobeTest) %>%
  select(shot_id,.pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(file = "kobe_preds_boost3.csv",
            x = boost_predictions,
            delim = ",")

##########################
##########################
##########################




boosted_recipe <- recipe(shot_made_flag ~ ., data = KobeTrain) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_bayes(all_nominal_predictors(), outcome = vars(shot_made_flag))

boosted_model <- boost_tree(tree_depth= tune(), #Determined by random store-item combos
                            trees= tune(),
                            learn_rate= tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model)

folds <- vfold_cv(KobeTrain, v = 5, repeats = 2)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 3)

CV_results <- boost_wf %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(roc_auc))

besttune <- CV_results%>%
  select_best("roc_auc")

boost_wf <- boost_wf %>%
  finalize_workflow(besttune) %>%
  fit(data = KobeTrain)

boost_preds <- boost_wf %>%
  predict(new_data = KobeTest, type = "prob")

boost_predictions <- boost_preds %>%
  bind_cols(.,KobeTest) %>%
  select(shot_id,.pred_0) %>%
  rename(shot_made_flag = .pred_0)

vroom_write(file = "kobe_preds_boost_mixed.csv",
            x = boost_predictions,
            delim = ",")

##########################
##########################
##########################
preg_recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag))

preg_model <- linear_reg(mode = "classification" ,penalty= 1, mixture= 1) %>%
  set_engine("glmnet")

preg_wf <- workflow() %>%
  add_recipe(preg_recipe) %>%
  add_model(preg_model) %>%
  fit(data = KobeTrain)

preg_preds <- preg_wf %>%
  predict(new_data = KobeTest, type = "numeric")


#tuning_grid <- grid_regular(penalty(),
#                            mixture(),
#                            levels = 2)

#folds <- vfold_cv(bikedata, v = 2, repeats=1)

#PREG_results <- preg_wf %>%
#  tune_grid(resamples = folds,
#            grid = tuning_grid,
#            metrics = metrics_set("rmse"))


preg_preds <- preg_wf %>%
  predict(new_data = KobeTest, type = "linear_pre")

preg_predictions <- preg_preds %>%
  bind_cols(.,KobeTest) %>%
  select(shot_id,.pred_0) %>%
  rename(shot_made_flag = .pred_0)

##########################
##########################
##########################
##########################
##########################
##########################

#{'bootstrap': True, 'criterion': 'gini', 'max_depth': 8, 'max_features': 20, 'n_estimators': 100} which gives out a log loss of
#~ 0.612


RF_recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag))

rf_spec <- rand_forest(mode = "classification", mtry = 20,trees = 100, min_n = 8) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <-workflow() %>%
  add_recipe(RF_recipe) %>%
  add_model(rf_spec) %>%
  fit(data = train)

rf_preds <- rf_workflow %>%
  predict(new_data = test, type = "prob")

rf_preds$shot_id <- Test_id
rf_preds <- rf_preds %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(file = "kobe_preds_rf_1.csv",
            x = rf_preds,
            delim = ",")

##########################
##########################
##########################
##########################
##########################
##########################





Kobe <- vroom("/Users/braiden/STAT348/R project/kobe-bryant-shot-selection/data.csv")

Kobe['season'] <- substr(str_split_fixed(Kobe$season, '-',2)[,2],2,2)

train <- Kobe %>%
  filter(shot_made_flag == 1 | shot_made_flag == 0) %>%
  mutate(shot_made_flag = as.factor(shot_made_flag)) 

test <- Kobe %>%
  filter(is.na(shot_made_flag)) %>%
  select(-shot_made_flag)

Test_id <- test$shot_id

train$shot_id <- NULL;
test$shot_id <- NULL;

train$time_remaining <- train$minutes_remaining*60+train$seconds_remaining;
test$time_remaining <- test$minutes_remaining*60+test$seconds_remaining;

train$shot_distance[train$shot_distance>45] <- 45;
test$shot_distance[test$shot_distance>45] <- 45;

train$game_date<-NULL;
test$game_date<-NULL;
train$seconds_remaining<-NULL;
test$seconds_remaining<-NULL;
train$team_name <- NULL;
test$team_name <- NULL;
train$team_id <- NULL;
test$team_id <- NULL;
train$game_event_id <- NULL;
test$game_event_id <- NULL;
train$game_id <- NULL;
test$game_id <- NULL;
train$lat <- NULL;
test$lat <- NULL;
train$lon <- NULL;
test$lon <- NULL;

xg_recipe <- recipe(shot_made_flag ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors())

xg_model <- boost_tree(tree_depth= 3, #Determined by random store-item combos
                            trees= 1500,
                            learn_rate= .01) %>%
  set_engine("xgboost") %>%
  set_mode("classification")

#folds <- vfold_cv(train, v = 4, repeats = 2)

#tuning_grid <- grid_regular(tree_depth = 3,
#                            trees = 1500,
#                            learn_rate = .01,
#                            levels = 3)

#xg_wf <- workflow() %>%
#  add_recipe(xg_recipe) %>%
#  add_model(xg_model) %>%
#  fit(data = train)

#XG_results <- xg_wf %>%
#  tune_grid(resamples=folds,
#            grid=tuning_grid,
#            metrics=metric_set(roc_auc))

#besttune <- XG_results %>%
#  select_best("roc_auc")

#XG_wf <- xg_wf %>%
#  finalize_workflow(besttune) %>%
#  fit(data = train)


xg_wf <- workflow() %>%
  add_recipe(xg_recipe) %>%
  add_model(xg_model) %>%
  fit(data = train)


xg_preds <- xg_wf %>%
  predict(new_data = test, type = "prob")

xg_predict <- bind_cols(Test_id, xg_preds$.pred_0)
xg_predict <- rename(xg_predict, shot_id = ...1)
xg_predict <- rename(xg_predict, shot_made_flag = ...2)


vroom_write(file = "xg_kobe_novel_unknown.csv",
            x = xg_predict,
            delim = ",")
