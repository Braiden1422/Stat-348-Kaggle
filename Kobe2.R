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
library(xgboost)
library(data.table)
library(Matrix)



Kobe <- vroom("/Users/braiden/STAT348/R project/kobe-bryant-shot-selection/data.csv")
Kobe$season <- as.factor(Kobe$season)
#Kobe['season'] <- substr(str_split_fixed(Kobe$season, '-',2)[,2],2,2)
#Kobe$season <- as.factor(Kobe$season)

Kobe <- Kobe %>%
  mutate_if(is.character, as.factor)

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

dist <- sqrt((train$loc_x/10)^2 + (train$loc_y/10)^2) 
train$shot_distance <- dist
dist2 <- sqrt((test$loc_x/10)^2 + (test$loc_y/10)^2) 
test$shot_distance <- dist2

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
train$playoffs <- NULL;
test$playoffs <- NULL;
train$matchup <- NULL;
test$matchup <- NULL;
train$minutes_remaining <- NULL;
test$minutes_remaining <- NULL;
train$loc_x <- NULL;
test$loc_x <- NULL;
train$loc_y <- NULL;
test$loc_y <- NULL;


################################################
################################################
################################################
################################################
################################################
################################################
################################################

recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_dummy(all_nominal_predictors())

bartmodel <- parsnip::bart(mode = "classification",
                           engine = "dbarts",
                           trees = 300)
bart_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(bartmodel) %>%
  fit(data=train)

bart_preds <- bart_wf %>%
  predict(new_data = test, type = "prob")

bart_preds$shot_id <- Test_id
bart_preds <- bart_preds %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(file = "bart_kobe3_dummy_gameid_300.csv",
            x = bart_preds,
            delim = ",")





