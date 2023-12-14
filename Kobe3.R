# Loading in the data 
kobe <- vroom("/Users/braiden/STAT348/R project/kobe-bryant-shot-selection/data.csv")
### Lat and Long ####
#ggplot(data = kobe) +
#  geom_point(mapping = aes(x = lon, y = lat), color = 'blue')
#ggplot(data = kobe) +
#  geom_point(mapping = aes(x = loc_x, y = loc_y), color = 'green')
### Feature Engineering ###
## Converting it to polar coordinates 
dist <- sqrt((kobe$loc_x/10)^2 + (kobe$loc_y/10)^2) 
kobe$shot_distance <- dist
#Creating angle column 
loc_x_zero <- kobe$loc_x == 0
kobe['angle'] <- rep(0,nrow(kobe))
kobe$angle[!loc_x_zero] <- atan(kobe$loc_y[!loc_x_zero] / kobe$loc_x[!loc_x_zero])
kobe$angle[loc_x_zero] <- pi / 2
# Create one time variable 
kobe$time_remaining = (kobe$minutes_remaining*60)+kobe$seconds_remaining
# Home and Away
kobe$matchup = ifelse(str_detect(kobe$matchup, 'vs.'), 'Home', 'Away')
# Season
kobe['season'] <- substr(str_split_fixed(kobe$season, '-',2)[,2],2,2)
### period into a factor
kobe$period <- as.factor(kobe$period)
# delete columns
kobe <- kobe %>%
  select(-c('shot_id', 'team_id', 'team_name', 'shot_zone_range', 'lon', 'lat', 
            'seconds_remaining', 'minutes_remaining', 'game_event_id', 
            'game_id','shot_zone_area',
            'shot_zone_basic', 'loc_x', 'loc_y'))
# Train
train <- kobe %>%
  filter(!is.na(shot_made_flag))
# Test 
test <- kobe %>% 
  filter(is.na(shot_made_flag))
## Make the response variable into a factor 
train$shot_made_flag <- as.factor(train$shot_made_flag)


recipe <- recipe(shot_made_flag ~ ., data = train) %>% 
  step_novel(all_nominal_predictors()) %>%
  step_unknown(all_nominal_predictors()) %>%
  step_dummy(all_nominal_predictors())




bartmodel <- parsnip::bart(mode = "classification",
                           engine = "dbarts",
                           trees = 1500)
bart_wf <- workflow() %>%
  add_recipe(recipe) %>%
  add_model(bartmodel) %>%
  fit(data=train)

bart_preds <- bart_wf %>%
  predict(new_data = test, type = "prob")

bart_predict <- bind_cols(Test_id, bart_preds$.pred_0)
bart_predict <- rename(bart_predict, shot_id = ...1)
bart_predict <- rename(bart_predict, shot_made_flag = ...2)

vroom_write(file = "bart_kobe_rec.csv",
            x = bart_predict,
            delim = ",")


#####################################
#####################################
#####################################
#####################################
#####################################
#####################################
#####################################

log_loss <- function(actual, predicted) {
  eps <- 1e-15  # Small value to avoid log(0)
  predicted <- pmax(pmin(predicted, 1 - eps), eps)  # Ensure predicted values are within [eps, 1 - eps]
  -mean(actual * log(predicted) + (1 - actual) * log(1 - predicted))
}

train$game_date <- NULL
test$game_date <- NULL

boosted_recipe <- recipe(shot_made_flag ~ ., data = train) %>%
  step_lencode_glm(all_nominal_predictors(), outcome = vars(shot_made_flag))

yeah <- bake(prep(boosted_recipe), new_data = train)

boosted_model <- boost_tree(tree_depth= 1, #Determined by random store-item combos
                            trees= 1000,
                            learn_rate= .01) %>%
  set_engine("xgboost") %>%
  set_mode("classification")


boost_wf <- workflow() %>%
  add_recipe(boosted_recipe) %>%
  add_model(boosted_model) %>%
  fit(data = train)

folds <- vfold_cv(KobeTrain, v = 5, repeats = 2)

boost_workflow <- boost_wf %>%
  finalize_workflow(besttuneB) %>%
  fit(data = KobeTrain)

boost_wf <- boost_wf %>%
  fit(data = train)

boost_preds <- boost_wf %>%
  predict(new_data = test, type = "prob")

boost_preds$shot_id <- Test_id
boost_preds <- boost_preds %>%
  select(shot_id, .pred_1) %>%
  rename(shot_made_flag = .pred_1)

vroom_write(file = "boost_kobe_lencode_glm.csv",
            x = boost_preds,
            delim = ",")

