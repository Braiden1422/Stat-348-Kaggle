#install.packages("timetk")
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
library(modeltime)
library(plotly)
library(bonsai)

#### score needed is 15.3

itemDtest <- vroom("/Users/braiden/STAT348/R project/Time series/demand-forecasting-kernels-only/test.csv")
itemDtrain <- vroom("/Users/braiden/STAT348/R project/Time series/demand-forecasting-kernels-only/train.csv")

itemDtest <- itemDtest %>%
  filter(store == 4 , item == 1)

itemDtrain <- itemDtrain %>%
  filter(store == 4 , item == 1)

#my_recipe <- recipe(sales ~., data = itemDtrain) %>% 
 # step_date(date, features = "dow") %>%
  #step_date(date, features = "doy") %>%
  #step_range(date_doy, min=0, max=pi) %>%
  #step_mutate(sinDOY=sin(date_doy), cosDOY=cos(date_doy))
my_recipe <- recipe(sales ~., data = itemDtrain) %>%
  step_timeseries_signature(date) %>%
  step_normalize(all_numeric_predictors())
  

baked <- bake(prep(item_recipe), itemDtrain)  
 

my_forest_mod <- rand_forest(mtry = tune(),
                             min_n=tune(),
                             trees=1000) %>%
  set_engine("ranger") %>%
  set_mode("regression")

item_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(my_forest_mod) %>%
  fit(data = itemDtrain)


tuning_grid <- grid_regular(mtry(range = c(1,ncol(itemDtrain)-1)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(itemDtrain, v = 4, repeats=2)


CV_results <- item_workflow %>%
  tune_grid(resamples=folds,
            grid=tuning_grid,
            metrics=metric_set(smape))

bestTune <- CV_results %>%
  select_best("smape")

collect_metrics(CV_results) %>%
  filter(mtry == 3, min_n == 40) %>%
  pull(mean)


forest_workflow <- item_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = itemDtrain)


forest_preds <- forest_workflow %>%
  predict(new_data = itemDtest)

#####################################
cv_split <- time_series_split(itemDtrain, assess="3 months", cumulative = TRUE)



prophet_model <- prophet_reg() %>%
set_engine(engine = "prophet") %>%
fit(sales ~ date, data = training(cv_split))


## Cross-validate to tune model5
cv_results <- modeltime_calibrate(prophet_model,
                                  new_data = testing(cv_split))



prophet_preds <- prophet_model %>%
  predict(new_data = itemDtest)

p3 <- cv_split %>%
  tk_time_series_cv_plan() %>% #Put into a data frame
  plot_time_series_cv_plan(date, sales, .interactive=FALSE)


## Visualize CV results
p4 <- cv_results %>%
modeltime_forecast(
                   new_data = testing(cv_split),
                   actual_data = itemDtrain) %>%
plot_modeltime_forecast(.interactive=TRUE)


subplot(p1,p3,p2,p4, nrows = 2)



##############################
##############################
##############################
##############################
item_recipe <- recipe(sales ~., data = itemDtrain) %>%
  step_timeseries_signature(date) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(sales))


boosted_model <- boost_tree(tree_depth=2, #Determined by random store-item combos
                            trees=1000,
                            learn_rate=0.01) %>%
  set_engine("lightgbm") %>%
  set_mode("regression")

boost_wf <- workflow() %>%
  add_recipe(item_recipe) %>%
  add_model(boosted_model)

n.stores <- max(itemDtrain$store)
n.items <- max(itemDtrain$item)

for(s in 1:n.stores){
  for(i in 1:n.items){
    
    ## Subset the data
    train <- itemDtrain %>%
      filter(store==s, item==i)
    test <- itemDtest %>%
      filter(store==s, item==i)
    
    ## Fit the data and forecast
    fitted_wf <- boost_wf %>%
      fit(data=itemDtrain)
    preds <- predict(fitted_wf, new_data=itemDtest) %>%
      bind_cols(itemDtest) %>%
      rename(sales=.pred) %>%
      select(id, sales)
    
    ## Save the results
    if(s==1 && i==1){
      all_preds <- preds
    } else {
      all_preds <- bind_rows(all_preds,
                             preds)
    }
    
  }
}








