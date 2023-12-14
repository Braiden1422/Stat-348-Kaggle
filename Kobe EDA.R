kobe <- vroom("/Users/braiden/STAT348/R project/kobe-bryant-shot-selection/data.csv")

library(tidyverse)
library(hexbin)
library(jsonlite)
library(httr)
library(scales)

train <- kobe %>%
  filter(!is.na(shot_made_flag))

train$shot_made_flag <- as.factor(train$shot_made_flag)

min_loc_x <- min(train$loc_x)
max_loc_x <- max(train$loc_x)

train <- train %>%
  mutate(Baseline = rescale(loc_x, to = c(-25, 25)))

train <- train %>%
  mutate(Sideline = rescale(loc_y, to = c(0, 45)))

trainseasons <- train %>%
  group_by(season) %>%
  summarise(PercentMade = mean(shot_made_flag))

ggplot(data = trainseasons, mapping = aes(x = season, y = PercentMade)) +
  geom_point() +
  geom_line(aes(group = 1)) +
  ylab("Field Goal Percentage") +
  xlab("Season") +
  ggtitle("Field Goal Percentage By Season") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))


train <- train %>%
  mutate(Shot = ifelse(train$shot_made_flag == 1, "Make", "Miss"))


ggplot(data = train) +
  geom_point(mapping = aes(x = Baseline, y = Sideline, color = Shot, group = Shot)) +
  theme_classic() +
  theme(axis.text = element_blank(), axis.title = element_blank(),
    axis.line = element_blank(),  axis.ticks = element_blank()) +
  scale_color_manual(values = c("Make" = "#76d54f", "Miss" = "#c34540")) 


trainteams <- train %>%
  group_by(opponent) %>%
  summarise(PercentMade = mean(shot_made_flag))



ggplot(data = trainteams, mapping = aes(x = opponent, y = PercentMade)) +
  geom_point() +
  geom_line(aes(group = 1)) +
  ylab("Field Goal Percentage") +
  xlab("Opponent") +
  ggtitle("Field Goal Percentage vs Opponent") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))








