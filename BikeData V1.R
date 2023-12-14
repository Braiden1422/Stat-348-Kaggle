library(tidyverse)
library(vroom)

bikedata <- vroom("/Users/braiden/STAT348/R project/train.csv")
GGally::ggpairs(bikedata)
plot1 <- DataExplorer::plot_intro(bikedata)
plot2 <- DataExplorer::plot_correlation(bikedata)
plot3 <- DataExplorer::plot_missing(bikedata)
plot4 <- plot(bikedata)
