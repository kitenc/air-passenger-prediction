# <center> Final Report - Ramp Challenge</center>
#### <center> predict the number of air passengers </center>

## 1. Introduction

Given the limited datasets about early booking time, departure and arrival information, our goal is to predicting air passengers. Obviously, we need more external datasets to more preciously interpret the target variable('log_PAX'). Many other external factors may affect the actual number of passengers in a specific airline, for example, the weather, the holiday or the weekend, delay time of a specific airline, flights count between two cities and so on. 
As we have been provided the comprehensive details about weather, we try to find more data about airport analysis(total airline operations, departure and arrival situations), city pair analysis (operations, on-time/delays, etc.) and routes arrangements. All data are collected from Federal Aviation Administration (FAA)’s databases.
In additions, we have also considered data about the socioeconomic conditions (city GDP, city or state population/increase, etc.) obtained from open source website and the Federal Reserve Bank (FRED). As season changes and upsurge in flights due to holidays, we have also taken the major public and school holidays into account. Appendix A​ shows the list of data sources.

## 2. Data Preprocessing and Model Selection

Firstly, we merge all external data on 'airport' and 'Date' into a single file. For further regression, We used daily data on air traffic and transformed the departure and destination airports with different prefix. We turn number of year, month,day, weekday and week into a series of 0-1 dummy variables. We compute the next workday, last workday to represent the leisure time and distance with longitude and latitude of each airport to represent the distance. We considered extreme weather indicators which might lead to flight delays or cancellations like minimum temperature, maximum wind, visibility, etc. and change the 'event' column into dummies columns, including thunderstorms and snowstorms. Then, we start the trails of models. We experiment methods, like Multi-regression,Random Forest, neural network and etc.

#### linear regression and Random forest
As respects to Multi-regression, we tried Linear regression, Ridge and Lasso from Scikit-Learn packages, which get automatically feature selection. And the Average RMSE remains above 0.43, which means simple linear regression may not take effect here. 

#### Neural Network
Afterwards, we try the simple neural network. As we know, neural network is mostly used in classification problems. But when we slightly change the final dimension of output layer, we may turn Neural network into non-linear regression method with one output at the final layer to do the regression issue. At the very beginning, we use only one hidden layer and choose 'relu' as activation function and Adam as optimizer. The main parameters are (batch size=64 and loss function=mse). The result is more than 0.5, which acting not pretty well. Afterwards, we add more layers and imcrease the batch size to 128, as small batch-size will decrease the generation error. RMSE remain above 0.4, even though we add more layers and change the parameters. Taking into account the limits of neural network, that we cannot interpret the final models with multiple layer neural networks, we decided to switch to another model.

#### XGBoost
The algorithm is one of the boosting algorithm, which can integrate weak learners into a powerful learner by learning higher-order interactions between features. As every iteration try to fit the residual of the previous k-1 base learners, the newly base learner will move greatly towards the direction of steepest gradient-descent, which lower RMSE for sure.

First, we try the default parameter using XGBoost in Python, where we get RMSE lower than 0.4. Accordingly, we decided to do the feature selection and preprocessing. We change our preprocessing methods to our external data. We add conditions of both and departure and arrival airports to the variables, merging data to the original dataframe. With coordinates in external data, we compute the geographic distance between airports. Because we can not integrate the 'Holidays' into the external data. We get dummies about weekdays and weeks in a year to reflect holidays which may occur in a fixed period. Missing value will use mean or 'ffill'(like for oil price) method to check all missing value.

Secondly, Thanks to sklearn package, we simply use the default estimator of XGBoost to conduct feature ranking with recursive feature elimination and cross-validated selection of the best number of features. We finally get the 56 parameter out of 208 parameter. Luckily, we get a nearly symmetric parameter for departure parameter and arrival parameter. . These included time period (weekday, week, month, days, next work day, last work day and etc.), air travel and airport analysis (Total operations, Actual Departure and Arrival, Delayed Arrival and etc.), socioeconomic factors (GDP, POP, RPI and Death Rate, Birth Rate and population) and geographic attributes (longitude, latitude, elevation, distance).

After doing all these, we reach RMSE(0.3560 +/- 0.0192).

## 3. Parameter Tuning
We first looked for a reasonable ​n_estimator ​with fixed parameter of others. From the previous loss curve, we knew that the XGBoost method rushes to over-fitting, which means we can lower the ​learning_rate​ without incurring a large loss but only to increase the time consumption. So we fix learning rate at 0.07 and max_child weight at 6. We choose n_estimator within the range $[500,2000]$, step size=100 to compute the RMSE. From the computation, we can find that the RMSE will stop reducing when n_estimator is larger than 1700. So we fix n_estimator to 1700.

As to learning rate, we conduct the same things as to n_estimator. We get the optimal learning rate for our cases, which is 0.05.

Finally, we focused on the parameters 'max_depth'​ and '​min_child_weight​'. The former limits the depth of decision trees and the latter prevents an arbitrary node with few samples from being further split. We used the grid search method to get the optimal combination: ​max_depth ​of 6 and ​min_child_weight ​of 3. 

| Regressor type | Score on test set 
|------|------|
|   Lasso Regression  | 0.5482 +/- 0.0144|
|   Random Forest (60 estimators)  |0.4340 +/- 0.0267|
|   Random Forest (300 estimators)| 0.4310 +/- 0.0257|
|   Neural network (batch size=128)| 0.4736 |
|   XGBoost(default parameter)| 0.3560 +/- 0.0192|
|   XGBoost(n_estimator$\geq$1700) |0.3528 +/- 0.0188|
|   XGBoost(n_estimator=1700, learning rate=0.05) |0.3317 +/- 0.0174|
|Final Submission of XGBoost|0.265|

# Appendix A

| Departure parameter | Arrival parameter| Other parameter|
|------|------|------|
|Min Temperature, Total operations, Total Delays, Actual Departures Actual Arrivals Delayed Arrivals, Division StPOP', 'StRBirth', 'StRDeath', 'StRMig','GDP, 'POP','RPI'|Total_ops', Total Delays, Actual Departures', Actual Arrivals', Delayed Arrivals','a_elevation_ft', 'a_Division', 'a_StPOP', 'a_StRBirth', 'a_StRDeath','a_StRMig', 'a_GDP', 'a_POP', 'a_RPI', 'a_Latitude', 'a_longitude'|'nextworkday', 'lastworkday','Distance', 'month', 'day', 'weekday', 'week', 'n_days', 'm_10', 'wd_1', 'wd_3', 'wd_5', 'wd_6', 'w_1', 'w_14','w_20', 'w_21', 'w_27', 'w_35', 'w_47', 'w_52']