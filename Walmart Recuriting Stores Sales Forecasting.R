
# Objectives ----

#The purpose of this workflow is to demonstrate my ability to take a public data set and turn it
#into a forecast. The data set that I will be using in the Walmart Recruiting - Store Sales Forecasting competition from 2014.
#Credit: 
  #Walmart Competition Admin, Will Cukierski. (2014). Walmart Recruiting - Store Sales Forecasting. 
  #Kaggle. https://kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

#I will not be following the competition exactly as it was specified. 
#Instead, I will create a forecast based on overall sales at the Chain Level. We will forecast 12 weeks out

#In this workflow, I will demonstrate the following:
# Data Wrangling/Munging/Manipulation
# Understanding the data 
# preparing the Data for Forecasting
# Create the Forecast
# Measure the Forecast Accuracy using MAPE




# Read in Libraries ----

library(tidymodels)
library(tidyverse)

library(modeltime)
library(timetk)

library(fastDummies)

library(DataExplorer)

library(plotly)


# Get Data ----

#The data comes from the Walmart Recruiting - Store Sales Forecasting competition from 2014.
# More info about it can be found at the following: https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting/data
# The data will be loaded from the Package TimeTK

data("walmart_sales_weekly")

df<-walmart_sales_weekly

write.csv(df,"C:\\Users\\dodso\\OneDrive - Texas Tech University\\Portfolio Builder\\Forecasting\\data.csv")


# Perform Data Wrangling ----

#We only care about sales at the Chain level, so we'll need to perform data wrangling to reflect this

df %>% glimpse()

summary_tbl <-df %>% 
  group_by(Date) %>%
  summarise(
    Total_Weekly_sales = sum(Weekly_Sales),
    AVG_Temperature = mean(Temperature),
    AVG_Fuel_Price = mean(Fuel_Price),
    AVG_CPI = mean(CPI),
    AVG_Unemployment_Rate = mean(Unemployment),
    Total_MD = sum(across(MarkDown1:MarkDown5, .fns = sum))
    ) %>%
  mutate(
    Total_MD_Rate = Total_MD/Total_Weekly_sales
  ) %>%
  ungroup() %>%
  # The week starts falls on a Friday, restate to start the week on a Sunday
  mutate(Date = Date-5)


## Perform EDA ----

# Lets See how many NAs we have

count_na <- colSums(is.na(summary_tbl))

print(count_na)


#Perform EDA
  
create_report(data = summary_tbl,
              output_file = "report.html",
              output_format = html_document(),
              y="Total_Weekly_sales",
              output_dir = "C:\\Users\\dodso\\OneDrive - Texas Tech University\\Portfolio Builder\\Forecasting\\"
            )



# There are a lot of missing Values for the MD columns. This is as expected since it was stated in the data set.
# However, since there are so many missing values, we will drop these columns since it will negatively impact ML models.

summary_tbl <- summary_tbl %>%
  select(-c(Total_MD,Total_MD_Rate))

summary_tbl


# Visualize ----

## Plot the Time Series ----


summary_tbl %>%
  plot_time_series(.date_var = Date,
                   .value=Total_Weekly_sales,
                   .smooth_period = "180 days",
                   .smooth_degree = 0,
                   .plotly_slider  = TRUE)



## Plot ACF Plot ----

summary_tbl %>%
  plot_acf_diagnostics(.date_var = Date,
                       .value=  Total_Weekly_sales)

## Plot CCF Plot ----


# Visualize for Potential Lagged Regressors in other variables

summary_tbl %>%
  plot_acf_diagnostics(.date_var = Date,
                       .value = Total_Weekly_sales,
                       .ccf_vars = AVG_Temperature:AVG_Unemployment_Rate,
                       .show_ccf_vars_only = TRUE)

## Plot Seasonality ----

summary_tbl %>%
  plot_seasonal_diagnostics(.date_var = Date,
                            .value = Total_Weekly_sales)


## Seasonal Decomposition

?plot.decomposed.ts
?plot_stl_diagnostics

summary_tbl %>%
  plot_stl_diagnostics(.date_var = Date,
                       .value= Total_Weekly_sales)



# Anomoaly Detection ----

## Plot The Anomaly ----
summary_tbl %>%
  plot_anomaly_diagnostics(.date_var = Date,
                           .value= Total_Weekly_sales,
                           .alpha=.2,
                           .max_anomalies = .1)

## Create The Anomaly Table ----

# This will be used later to ensure that the anomaly doesn't fall on weeks with holidays

anomaly_tbl <-summary_tbl %>%
  tk_anomaly_diagnostics(.date_var = Date,
                         .value= Total_Weekly_sales,
                         .alpha=.2,
                         .max_anomalies = .1
  ) %>% 
  mutate(Anomaly_Flag = ifelse(
                                anomaly =="Yes",1,0)
                        )


# Feature Engineering ----
# We will incorporate what we learned visualizing the time series to conduct featuring engineering to model time


## Model Time ----

### Holidays By Week ----

#Since the time series is by week, we will have to build a daily dataset, find the holidays,
# then group be week

weekly_holiday <- tk_make_timeseries(start_date = "2010",
                                    end_date = "2015",
                                    by = "day") %>%
  # Build Holidays for each date
  tk_get_holiday_signature(holiday_pattern = "US_",
                           locale_set = "US",
                           exchange_set = "NYSE") %>% 

  #Group by Week
  summarise_by_time(.date_var = index,
                    .by= "week",
                    Isholiday=rowSums(across(exch_NYSE:US_ChristmasDay),na.rm = FALSE),
                    New_Year_Day = rowSums(across(US_NewYearsDay:US_NewYearsDay),na.rm = FALSE),
                    US_MemorialDay = rowSums(across(US_MemorialDay:US_MemorialDay),na.rm = FALSE),
                    US_IndependenceDay = rowSums(across(US_IndependenceDay:US_IndependenceDay),na.rm = FALSE),
                    US_LaborDay = rowSums(across(US_LaborDay:US_LaborDay),na.rm = FALSE),
                    US_ThanksgivingDay = rowSums(across(US_ThanksgivingDay:US_ThanksgivingDay),na.rm = FALSE),
                    US_ChristmasDay = rowSums(across(US_ChristmasDay:US_ChristmasDay),na.rm = FALSE)
                    ) %>%
  filter(Isholiday >=1) %>%
  ungroup() %>%
  mutate(Isholiday = if_else(Isholiday >=1,1,0)) %>%
  mutate(holiday=rowSums(across(New_Year_Day:US_ChristmasDay))) %>%
  arrange(index, desc(holiday)) %>%
  distinct(index,.keep_all = TRUE) %>%
  select(-holiday)
  
### Model the Week ----

time_series_sig<-tk_augment_timeseries_signature(summary_tbl) %>% 
  #We will only select month/week since that was the most important in the seasonal diagnostcs
  select(Date,month.lbl,week)


# Create Final Data set ----

#Merge the sales data to the Anomaly, Holidays & Time Modeling


dataset_tbl<- summary_tbl %>%
  #Anomaly
  left_join(y=anomaly_tbl[ , c("Date","Anomaly_Flag")],
            by= join_by(Date)) %>%
  #Holidays
  left_join(y=weekly_holiday,
            by= join_by(Date==index)) %>%
  #Time Modeling
  left_join(y=time_series_sig,
            by = join_by(Date)) %>%
# Fix NA
  replace_na(list(Isholiday= 0,
                  New_Year_Day=0,
                  US_MemorialDay=0,
                  US_IndependenceDay=0,
                  US_LaborDay=0,
                  US_ThanksgivingDay=0,
                  US_ChristmasDay=0))
  
  
# PreProcess the  Final Data set ----

#Replace Anomalies if its not part of a holiday with the 3 periods around the anomaly

preprocess_tbl<-dataset_tbl %>%
  mutate(Lag_3_Weeks_avg = slidify_vec(.x=Total_Weekly_sales,
                                       .f=mean,
                                       .period=3,
                                       .align="center",
                                       .partial= TRUE
                                       )
         ) %>%
  #Replace Anomaly ----
  mutate(Total_Weekly_sales = if_else(Anomaly_Flag ==1 & Isholiday ==0,Lag_3_Weeks_avg,Total_Weekly_sales )) %>%
  select(-Lag_3_Weeks_avg) %>%
  
  #Scale the data ----
  mutate(Total_Weekly_sales=standardize_vec(Total_Weekly_sales)) %>%
  mutate(AVG_Temperature=standardize_vec(AVG_Temperature)) %>%
  mutate(AVG_Fuel_Price=standardize_vec(AVG_Fuel_Price)) %>%
  mutate(AVG_CPI=standardize_vec(AVG_CPI)) %>%
  mutate(AVG_Unemployment_Rate=standardize_vec(AVG_Unemployment_Rate)) %>%
  
  
  
  #Create lags based on learning from the PACF plot ----

  tk_augment_lags(.value=Total_Weekly_sales,
                  .lags = c(4,9,13,52)) %>%
  
  # Turn Months into Dummy Variables ----
  #dummy_cols(select_columns = "month.lbl") %>%
  select(-c("Anomaly_Flag","month.lbl"))
  
# Capture the mean and standard deviation so you can un-scale the data
Total_Weekly_sales_mean = 382325.824731935
Total_Weekly_sales_sd = 26090.8422431553
AVG_Temperature_mean =68.3067832167832
AVG_Temperature_sd =14.2504864063958
AVG_Fuel_Price_mean =3.2196993006993
AVG_Fuel_Price_sd =0.427312516090926
AVG_CPI_mean =215.996891766434
AVG_CPI_sd =4.35089008300814
AVG_Unemployment_Rate_mean =7.61041958041958
AVG_Unemployment_Rate_sd =0.383748825067475



## Time Series Regression Plot ----

preprocess_tbl %>%
  #select(-Total_Weekly_sales_lag52) %>%
  plot_time_series_regression(.date_var = Date,
                              .formula = (Total_Weekly_sales) ~ .,
                              .show_summary = TRUE)



# ****Key Learnings**** ----
#Processing the data to add additional lagged features to model time, holidays and lags
#has produced an R-Squared of .57! Even though the p-values are high, we will keep these features
# and let the ML models remove them if they are not helpful in predicting weekly sales



#Build the Full Data Set including the future dates to forecast ----

data_prepared_full_tbl <- preprocess_tbl %>%
  select (c(Date,Total_Weekly_sales,AVG_Temperature,AVG_Fuel_Price,AVG_CPI, 
            AVG_Unemployment_Rate)) %>%
  # Add Future Dates
  bind_rows(
    future_frame(.data = .,
                 .date_var = Date,
                 .length_out = 12)
           ) %>%
  #Add in the data for the regressors into the future frame
  tk_augment_lags(.value = c(AVG_Temperature,AVG_Fuel_Price,AVG_CPI,AVG_Unemployment_Rate),
                  .lags = 52) %>%
  mutate(AVG_Temperature = ifelse(is.na(AVG_Temperature),AVG_Temperature_lag52,AVG_Temperature)) %>%
  mutate(AVG_Fuel_Price = ifelse(is.na(AVG_Fuel_Price),AVG_Fuel_Price_lag52,AVG_Fuel_Price)) %>%
  mutate(AVG_CPI= ifelse(is.na(AVG_CPI),AVG_CPI_lag52,AVG_CPI)) %>%
  mutate(AVG_Unemployment_Rate =ifelse(is.na(AVG_Unemployment_Rate),AVG_Unemployment_Rate_lag52,AVG_Unemployment_Rate)) %>%
  select(-c(contains("lag"))) %>%
  # Add in holidays
  left_join(y=weekly_holiday,
            by = join_by(Date == index)) %>%
  #Add Calendar Features
    tk_augment_timeseries_signature(.data =.,
                                  .date_var = Date) %>%
  select(Date,Total_Weekly_sales,AVG_Temperature,AVG_Fuel_Price,AVG_CPI,AVG_Unemployment_Rate,
         Isholiday,New_Year_Day,US_MemorialDay,US_IndependenceDay,US_LaborDay,US_ThanksgivingDay,
         US_ChristmasDay,month.lbl,week) %>%
  
  #Add Lags 
  tk_augment_lags(.value= Total_Weekly_sales,.lags = c(13,52)) %>%
  mutate(across(Isholiday:US_ChristmasDay, ~ifelse(is.na(.), 0, .))) %>%
  #Month to Dummy Variables
  dummy_cols(select_columns = "month.lbl",
             remove_selected_columns = TRUE) %>%
  
  #Remove NA 
  
  filter(!is.na(Total_Weekly_sales_lag52))
  
 
data_prepared_full_tbl %>% glimpse()

  #After Dummy variables, just need to seperate out the future data frame then 
  
  
  #splitt the data into testing and training

data_prepared_full_tbl %>%
  plot_time_series_regression(.date_var = Date,
                              .formula = (Total_Weekly_sales) ~ .,
                              .show_summary = TRUE)
  




#for the below Refer to 02_advanced_feature_engineering line 75
#Add future windows extend the data timeseries into the forecast horizon 
# Add the external regressiors to the future data frame
#remove thte future data frame as an obect to use to preict on later


# Seperate out future table from base data ----

data_prepared_full_tbl_recipe_stemps_rm <- data_prepared_full_tbl %>%
  select(-c(contains("month"))) %>%
  select(-week)

data_prepared_full_tbl_recipe_stemps_rm %>% glimpse()

processed_tbl <- data_prepared_full_tbl_recipe_stemps_rm %>%
  filter(!is.na(Total_Weekly_sales))

forecast_tbl <- data_prepared_full_tbl_recipe_stemps_rm %>%
  filter(is.na(Total_Weekly_sales))

forecast_tbl %>% glimpse()



# Split the Processed data into training/testing ----

#Training Data will have 12 weeks, which is the same as the forecast horizon


split<- time_series_split(data = processed_tbl,
                          date_var = Date,
                          assess = "12 weeks",
                          cumulative = TRUE)

# View the Training and Testing Data
split %>%
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(.date_var = Date,
                           .value = Total_Weekly_sales)

# Create the Recipe Spec ----

# This is basically a recreation of the processed_tbl, but put into a Recipe Specification,
# so that it's easier to work with and saves time

# We will have 2 Receipes that are almost the same, only difference is the date feature

## Recipe Date ----
recipe_date <- recipe(Total_Weekly_sales ~.,
                  data = training(split)) %>%
  step_timeseries_signature(Date) %>%
  step_select(Total_Weekly_sales,Date,AVG_Temperature, AVG_Fuel_Price , AVG_CPI , AVG_Unemployment_Rate,
              Isholiday , New_Year_Day,US_MemorialDay ,  US_IndependenceDay , 
              US_LaborDay,US_ThanksgivingDay ,US_ChristmasDay,Total_Weekly_sales_lag13,
              Total_Weekly_sales_lag52,Date_month.lbl,Date_week) %>%
  step_dummy(Date_month.lbl, one_hot= TRUE)
  

recipe_date %>% prep() %>% juice() %>% glimpse()


## Recipe No Date ----
recipe_no_date <- recipe(Total_Weekly_sales ~.,
                      data = training(split)) %>%
  step_timeseries_signature(Date) %>%
  step_select(Total_Weekly_sales,AVG_Temperature, AVG_Fuel_Price , AVG_CPI , AVG_Unemployment_Rate,
              Isholiday , New_Year_Day,US_MemorialDay ,  US_IndependenceDay , 
              US_LaborDay,US_ThanksgivingDay ,US_ChristmasDay,Total_Weekly_sales_lag13,
              Total_Weekly_sales_lag52,Total_Weekly_sales,Date_month.lbl,Date_week) %>%
  step_dummy(Date_month.lbl, one_hot= TRUE)


recipe_no_date %>% prep() %>% juice() %>% glimpse()



# Create ML Models ----

#Create the Base Model
#Hyperparameter tune the model



#GLMNET/XG BOOST/Arima  Boost
#Sequential Models: ARIMA

# GLMNET ----
#The GLM NET model will be our base model that will serve as a benchmark for all other models
# GLM NET is a combination of Lasso/Ridge Regression


## Build the Base Model----

model_base_glmnet <- linear_reg(
  penalty = .5,
  mixture =.5
) %>%
  set_engine("glmnet")


# Build a workflow to fit the base model
workflow_fit_base_glmnet <- workflow() %>%
  add_recipe(recipe_no_date) %>%
  add_model(model_base_glmnet) %>%
  fit(training(split))


#Add the model to a modeltime table
model_table <- modeltime_table(
  workflow_fit_base_glmnet
) %>%
  update_model_description(1,"Base GLMNET")


# Calibrate the model

calibration_tbl <- model_table %>%
  modeltime_calibrate(new_data = testing(split))


# calibration_tbl %>%
#   slice(1) %>%
#   unnest(.calibration_data)  

#Test the Accuracy
calibration_tbl %>%
  modeltime_accuracy()


## Hyperparameter Tuning ----

#Initial Setup for Non Sequential Models
set.seed(123)
resamples_kfold <- vfold_cv(
  data=training(split),
  v=10
)

resamples_kfold %>% 
  tk_time_series_cv_plan() %>%
  plot_time_series_cv_plan(Date, Total_Weekly_sales, .facet_ncol = 2)

#Setup GLMNET for tuning

model_hyper_glmnet <- linear_reg(
  penalty = tune(),
  mixture = tune()
) %>%
  set_engine("glmnet")



grid_tuning_glmnet <- grid_latin_hypercube(
  parameters(model_hyper_glmnet) %>%
    update(
      penalty = penalty(c(-3,0)),
      mixture = mixture(c(.6,.65))
    ),
  size=10
)



##  K-Fold Cross Validation

set.seed(123)

hyper_results_kfold_glmnet <-workflow_fit_base_glmnet %>%
  update_model(model_hyper_glmnet) %>%
  tune_grid(
    resamples = resamples_kfold,
    grid= grid_tuning_glmnet,
    metrics =default_forecast_accuracy_metric_set(),
    control = control_grid(save_pred = TRUE)
  )



# Show Best

hyper_results_kfold_glmnet %>%
  show_best(metric ="mae", n = Inf)


# The below will Graph the result to help understand where to focus the tune results on

g<- hyper_results_kfold_glmnet %>%
  autoplot()+
  geom_smooth(se= FALSE)

ggplotly(g)


# Select the best Tuned Model



# Train the Model and Assess the hypertuned inputs


workflow_fit_hyper_glmnet <- workflow_fit_base_glmnet %>%
  update_model(model_hyper_glmnet) %>%
  finalize_workflow(
    hyper_results_kfold_glmnet %>%
      show_best((metric = "mae"),n=1)
    ) %>% 
  fit(training(split))


 ## Final GLMNET Models ----
model_table_glmnet <- modeltime_table(
  workflow_fit_base_glmnet,
  workflow_fit_hyper_glmnet
  ) %>%
  update_model_description(1,"Base GLMNET") %>%
  update_model_description(2,"Hyper GLMNET") 


calibration_glmnet_tbl <- model_table_glmnet %>%
  modeltime_calibrate(new_data = testing(split))

calibration_glmnet_tbl %>%
  modeltime_accuracy()



# Forecast GLMNET Models----
forecast_glmnet_tbl <- calibration_glmnet_tbl %>%
  modeltime_forecast(
    new_data= testing(split),
    actual_data = processed_tbl
  )

forecast_glmnet_tbl %>%
  plot_modeltime_forecast()


# XGBOOST ----


## Build the Base Model----


model_base_xgboost <- boost_tree(
  mode= "regression",
  tree_depth = 6,
  trees = 15,
  learn_rate = .3,
  mtry= 15,
  min_n=1,
  loss_reduction = 0
) %>%
  set_engine("xgboost")



workflow_fit_base_xgboost <- workflow() %>%
  add_model(model_base_xgboost) %>%
  add_recipe(recipe_no_date) %>%
  fit(training(split))



model_table_base_xgboost <- modeltime_table(workflow_fit_base_xgboost)


calibration_tbl_xgboost <- model_table_base_xgboost %>%
  modeltime_calibrate(new_data = testing(split))


calibration_tbl_xgboost %>%
  modeltime_accuracy()






## Hyperparamter Tuning ----


## Forecast XGBOOST Models----




# ARIMA Boost ----

recipe_spec_arima <-recipe(Total_Weekly_sales ~Date , data = training(split)) %>%
  step_fourier(optin_time, period = c(7,14,30,90), K=1)

model_spec_arima<-arima_reg() %>%
  set_engine("auto_arima")


workflow_fit_arima<-workflow() %>%
  add_recipe(recipe_spec_arima) %>%
  add_model(model_spec_arima) %>%
  fit(training(split))



modeltime_






