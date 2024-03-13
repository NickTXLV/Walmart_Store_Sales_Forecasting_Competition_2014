# Walmart Store Sales Forecasting Competition 2014

# Objectives:

The purpose of this workflow is to demonstrate my ability to take a public data set and turn it
into a forecast. The data set that I will be using in the Walmart Recruiting - Store Sales Forecasting competition from 2014.

Credit: 
  Walmart Competition Admin, Will Cukierski. (2014). Walmart Recruiting - Store Sales Forecasting. 
  Kaggle. https://kaggle.com/competitions/walmart-recruiting-store-sales-forecasting

I will not be following the competition exactly as it was specified. Instead, I will create a forecast based on overall sales at the Chain Level. We will forecast 12 weeks out

# In this workflow, I will demonstrate the following:
 Data Wrangling/Munging/Manipulation
 
 Understanding the data 
 
 Preparing the Data for Forecasting
 
 Create the Forecast
 
 Measure the Forecast Accuracy using MAE


 # Results:
 I created a total of 3 different types of forecasts, each has a base model and a hyper parameter tuned model.

 The accuracy of each model:

 ![image](https://github.com/NickTXLV/Walmart_Store_Sales_Forecasting_Competition_2014/assets/84823331/dc380868-d7b8-4a88-be7a-6cb1b4c460d8)


 When using MAE as the metric:

 The base ARIMA model performed the best at .39 followed closely by the hyper turned ARIMA model at .4. The non-sequential base models all performed  the worst, however their hyper tuned counterparts performed well with a MAE of approx .43.

 Key Learnings:


Given the ARIMA models performed the best, there could be an opportunity to model seasonality better in the non-sequential models. This could come with better calendar feature modeling or focus on trend modeling to better capture seasonality. I only did 2 rounds of hyper-parameter tuning, focusing more on this could also yield better results. 

 # Forecast:


![image](https://github.com/NickTXLV/Walmart_Store_Sales_Forecasting_Competition_2014/assets/84823331/c77cef31-a37e-4044-bd41-a8917f8b8330)


The base GLMNET model is underfitting. It doesn't seem to pick up on the demand shifts very well. However, notice the performance increase in the hyper tuned GLMNET model. This seems to pick up more on the peaks and valleys of demand. 
 

