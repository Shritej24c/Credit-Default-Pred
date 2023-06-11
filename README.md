# Credit Default Prediction using XGBoost and Neural Network

**Importance of Credit Default prediction**

In 2022 alone, American Express generated around $53 Billion dollars of revenue and it's only source of revenue is credit cards. Moreover, TransUnion forecasted severe credit card delinquencies to rise to 2.6% at the end of 2023 from 2.1% at the close of 2022. By predicting which customers are at the highest risk of defaulting on their credit card accounts, issuers can take proactive steps to minimize risk and exposure.

Now, given the huge amount of data on customers is readily available and the number of signals, it's lucrative to involve Machine Learning algorithms to make predictions for defaults. 

**Data**

Historical Credit Card (CC) data consists of 458,913 customers spread across 13 months on 190 variables in categories like Payment, Spend and Balance
30k – 40k observation each month and % of customers defaulted in each month [23%, 28%]

Target Variable = 1 if customer default on CC payment
      = 0 if customer didn’t default

**Features**

All Features divided into 5 categories: Delinquency, Payment, Balance, Risk & Spend 

<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/total%20features.png width="200" height="225" style="float:right">

**Sampling**

Test1 and Test2 before and after the training period to maintain randomness of unseen data and not create bias due to time period

Default Rate increases with Time

<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/sampling.png width="700" height="225" style="float:right">

**Feature Selection**

Built 2 XGBoost models to rank features according to their feature importance score. 

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/feat_imp-PhotoRoom.png-PhotoRoom.png)


**XGBoost - Grid Search **

The following combinations in the grid search:
1. Number of trees: 50, 100, and 300 : 50 to decrease the complexity and the variance, then we tried 300 for lower bias 
2. Learning Rate: 0.01, 0.1 :  0.1 – Conventional and 0.01 to validate if slower learning rate arrive at global minima smoothly without overshooting
3. % of observations used in each tree: 50%, 80% - 50% for faster training & 80% to avoid overfitting
4. % of features used in each tree: 50%, 100% - 50% again to avoid overfitting and faster training and 100% for better results and low bias
5. Weight of default observations: 1, 5, 10 – Since most of are non-default we need weights > 1 

<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/Xgb%20plot1%20(1).png width = '500' height = '225'>
<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/XGB%20plot2%20(2).png width = '500' height = '225' style="float:right">

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/Xgb%20plot1%20(1).png)
