# Credit Default Prediction using XGBoost and Neural Network

**Importance of Credit Default prediction**

In 2022 alone, American Express generated around $53 Billion dollars of revenue and TransUnion forecasted severe credit card delinquencies to rise to 2.6% at the end of 2023 from 2.1% at the close of 2022. By predicting which customers are at the highest risk of defaulting on their credit card accounts, issuers can take proactive steps to minimize risk and exposure.
Now, given the huge amount of data on customers is readily available and the number of signals, it's lucrative to involve Machine Learning algorithms to make predictions for defaults. 

**Data**

Historical Credit Card (CC) data consists of 458,913 customers spread across 13 months on 190 variables in categories like Payment, Spend and Balance
30k – 40k observation each month and % of customers defaulted in each month [23%, 28%]

Target Variable = 1 if the customer default on CC payment
                = 0 if the customer didn’t default

**Features**

All Features are divided into 5 categories: Delinquency, Payment, Balance, Risk & Spend 


**Feature Selection**

Built 2 XGBoost models to rank features according to their feature importance score. 

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/feat_imp-PhotoRoom.png-PhotoRoom.png)


**XGBoost - Grid Search**

The following combinations in the grid search:
1. Number of trees: 50, 100, and 300: 50 to decrease the complexity and the variance, then we tried 300 for lower bias 
2. Learning Rate: 0.01, 0.1:  0.1 – Conventional and 0.01 to validate if slower learning rate arrive at global minima smoothly without overshooting
3. % of observations used in each tree: 50%, 80% - 50% for faster training & 80% to avoid overfitting
4. % of features used in each tree: 50%, 100% - 50% again to avoid overfitting and faster training, and 100% for better results and low bias
5. Weight of default observations: 1, 5, 10 – Since most of our non-default we need weights > 1 

<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/Xgb%20plot1.png width = '500' height = '300'> <img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/XGB%20plot2.png width = '500' height = '300' style="float:right">

Plot 1: Technically, Bias-Variance Tradeoff at X=0.94 & Y = 0.0075 (diff in Y is small, therefore lowest bias preferred) 

Plot 2: Linear relationship between AUC train and test2, therefore highest AUC train preferred

**Final XGBoost Model Parameters**

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/xgb%20final.png)

**Rank Ordering**

Rank ordering here checks if the threshold is increased then we can see that the default keeps increasing for larger threshold brackets

<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/Screenshot%202023-06-11%20at%205.45.52%20PM.png width = '500' height = '300'>


**SHAP Analysis**

BeeSwarm - Explains the cumulative impact of features on model 

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/beeswarm.png)

P_2 higher values drive the score down meaning higher the payment variable lower will be probability of default

Most features increase their  impact on model output with higher feature value


Waterfall - Explains prediction for specific observation  
![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/waterfall.png)
Expected Model Output = -1.308, Output for 1100th customer = -4.311

P_2 singlehandedly drives prediction down by 1.26 whereas 37 other features collectively drive it down by 1.17


**Neural Network**

**NN Grid Search**

Combination of Hyper-Parameters in the grid search:
1. Number of hidden layers: 2, 4 – 4 to increase the complexity to get low bias and 2 for faster runtime 
2. nodes in each hidden layer: 4, 6 – 2 for simple neural network and 6 for complex neural network 
3. Activation function: ReLu, Tanh – ReLu isn’t saturated/zero-centered, tanh causes vanishing gradients 
4. Dropout regularization: 50%, 100% (no dropout) – 50% to decrease complexity and avoid overfitting
5. Batch size: 100, 10000 – 100 not low enough to overfit every batch and 10000 for faster processing 


<img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/NN%20Plot1.png width = '500' height = '300'> <img src = https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/NN%20Plot%202.png width = '500' height = '300' style="float:right">

**Final Model**

![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/final%20model.png)


Bias – Winner XGBoost 

Variance – Winner Neural Network (diff in Std Dev is negligible) 

Explanability – Winner XGBoost ( SHAP Analysis) 


**Strategy**

The conservative strategy has a lower threshold compared with aggressive one; hence accepts less applicants.


![image](https://github.com/Shritej24c/Credit-Risk/blob/main/Graphs/ex%20strategy%20final.png)

0.5 – Aggressive strategy because we want to increase our Revenue while maintaining the default rate below 10%

0.3 – Conservative Strategy because the default decreases almost by half but revenue isn’t drastically affected






