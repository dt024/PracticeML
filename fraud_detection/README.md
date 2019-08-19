# What I gain:
## Using LightGBM for classification problem.
## How to deal with similar columns in dataset:

Have to carefully check them:
For example, checking V319 and V320:
- Create a diff_V319_V320 column:	df.loc[df[“V319”]!=df[“V320”], “diff_V319_V320”] = 1
- Explore the result in each group: df.groupby(“diff_V319_V320”).mean().isFraud
We find that transaction which has different V319-V320 is doubtful.
If you think this column is useful, use them.
https://www.kaggle.com/yasagure/how-do-we-treat-with-similar-columns-v319-v321


## Data Exploring:
https://www.kaggle.com/artgor/eda-and-models
- Train and test transaction dates don't overlap, so it would be prudent to use time-based split for validation


## Data leakage:
- When the data you are using to train a machine learning algorithm happens to have the information you are trying to predict.
- Happens a lot when dealing with time-series data
- One of the rules of thumb for trying to avoid Data Leakage for time series data is performing cross validation.


## Day Forward-Chaining (Time Series cross-validator)
-we successively consider each day as the test set and assign all previous data into the training set.
-TimeSeriesSplit: This cross-validation object is a variation of KFold. In the kth split, it returns first k folds as train set and the (k+1)th fold as test set.



## Day and Time - powerful predictive feature
https://www.kaggle.com/fchmiel/day-and-time-powerful-predictive-feature
Time of day has a strong dependence on whether the transaction is fraudulent and will likely make a good feature.
How to tune parameters for light gbm, catboost and xgboost in classification tasks:
bayesian optimization with this package: https://github.com/fmfn/BayesianOptimization



## How you define "top features"
At first I train LGBM on all features and then take some of the features with the highest feature importance.


## Reduce memory usage:
https://www.kaggle.com/gemartin/load-data-reduce-memory-usage


## Questions: Don't we need to standardize the data by using StandardScaler method? also shouldn't we check the correlation between features so we can drop more columns?
https://www.kaggle.com/artgor/eda-and-models
While standardizing data for linear models is required, for tree-based models it is mostly useless. Tree-based models make splits between values of variables, so exact values matter little. The difference could be if we limit the number of bins to analyze.
The same for correlation. It is important to analyze it for linear models, but for tree-based models feature interactions are important. While a feature could have little correlation with target, it could have a serious effect, when combined with some other variable.


##Domain names can be binned:
yahoo / ymail / frontier / rocketmail -> Yahoo
hotmail / outlook / live / msn -> Microsoft
icloud / mac / me -> Appe
prodigy / att / sbcglobal-> AT&T
centurylink / embarqmail / q -> Centurylink
aim / aol -> AOL
twc / charter -> Spectrum

