## Predicting Customer Churn

This repository contains an implementation of a model for predicting customer churn classification.

### Data and Assumptions:
The original data contains synthetically generated information about customers, including indices for both customer and date of transaction, type of plan they are assigned to and whether or not they churned.
I decided I will disregard the fact that a customer had already been marked as churned at month X, and will still use month X+1 - my reasoning is that in the given data, the customer still passes transactions even in the months following the month they are labelled as churned. Therefore, my assumption here is that they get a '1' when the customer started inclining towards leaving but not necessarily leaving yet.

### Chosen Model, Input, and Output:
I've chosen to use Random Forest model for the following reasons:
Tree-based models may handle small amount of data well, they allow handling non-linear relationships,
the training time is faster than RNNs (e.g. LSTM) for this amount of data, trees-based model allow easier interpretability (using Gini index, SHAP, etc..).
I've also decided to treat each *customer and month* as a separate entry, rather than aggregating the customer to a single row of information, this will mean that if I'd like to make a future prediction based on this data (e.g. whether the customer will churn on Jan 24), I'll need to assess the final output aggregation methods, this is out of the current mini-project scope, and so I will provide a per-month per-customer prediction.
The data used for training are month 1-9 of 2023, months 10-12 were used for testing.

### Possible Fututre Improvements:
- try different models (e.g. LSTM, XGBoost).
- use a validation set to test different models, and hyperparameters.
- fully use some of the customers for test set (and not just months 9-12).
- handle imbalanced class sampling using SMOTE for example.

### Files:
1. model_runner.py - a python file that's used for running the training of the model and includes calls for all relevant classes and functions. 
2. data_loader.py - contains a class that loads and preprocesses input data (e.g. imputing missing values and encoding 'plan type').
3. feature_engineer.py - contains a class that adds input features to model.
4. model_trainer.py - contains a class that trains the model using RandomForest classifier.
5. model_inference.py - given a model, a data path, and an output path (to be set using the config file) makes predictions (1- churn, 0- otherwise).
6. config.py - contains the variables needed for running the model both in training and inference.
7. utils.py - contains utility functions that do not belong to any speicfic class. 
8. eda.py - contains a (very) basic exploratory data analysis.
9. CPI.csv - contains information about the US Consumer Price Index per month, added to enrich the data with external sources (source: https://www2.nhes.nh.gov/GraniteStats/SessionServlet?page=CPI.jsp&SID=5&country=000000&countryName=United%20States)
10. churn_model.pkl - pickle file containing the output selected model.
11. churn_data.csv - original data file, containing the raw data and the target value (churn 1/0).
12. predictions.csv - original churn_data.csv file, that has additional columns for features and a prediction column.
13. visualizations - contains several plots created using eda.py and model_trainer.py (SHAP-related).
14. shap_values.csv - calculated shap values for chosen features.
15. model_metrics.json - file containing model assessment metrics
16. requirements.txt - a requirement file that sets up that necessary environment.
17. README.md - this file.

### Setup:
```
pip install -r requirements.txt
```

Usage:
On terminal / command line:

Training script:
```
python -m model_runner
```
Inference script:
```
python -m model_inference
```
EDA script:
```
python -m eda
```

### Output:
Can be found in predictions.csv
Note that the output still needs to be post-processed based on the model goal (see Chosen 'Model, Input, and Output' section for more info).
