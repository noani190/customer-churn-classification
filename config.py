
'''
training configurations
'''
DATA_PATH = 'churn_data.csv'
BASE_NAME = 'churn_data'
CPI_FILE_NAME = 'CPI.csv'
METRICS_PATH = 'model_metrics.json'
CPI_RELEVANT_DATA_YEARS = ['2023']

# should the data be scaled
SCALE = False

# for testing purposes
SAVE_LOADER_INTERMEDIATE = False
SAVE_FEATURE_ENGINEER_INTERMEDIATE = False

# rolling windows to be used for feature engineering
ROLLING_AVERAGE_WINDOWS = [3, 6]

# set of features to be used in training and inference
FEATURES = ['transaction_amount', 'Basic', 'Premium', 'Standard',
             'rolling_avg_3m', 'rolling_avg_6m', 'rolling_avg_std_3m', 'rolling_avg_std_6m', 
             'cpi']  #, 'year_percent_change', 'month_percent_change', 'day_sin', 'day_cos']

# threshold date for splitting the data into 
# training and testing (this data and everything following it will be used for testing)
THRESH_DATE_FOR_TEST_SET = '2023-10-01'


'''
inference configurations
'''
MODEL_PATH = 'churn_model.pkl'
INFERENCE_DATA_PATH = 'churn_data.csv'
OUTPUT_PATH = 'predictions.csv'
