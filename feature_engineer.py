import pandas as pd
import numpy as np

from utils import save_intermediate_df
from config import (SAVE_FEATURE_ENGINEER_INTERMEDIATE, ROLLING_AVERAGE_WINDOWS,
                     FEATURES, CPI_FILE_NAME, CPI_RELEVANT_DATA_YEARS)

class FeatureEngineer:
    """
    Extracts relevant features from time series data.
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def extract_features(self):
        """Extracts and returns feature matrix and target variable."""
        '''
        extract month data as a numeric
        '''
        self.data['month'] = self.data['date'].str.split('-').str[1].apply(pd.to_numeric)
        self.data['date'] = pd.to_datetime(self.data['date'])  # Ensure date is datetime
        self.data['day_of_year'] = self.data['date'].dt.dayofyear
        '''
        add day of year feature (includes a circular representation)
        '''
        # Apply sinusoidal encoding
        self.data['day_sin'] = np.sin(2 * np.pi * self.data['day_of_year'] / 365)
        self.data['day_cos'] = np.cos(2 * np.pi * self.data['day_of_year'] / 365)

        '''
        add running average feature for transaction amount
        '''
        for i in ROLLING_AVERAGE_WINDOWS:
            self.data[f'rolling_avg_{i}m'] = self.data.groupby(
                'customer_id')['transaction_amount'].transform(lambda x: x.rolling(i, min_periods=1).mean())
            self.data[f'rolling_avg_std_{i}m'] = self.data.groupby(
                'customer_id')['transaction_amount'].transform(lambda x: x.rolling(i, min_periods=1).std()).fillna(method='bfill')
        '''
        data enrichment with consumer price index 
        source: https://www2.nhes.nh.gov/GraniteStats/SessionServlet?page=CPI.jsp&SID=5&country=000000&countryName=United%20States
        '''
        cpi = pd.read_csv(CPI_FILE_NAME, skiprows=1)
        cpi = cpi[cpi.Year.isin(CPI_RELEVANT_DATA_YEARS)]
        cpi.columns = ['year', 'period', 'cpi', 'year_percent_change', 'month_percent_change']
        cpi['month'] = pd.to_datetime(cpi.period, format='%B').dt.month
        self.data = pd.merge(self.data, cpi)

        if SAVE_FEATURE_ENGINEER_INTERMEDIATE:
            save_intermediate_df(self.data, 'feature_engineer')

        return self.data[['customer_id', 'date'] + FEATURES + ['churn']]
    