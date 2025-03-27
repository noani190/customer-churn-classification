import pandas as pd
import time

from config import BASE_NAME, SAVE_LOADER_INTERMEDIATE
from utils import save_intermediate_df

class DataLoader:
    """
    Handles loading and preprocessing of time series data.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
    
    def load_data(self):
        """Loads data from a CSV file."""
        self.data = pd.read_csv(self.file_path)
        return self.data
    
    def preprocess(self):
        """Handles missing values and feature engineering."""
        for column in self.data.columns:
            if column == 'transaction_amount':
                self.data[column].fillna(self.data[column].mean(), inplace=True)
            elif column == 'plan_type':
                self.data[column].fillna(self.data[column].mode(), inplace=True)
        '''
        one hot encoding of plan type
        '''
        plan_type_encoding = pd.get_dummies(self.data.plan_type)
        self.data = self.data.drop('plan_type',axis = 1)
        # Join the encoded df
        self.data = self.data.join(plan_type_encoding)

        if SAVE_LOADER_INTERMEDIATE:
            save_intermediate_df(self.data, 'data_loader')

        return self.data
    