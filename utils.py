import time

from config import BASE_NAME

def save_intermediate_df(df, source='data_loader'):
    """Saves intermediate dataframes to CSV files."""
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    BACKUP_NAME = f'{BASE_NAME}_{source}_{timestamp}.csv'
    df.to_csv(BACKUP_NAME)
    