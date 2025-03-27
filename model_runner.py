import os
import pandas as pd
from sklearn.model_selection import train_test_split

from data_loader import DataLoader
from feature_engineer import FeatureEngineer
from model_trainer import ModelTrainer

from config import DATA_PATH, THRESH_DATE_FOR_TEST_SET, MODEL_PATH

if __name__ == "__main__":
    # Load data
    data_loader = DataLoader(DATA_PATH)
    data = data_loader.load_data()
    data = data_loader.preprocess()
    
    # Feature extraction
    fe = FeatureEngineer(data)
    data = fe.extract_features()
    
    # Split data
    data['date'] = pd.to_datetime(data['date'])
    train_df = data[data['date'] < THRESH_DATE_FOR_TEST_SET]
    test_df = data[(data['date'] >= THRESH_DATE_FOR_TEST_SET)]
    print(train_df.shape)
    print(test_df.shape)
    X_train = train_df.drop(columns=['churn', 'date', 'customer_id'])
    y_train = train_df['churn']
    X_test = test_df.drop(columns=['churn', 'date', 'customer_id'])
    y_test = test_df['churn']
    
    # Train model
    trainer = ModelTrainer()
    print(X_train)
    trainer.train(X_train, y_train)
    trainer.evaluate(X_test, y_test)
    trainer.plot_feature_importance(X_train)
    
    # Save model
    trainer.save_model(MODEL_PATH)
