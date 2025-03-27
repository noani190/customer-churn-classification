import pandas as pd
import numpy as np
import joblib
from feature_engineer import FeatureEngineer
from config import MODEL_PATH, INFERENCE_DATA_PATH, OUTPUT_PATH, SCALE
from data_loader import DataLoader

class Inference:
    """
    Handles model inference for predicting customer churn.
    """
    def __init__(self, model_path: str):
        # Load trained model and scaler
        self.model, self.scaler = joblib.load(MODEL_PATH)

    def predict(self, data_path: str, output_path: str):
        """
        Reads new data, extracts features, makes predictions, and saves output.
        """
        # Load data
        data_loader = DataLoader(data_path)
        data = data_loader.load_data()
        data = data_loader.preprocess()
        
        # Feature extraction
        fe = FeatureEngineer(data)
        data = fe.extract_features()
        X = data.drop(columns=['customer_id', 'churn', 'date'])
        
        # Apply the same scaler used during training
        if SCALE:
            X = self.scaler.transform(X)

        # Make predictions
        predictions = self.model.predict(X)

        # Append predictions to data
        data['churn_prediction'] = predictions

        # Save output CSV
        data.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

if __name__ == "__main__":
    inference = Inference(MODEL_PATH)  # Load trained model
    inference.predict(INFERENCE_DATA_PATH, OUTPUT_PATH)  # Run inference
