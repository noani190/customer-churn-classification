import joblib
import shap
import json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, precision_score, recall_score, f1_score)
from config import SCALE, METRICS_PATH

class ModelTrainer:
    """
    Handles model training and evaluation.
    """
    def __init__(self, model=None):
        self.model = model if model else RandomForestClassifier()
        self.scaler = StandardScaler()
    
    def train(self, X_train, y_train):
        """Trains the model."""
        if SCALE:
            X_train = self.scaler.fit_transform(X_train)
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluates the model."""
        if SCALE:
            X_test = self.scaler.transform(X_test)
        y_pred = self.model.predict(X_test)
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        print(f'tn: {tn}, fp: {fp}, fn: {fn}, tp: {tp}')

        metrics = {
            'Precision': precision_score(y_test, y_pred),
            'Recall': recall_score(y_test, y_pred),
            'Accuracy': accuracy_score(y_test, y_pred),
            'F1': f1_score(y_test, y_pred)
        }

        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=4)


    def save_model(self, model_path: str):
        """Saves the trained model."""
        joblib.dump((self.model, self.scaler), model_path)

    def plot_feature_importance(self, X_train):
        """Generates and saves SHAP feature importance visualization."""
        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_train)

        # Feature Importance assessment using SHAP
        plt.figure(figsize=(16, 8))
        shap.plots.bar(shap_values[:,:,1], show=False)
        plt.savefig("visualizations/shap_feature_importance.png", bbox_inches='tight', dpi=300)
        plt.close()
        shap.plots.waterfall(shap_values[0,:,1], show=False)
        plt.savefig("visualizations/shap_feature_importance_0.png", bbox_inches='tight', dpi=300)
        plt.close()
        shap.plots.beeswarm(shap_values[:,:,1], show=False)
        plt.savefig("visualizations/shap_feature_importance_beeswarm.png", bbox_inches='tight', dpi=300)
        plt.close()
        print("SHAP feature importance saved as shap_feature_importance.png")

        # Save raw SHAP values
        shap_df = pd.DataFrame(shap_values[:,:,0].values, columns=X_train.columns)
        shap_df.to_csv("shap_values.csv", index=False)
        print("SHAP values saved as shap_values.csv")
