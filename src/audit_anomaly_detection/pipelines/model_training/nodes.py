"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.3
"""


import logging
from typing import Dict
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


# Node 1
def fit_models(X: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    # Remove audit_id column
    X_features = X.drop([parameters['audit_id']], axis=1)
    # Create pipelines
    pipelines = create_pipeline()
    # Fit the models
    for model_name, pipeline in pipelines.items():
        logger.info(f"Training model: {model_name}")
        pipeline.fit(X_features)  
    return pipelines

# Node 1 -- Inter function
def create_pipeline():
    # Define the base models for anomaly detection
    base_models = {
        'IsolationForest': IsolationForest(contamination=0.1, random_state=42), #The lower, the more abnormal. 
        'OneClassSVM':  OneClassSVM(nu=0.1, kernel='rbf'), #OneClassSVM(nu=0.05, kernel='rbf'), # Signed distance is positive for an inlier and negative for an outlier.
        'LOF': LocalOutlierFactor(n_neighbors=20, novelty=True), # Bigger is better, i.e. large values correspond to inliers.
    }
    
    # Prepare the pipelines for each model
    pipelines = {}
    for model_name, model in base_models.items():
        pipelines[model_name] = Pipeline([
            # ('scaler', StandardScaler()),
            ('model', model)
        ])
    return pipelines