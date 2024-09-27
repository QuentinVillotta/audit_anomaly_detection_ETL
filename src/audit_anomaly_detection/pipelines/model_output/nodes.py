"""
This is a boilerplate pipeline 'model_output'
generated using Kedro 0.19.3
"""
import logging
from typing import Dict, Any
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import shap

logger = logging.getLogger(__name__)

# Node 1
def get_predict_and_score(pipelines, X: pd.DataFrame, parameters: Dict)-> pd.DataFrame:
    # Init DF
    scores = pd.DataFrame()
    predictions = pd.DataFrame()
    # Remove audit_id column
    X_features = X.drop([parameters['audit_id']], axis=1)
    for model_name, pipeline in pipelines.items():
            y_pred = pipeline.predict(X_features)
            y_score = pipeline.decision_function(X_features)
            if model_name in  parameters["models_to_invert_score"] :
                # Convert predictions from -1/1 to 1/0
                y_pred = (y_pred == -1).astype(int)
            predictions[model_name] = y_pred
            scores[model_name] = y_score
    # Concat audit_id col
    predictions.insert(0, parameters['audit_id'],  X[parameters['audit_id']])
    scores.insert(0, parameters['audit_id'],  X[parameters['audit_id']])
    return predictions, scores

# Node 2
def anomaly_prediction(df: pd.DataFrame, parameters: Dict)-> pd.DataFrame:
    df['anomaly_prediction'] = (
    (df['IsolationForest'] == 1) &
    (
        (df['OneClassSVM'] + df['LOF'] + df['IsolationForest']) >= 2
    )).astype(int)
    return df[[parameters["audit_id"], 'anomaly_prediction']]

# Node 3
def anomaly_score(df: pd.DataFrame, parameters: Dict)-> pd.DataFrame:
    # Copy the initial scores to preserve them
    initial_scores = df.drop(columns=[parameters["audit_id"]]).copy()
    # List of models whose scores need to be inverted
    models_to_invert = parameters["models_to_invert_score"] 
    # Invert the scores for the specified models
    initial_scores[models_to_invert] = initial_scores[models_to_invert].apply(lambda x: -x)
    # Normalize the scores between 0 and 1 for all columns except 'audit_id'
    scaler = MinMaxScaler()
    initial_scores = scaler.fit_transform(initial_scores)
    # Calculate the average of the normalized scores to get the final score
    df['anomaly_score'] = initial_scores.mean(axis=1)
    return df[[parameters["audit_id"], 'anomaly_score']]

# Node 4
def merge_scores_and_predictions(df_features: pd.DataFrame, df_predictions: pd.DataFrame, df_scores: pd.DataFrame, raw_data: pd.DataFrame | dict | list, parameters: Dict)-> pd.DataFrame:
    # Merge the DataFrames on the audit_id column with specified suffixes
    df_merged_predict_and_score = pd.merge(
        df_predictions, df_scores,
        on=parameters["audit_id"]
    )

    # Merge with raw features
    df_merged_raw_features = pd.merge(
        df_merged_predict_and_score, df_features,
        on=parameters["audit_id"]
    )

    # Merge with raw data - To get enum ID
    # Insert enum id
    if isinstance(raw_data, (list, dict)):
        raw_data = pd.DataFrame(raw_data)

    raw_data_enum_id = raw_data[[parameters["raw_audit_id"], parameters["raw_enum_id"]]]
    raw_data_enum_id = raw_data_enum_id.rename(columns={parameters["raw_enum_id"]: parameters["enum_id"],
                                                        parameters["raw_audit_id"]: parameters["audit_id"]})

    df_enum_id_features = pd.merge(
    raw_data_enum_id, df_merged_raw_features,
    on = parameters["audit_id"]
    )
    
    # Sort the surveys based on the final score
    df_raw_features_sorted = df_enum_id_features.sort_values(by=['anomaly_prediction', 'anomaly_score'],
                                                             ascending=[False, False])

    return df_raw_features_sorted

# Node 5
def raw_data_predict_and_scores(raw_data: pd.DataFrame | dict | list, df_predictions: pd.DataFrame, df_scores: pd.DataFrame, parameters: Dict)-> pd.DataFrame:
    # Merge the DataFrames on the audit_id column with specified suffixes
    df_merged_predict_and_score = pd.merge(
        df_predictions, df_scores,
        on=parameters["audit_id"]
    )

    # Merge with raw features
    if isinstance(raw_data, (list, dict)):
        raw_data = pd.DataFrame(raw_data)

    df_merged_raw_data = pd.merge(
        df_merged_predict_and_score, raw_data,
        left_on = parameters["audit_id"],
        right_on = parameters["raw_audit_id"]
    )
    
    # Sort the surveys based on the final score
    df_raw_data_sorted = df_merged_raw_data.sort_values(by=['anomaly_prediction', 'anomaly_score'],
                                                        ascending=[False, False])

    return df_raw_data_sorted

# Node 6
def SHAP_interpretation(models_pipeline_dict: Dict[str, Any], features: pd.DataFrame,  parameters: Dict) -> Dict[str, Any]:
    # Remove audit_id column
    X = features.drop([parameters['audit_id']], axis=1)
    # Compute only SHAP values for Isolation Forest because for LOF and OneClassSVM, computation too slow
    isolation_forest_pipeline = models_pipeline_dict['IsolationForest']
    logger.info(f"SHAP values computation for model : IsolationForest")
    model = isolation_forest_pipeline.named_steps['model']
    explainer = shap.TreeExplainer(model, X)
    shap_values = explainer(X)
    return {'model': model, 'explainer': explainer, 'shap_values': shap_values, 'features': features} 

# Node 7
def export_output(raw_features_prediction: pd.DataFrame, raw_data_prediction: pd.DataFrame, SHAP_interpretation: Dict[str, Any]) -> Dict[str, Any]:
    output_dict = {'features_prediction_score': raw_features_prediction,
                   'raw_data_prediction_score': raw_data_prediction,
                   'SHAP_interpretation': SHAP_interpretation}
    return output_dict