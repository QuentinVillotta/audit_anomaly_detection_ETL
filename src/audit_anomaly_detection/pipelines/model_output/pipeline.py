"""
This is a boilerplate pipeline 'model_output'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from audit_anomaly_detection.pipelines.model_output import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            nodes.get_predict_and_score,
            inputs=["04_trained_models", "memory_03_features_scaled", "params:model_output"],
            outputs=["memory_model_output_predictions", "memory_model_output_scores"],
            ),
        node(
            nodes.anomaly_prediction,
            inputs=["memory_model_output_predictions", "params:model_output"],
            outputs="memory_anomaly_prediction",
            ),
        node(
            nodes.anomaly_score,
            inputs=["memory_model_output_scores", "params:model_output"],
            outputs="memory_anomaly_score",
            ),
        node(
            nodes.merge_scores_and_predictions,
            inputs=["memory_03_features", "memory_anomaly_prediction", "memory_anomaly_score", "memory_01_raw_data", "params:model_output"],
            outputs="memory_raw_features_predict_and_score",
            ),
        node(
            nodes.raw_data_predict_and_scores,
            inputs=["memory_01_raw_data", "memory_anomaly_prediction", "memory_anomaly_score", "params:model_output"],
            outputs="memory_raw_data_predict_and_score",
            ),
        node(
            nodes.SHAP_interpretation,
            inputs=["04_trained_models", "memory_03_features_scaled", "params:model_output"],
            outputs="memory_SHAP_interpretation"
            ),
        node(
            nodes.export_output,
            inputs=["memory_raw_features_predict_and_score", "memory_raw_data_predict_and_score", "memory_SHAP_interpretation"],
            outputs="05_output_model"
            )
        ]
    )