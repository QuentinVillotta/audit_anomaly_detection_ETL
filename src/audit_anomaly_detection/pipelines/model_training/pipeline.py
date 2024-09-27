"""
This is a boilerplate pipeline 'model_training'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from audit_anomaly_detection.pipelines.model_training import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        node(
            nodes.fit_models,
                inputs=["03_features", "params:models"],
                outputs="04_trained_models"
            )
        ]
    )