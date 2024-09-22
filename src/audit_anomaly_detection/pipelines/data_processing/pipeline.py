"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline

from audit_anomaly_detection.pipelines.data_processing import nodes


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
        # Questionnaire
            node(
                nodes.format_columns,
                inputs=[
                    "memory_01_questionnaire",
                    "params:questionnaire_columns",
                    "params:drop_other_questionnaire_vars"
                ],
                outputs="memory_02_intermediate_questionnaire"
            ),
        # Raw Data
            node(
                nodes.format_columns,
                inputs=[
                    "memory_01_raw_data",
                    "params:data_columns",
                    "params:drop_other_data_vars"
                ],
                outputs="memory_format_raw_data"
            ),
            node(
                nodes.attach_start_date,
                inputs=["memory_format_raw_data", "params:raw_data"],
                outputs="memory_feature_start_date",
            ),
            node(
                nodes.attach_survey_count,
                inputs=["memory_feature_start_date", "params:raw_data"],
                outputs="memory_02_intermediate_raw_data",
            ),
        # Audit Data
             node(
                nodes.format_columns,
                inputs=[
                    "memory_01_raw_audit",
                    "params:audit_columns",
                    "params:drop_other_audit_vars"
                ],
                outputs="memory_format_audit"
            ),
            node(
                nodes.attach_duration_seconds,
                inputs=[
                    "memory_format_audit",
                    "params:audit_data",
                ],
                outputs="memory_feature_duration_seconds",
            ),
            node(
                nodes.attach_duration_log_sqrt_seconds,
                inputs=[
                    "memory_feature_duration_seconds",
                    "params:audit_data",
                ],
                outputs="memory_feature_duration_log_sqrt_seconds",
            ),
            node(
                nodes.attach_median_seconds,
                inputs=[
                    "memory_feature_duration_log_sqrt_seconds",
                    "params:audit_data",
                ],
                outputs="memory_feature_median_seconds",
            ),
            node(
                nodes.attach_event_time_std,
                inputs=[
                    "memory_feature_median_seconds",
                    "params:audit_data",
                ],
                outputs="memory_feature_event_time_std_factors",
            ),
            node(
                nodes.attach_event_time_IQR,
                inputs=[
                    "memory_feature_event_time_std_factors",
                    "params:audit_data",
                ],
                outputs="memory_feature_event_time_IQR",
            ),
            node(
                nodes.attach_mean_sqrt_log_seconds,
                inputs=[
                    "memory_feature_event_time_IQR", 
                    "params:audit_data"],
                outputs="memory_feature_mean_sqrt_log_seconds",
            ),
            node(
                nodes.attach_node_base_path,
                inputs=["memory_feature_mean_sqrt_log_seconds",
                        "params:audit_data"],
                outputs="memory_feature_node_base_path",
            ),
            node(
                nodes.attach_relative_pace, 
                inputs=["memory_feature_node_base_path",
                        "params:audit_data"],
                outputs="memory_02_intermediate_audit",
            )
        ]
    )
