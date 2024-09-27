"""
This is a boilerplate pipeline 'features_engineering'
generated using Kedro 0.19.3
"""

from kedro.pipeline import Pipeline, node, pipeline
from audit_anomaly_detection.pipelines.features_engineering import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                nodes.attach_residual_time_event,  
                inputs=["02_intermediate_audit", 
                        "params:FE_audit_data"],
                outputs="memory_feature_residual_time_event",
            ),
            node(
                nodes.attach_count_outliers_event_time_difference,  
                inputs=["memory_feature_residual_time_event", 
                        "params:FE_audit_data"],
                outputs="memory_feature_count_outliers_event_time_difference",
            ),
            node(
                nodes.attach_residual_time_group_question,  
                inputs=["memory_feature_count_outliers_event_time_difference", 
                        "params:FE_audit_data"],
                outputs="memory_feature_residual_time_group_question",
            ),
            node(
                nodes.isnot_survey_in_daytime,  
                inputs=["memory_feature_residual_time_group_question", 
                        "params:FE_audit_data"],
                outputs="memory_feature_isnot_survey_in_daytime",
            ),
            node(
                nodes.attach_largest_relative_pace_increase, 
                inputs=[
                    "memory_feature_isnot_survey_in_daytime",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_largest_relative_pace_increase",
            ),
            node(
                nodes.attach_duration_survey_ignoring_pauses_minutes,
                inputs=[
                    "memory_feature_largest_relative_pace_increase",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_duration_survey_ignoring_pauses_minutes",
            ),
            node(
                nodes.attach_time_per_question_minutes,  
                inputs=[
                    "memory_feature_duration_survey_ignoring_pauses_minutes",
                    "params:FE_audit_data",
                ],
                outputs="memory_time_per_question_minutes",
            ),
            node(
                nodes.attach_nb_value_modifications,  
                inputs=[
                    "memory_time_per_question_minutes",
                    "params:FE_audit_data",
                ],
                outputs="memory_nb_value_modifications",
            ),
            node(
                nodes.attach_constraint_note_count, 
                inputs=[
                    "memory_nb_value_modifications",
                    "02_intermediate_questionnaire",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_constraint_note_count",
            ),
            node(
                nodes.attach_constraint_backtrack_count, 
                inputs=[
                    "memory_feature_constraint_note_count",
                    "02_intermediate_questionnaire",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_constraint_backtrack_count",
            ),
            node(
                nodes.attach_resume_count,  
                inputs=[
                    "memory_feature_constraint_backtrack_count",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_resume_count",
            ),
            node(
                nodes.attach_constraint_error_count, 
                inputs=[
                    "memory_feature_resume_count",
                    "params:FE_audit_data",
                ],
                outputs="memory_feature_constraint_error_count",
            ),
            node(
                nodes.group_dataframe_by_audit_id,
                inputs=["memory_feature_constraint_error_count",
                        "params:FE_audit_data"],
                outputs="memory_grouped_features",
            ),
            node(
                nodes.attach_duration_minimum_outlier,
                inputs=["memory_grouped_features",
                        "params:FE_audit_data"],
                outputs="memory_duration_minimum_outlier",
            ),
            node(
                nodes.keep_selected_features,
                inputs=[
                    "memory_duration_minimum_outlier",
                    "params:model_input"],
                outputs="memory_selected_features",
            ),
            node(
                nodes.remove_nans,
                inputs="memory_selected_features",
                outputs="03_features",
            )
        ]
    )

