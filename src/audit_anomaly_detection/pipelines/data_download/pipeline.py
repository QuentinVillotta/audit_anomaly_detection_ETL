"""
This is a boilerplate pipeline 'data_download'
generated using Kedro 0.19.2
"""

from kedro.pipeline import Pipeline, node, pipeline
from audit_anomaly_detection.pipelines.data_download import nodes

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                nodes.extract_raw_data_from_api,
                inputs=[
                    "00_kobo_api_raw_data",
                    "params:api_data_key"
                ],
                outputs="memory_01_raw_data"
            ),
            node(
                nodes.extract_audit_url,
                inputs=[
                    "memory_01_raw_data",
                    "params:audit_id",
                    "params:api_audit_location_key",
                    "params:api_audit_url_key"
                ],
                outputs="tmp_audit_url_df"
            ),
            node(
                nodes.extract_audit_files,
                inputs=[
                    "tmp_audit_url_df",
                    "params:audit_id",
                    "params:kobo_credentials",
                    "params:dask_nb_worker",
                    "params:dask_nb_thread_per_worker"
                ],
                outputs="memory_01_raw_audit"
            ),
            node(
                nodes.extract_questionnaire_from_api,
                inputs=[
                    "00_kobo_api_questionnaire",
                    "params:api_questionnaire_location_key",
                    "params:api_questionnaire_survey_key"
                ],
                outputs="memory_01_questionnaire"
            )
        ]
    )
