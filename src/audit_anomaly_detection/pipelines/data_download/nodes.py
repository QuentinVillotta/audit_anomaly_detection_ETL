"""
This is a boilerplate pipeline 'data_download'
generated using Kedro 0.19.2
"""
import logging
import pandas as pd
import requests as r
import time
import os
import dask
from dask.distributed import Client, LocalCluster
import streamlit.components.v1 as components
import streamlit as st

logger = logging.getLogger(__name__)



def extract_raw_data_from_api(
    api_response: r.Response, 
    api_data_key: str
)-> dict:
    """
    Extracts and formats raw data from an API response into a Pandas DataFrame.

    Parameters:
    - api_response (requests.Response): The response object obtained from the API request.
    - api_keywords_data_location (str): The key indicating the location of the data within the JSON response of the API.
    Returns:
    - dict: A dict containing the extracted data from the API response.
    """
    return api_response.json()[api_data_key]


def extract_questionnaire_from_api(
    api_response: r.Response, 
    api_location_key: str = "content",
    api_survey_key: str = "survey"
)-> dict:
    data = api_response.json()[api_location_key][api_survey_key]
    return pd.DataFrame(data)


def extract_audit_url(
    data: dict,
    audit_id: str,
    api_audit_location_key: str = "_attachments",
    api_audit_url_key: str = "download_url"
)-> pd.DataFrame:
    df = pd.DataFrame(data)
    df["audit_url"] = df[api_audit_location_key].str[0].str[api_audit_url_key]
    df["audit_url"] = df["audit_url"].str.split("?").str[0]
    return df[[audit_id, "audit_url"]]

def _attach_id(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """
    Attaches audit_id to the input dataframe.

    Args:
        df: Pandas DataFrame representing the input data.
        key: str representing the partition id.

    Returns:
        Pandas DataFrame with audit_id attached.
    """
    if "/" in key:
        audit_id = os.path.basename(os.path.normpath(key))
    else:
        audit_id = key

    df.loc[:, "audit_id"] = audit_id
    return df


def _download_audit_files(url: str, 
                          audit_id: str, 
                          kobo_credentials: str,
                          max_retries: int = 5,
                          retry_delay: int = 2) -> pd.DataFrame:
    attempt = 0
    while attempt < max_retries:
        try:
            # Attempt to download the data
            df = pd.read_csv(url,
                             storage_options={'Authorization': kobo_credentials})
            # Normaize columns names - Remove white space
            df = df.rename(columns=lambda x: x.strip())
            # Attach the audit ID
            df = _attach_id(df, audit_id)
            logger.debug(f"Audit ID: {audit_id} successfully downloaded")
            return df
        except Exception as e:
            attempt += 1
            logger.warning(f"Attempt {attempt} failed to download {url}. Error: {e}")
            if attempt < max_retries:
                time.sleep(retry_delay)
            else:
                logger.error(f"Failed to download after {max_retries} attempts: {url}")
                raise e 

def extract_audit_files(
    df: pd.DataFrame,
    audit_id: str,
    kobo_credentials: str,
    dask_nb_worker: int = 10,
    dask_nb_thread_per_worker: int = 1,
    dask_dashboard_url: str = "http://127.0.0.1:8787"
)-> None:
    # iframe_placeholder = st.empty()
    placeholder = st.empty()
    cluster = LocalCluster(n_workers = dask_nb_worker, threads_per_worker = dask_nb_thread_per_worker)
    with Client(cluster, asynchronous = True) as client:
        with placeholder.container():
            components.iframe(src=dask_dashboard_url, height=500)
        # Dask Parallelisation
        start_time = time.time()
        delayed_audit_download = [dask.delayed(_download_audit_files)(url, id_audit, kobo_credentials) for url, id_audit in zip(df['audit_url'],df[audit_id])]
        # Lazy Computation
        audit_downloaded = dask.compute(delayed_audit_download)[0]
        logger.info("Audit files download time : %s seconds" % (round(time.time() - start_time)))
        # Concatenation all audit files
        audit_df = pd.concat(audit_downloaded, ignore_index=True)
        # Calculate max number of columns among the DataFrames
        max_columns = max(df.shape[1] for df in audit_downloaded)
        # Verify the number of columns after concatenation
        if audit_df.shape[1] != max_columns:
            raise ValueError(f"Column count mismatch: concatenated DataFrame has {audit_df.shape[1]} columns, expected {max_columns}.")
    cluster.close()
    placeholder.empty()
    return audit_df