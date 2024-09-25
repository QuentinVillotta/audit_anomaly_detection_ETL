"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.2
"""
import logging
import os
from typing import Dict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Node 1
def format_columns(data: pd.DataFrame | dict,
                   config: dict,
                   drop_other_vars: bool
) -> pd.DataFrame:
    """
    Format the columns of a DataFrame based on a configuration dictionary.

    Parameters:
    df (pd.DataFrame): The DataFrame to be transformed.
    config (dict): A dictionary containing the column transformations.
                   Format:
                   {
                       'new_col_name': {'mapping': 'original_col_name', 'dtype': 'desired_dtype'},
                       ...
                   }
    Returns:
    pd.DataFrame: The transformed DataFrame.
    """
    if isinstance(data, (list, dict)):
        data = pd.DataFrame(data)

    # Create a dictionary for renaming columns
    rename_dict = {v['mapping']: k for k, v in config.items() if 'mapping' in v and v['mapping'] in data.columns}
    # Rename the columns
    data = data.rename(columns=rename_dict)
    # Create a dictionary for the dtype conversion
    dtype_dict = {k: v['dtype'] for k, v in config.items() if 'dtype' in v and v['mapping'] in data.columns}
    # Convert the column types
    data = data.astype(dtype_dict)
    # Drop others columns
    if drop_other_vars:
        selected_vars = list(config.keys())
        data = data.loc[:, selected_vars]
    return data

# Node 2
def attach_start_date(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Returns a DataFrame with a new column 'start_date' which represents the date of the start time of the survey.

    Args:
        df: Pandas DataFrame representing the input data.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "start_date" added.
    """

    df[parameters["start"]] = pd.to_datetime(df[parameters["start"]], utc=True)
    # Extract date and time from datetime
    df[parameters["start_date"]] = df[parameters["start"]].dt.date
    df[parameters["start_time"]] = df[parameters["start"]].dt.time
    return df

# Node 3
def attach_survey_count(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Adds a new column 'num_surveys' to the input DataFrame representing the number of surveys conducted by each enumerator.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "num_surveys" added.
    """
    df.loc[:, parameters["num_surveys"]] = df.groupby(parameters["i_enum_id"])[
        parameters["start"]
    ].transform("count")
    return df

# Node 4
def attach_duration_seconds(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Attaches duration in seconds to the input dataframe.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dict containing the following keys:
            - seconds: str representing the name of the column to store duration in seconds.
            - end: str representing the name of the column containing the end time.
            - start: str representing the name of the column containing the start time.

    Returns:
        Pandas DataFrame with duration in seconds attached.
    """
    df["date_start"] = pd.to_datetime(df[parameters["start"]], utc=True, unit='ms')
    df["date_end"] = pd.to_datetime(df[parameters["end"]], utc=True, unit='ms')
    df[parameters["seconds"]] = round((df["date_end"] - df["date_start"]).dt.total_seconds())
    return df

# Node 5
def attach_median_seconds(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Adds a new column 'median_seconds' to the input DataFrame representing the median of seconds for each node.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "median_seconds" added.
    """
    df.loc[:, parameters["median_seconds"]] = df.groupby([parameters["event"], parameters["full_path"]], dropna=False)[
           parameters["seconds"]].transform("median")
    return df

# Node 6
def attach_event_time_std(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:

    df["event_time_std"] = df.groupby([parameters["event"], parameters["full_path"]], dropna=False)[
                                       parameters["seconds"]].transform("std")
    df["event_time_mean"] = df.groupby([parameters["event"], parameters["full_path"]], dropna=False)[
                                       parameters["seconds"]].transform("mean")
    df["event_time_std_factor"] = parameters["event_time_std_delta"] * df["event_time_std"]

    # Déterminer les bornes inférieure et supérieure pour 3 fois l'écart-type
    df['sigma_lower_bound'] = df['event_time_mean'] - parameters["event_time_std_delta"]  * df['event_time_std']
    df['sigma_upper_bound'] = df['event_time_mean'] + parameters["event_time_std_delta"]  * df['event_time_std']

    # Vérifier si chaque valeur est comprise dans les bornes
    df[parameters['event_time_outside_delta_std']] = ~df['seconds'].between(df['sigma_lower_bound'], df['sigma_upper_bound'])
 
    return df

# Node 7
def attach_event_time_IQR(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    # Calculer la médiane, le premier quartile (Q1) et le troisième quartile (Q3) pour chaque événement
    df['q1_seconds'] = df.groupby([parameters["event"], parameters["full_path"]], dropna=False)['seconds'].transform(lambda x: x.quantile(0.25))
    df['q3_seconds'] = df.groupby([parameters["event"], parameters["full_path"]], dropna=False)['seconds'].transform(lambda x: x.quantile(0.75))
    # Calculer l'IQR
    df['iqr_seconds'] = df['q3_seconds'] - df['q1_seconds']
    # Déterminer les bornes inférieure et supérieure pour 1.5 fois l'IQR autour de la médiane
    df['IQR_lower_bound'] = df[parameters["median_seconds"]] - 1.5 * df['iqr_seconds']
    df['IQR_upper_bound'] = df[parameters["median_seconds"]] + 1.5 * df['iqr_seconds']
    # Vérifier si chaque valeur est comprise dans les bornes
    df[parameters['event_time_outside_1_5_IQR']] = ~df['seconds'].between(df['IQR_lower_bound'], df['IQR_upper_bound'])
    return df

# Node 8
def attach_node_base_path(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Adds a new column to the input DataFrame representing the base path of the node. The new column is named `node_base_path`.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column added.
    """
    df.loc[:, parameters["node_base_path"]] = df[parameters["full_path"]].apply(
        lambda x: os.path.basename(os.path.normpath(x)) if isinstance(x, str) else ""
    )
    return df

# Node 9
def attach_relative_pace(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Add a new column to the input DataFrame representing the relative pace of the audit. The new column is named
    `relative_pace` and is calculated as the median response time divided by the response time of the audit.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column added.
    """
    df.loc[:, parameters["relative_pace"]] = (
        df[parameters["median_seconds"]] / df[parameters["seconds"]]
    )
    return df