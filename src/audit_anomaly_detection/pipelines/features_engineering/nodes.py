"""
This is a boilerplate pipeline 'features_engineering'
generated using Kedro 0.19.3
"""
import logging
from typing import Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Node 1
def attach_residual_time_event(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation (by survey) of the difference between the time spent on an event
    and the median time spent on that event across all surveys.

    Parameters:
    Pandas DataFrame representing the input data.
     parameters: Dictionary containing parameters for the function.

    Returns:
    pd.DataFrame: DataFrame with columns 'survey_id', 'mean_difference', and 'std_difference'.
    """

    df[parameters["event_time_difference_from_median"]] = df[parameters["seconds"]] - df[parameters["median_seconds"]] 
    df[parameters["mean_event_time_difference"]] = df.groupby(parameters["audit_id"])[parameters["event_time_difference_from_median"]].transform("mean")
    df[parameters["median_event_time_difference"]] = df.groupby(parameters["audit_id"])[parameters["event_time_difference_from_median"]].transform("median")
    df[parameters["std_event_time_difference"]] = df.groupby(parameters["audit_id"])[parameters["event_time_difference_from_median"]].transform("std")
    # Compute diff between median and mean
    df[parameters["mean_median_difference_event_time_difference"]] =  df[parameters["mean_event_time_difference"]] - df[parameters["median_event_time_difference"]]
    # Normalize diff
    #df[parameters["normalized_mean_median_difference_event_time_difference"]]  = df[parameters["mean_median_difference_event_time_difference"]] / (df[parameters["mean_event_time_difference"]] + df[parameters["median_event_time_difference"]])
    return df

# Node 2
def attach_count_outliers_event_time_difference(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    # Mean outlier
    df[parameters["nb_event_time_outside_delta_std"]] = df.groupby(parameters["audit_id"])[parameters["event_time_outside_delta_std"]].transform("sum")
    # Median outlier
    df[parameters["nb_event_time_outside_1_5_IQR"]] = df.groupby(parameters["audit_id"])[parameters["event_time_outside_1_5_IQR"]].transform("sum")
    return df

# Node 3
def attach_residual_time_group_question(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Calculates the mean and standard deviation (by survey) of the difference between the time spent on an event
    and the median time spent on group questions across all surveys.

    Parameters:
    Pandas DataFrame representing the input data.
     parameters: Dictionary containing parameters for the function.

    Returns:
    pd.DataFrame: DataFrame with columns 'survey_id', 'mean_difference', and 'std_difference'.
    """

    # Vérification et conversion des colonnes 'start' et 'end' en datetime si nécessaire
    # for column in [parameters["start"], parameters["end"]]:
    #     if not pd.api.types.is_datetime64_any_dtype(df[column]):
    #         df[column] = pd.to_datetime(df[column], utc=True, unit='ms')

    df["start_datetime"] =  pd.to_datetime(df[parameters["start"]], utc=True, unit='ms')
    df["end_datetime"] =  pd.to_datetime(df[parameters["end"]], utc=True, unit='ms')

    # Get the group question from node
    df[parameters['node_group']] = df[parameters['full_path']].str.rsplit('/', n=1).str[0]
    filtered_df = df.dropna(subset=[parameters['node_group']])

    group_question_time_df = (filtered_df
          .groupby([parameters['audit_id'], parameters['node_group']])
          .agg({'start_datetime': 'min', 'end_datetime': 'max'})
          .reset_index()
          .assign(duration=lambda x: round((x['end_datetime'] - x['start_datetime']).dt.total_seconds()))
          )
    # 
    group_question_time_df['median_duration_group_question'] = group_question_time_df.groupby(parameters['node_group'])['duration'].transform('median')
    group_question_time_df['diff_duration_group_question'] = group_question_time_df['duration'] - group_question_time_df['median_duration_group_question']
 
    result = group_question_time_df.groupby(parameters["audit_id"])['diff_duration_group_question'].agg(['mean', 'median', 'std']).reset_index()
    result.columns = [parameters["audit_id"], parameters['mean_event_time_difference_group_question'],
                     parameters['median_event_time_differesnce_group_question'],  parameters['std_event_time_difference_group_question']]
    #Compute diff between median and mean
    result[parameters["mean_median_difference_time_difference_group_question"]] =  result[parameters["mean_event_time_difference_group_question"]] - result[parameters["median_event_time_differesnce_group_question"]]
    # Normalize diff
    # result[parameters["normalized_mean_median_difference_time_difference_group_question"]]  = result[parameters["mean_median_difference_time_difference_group_question"]] / (result[parameters["mean_event_time_difference_group_question"]] + result[parameters["median_event_time_differesnce_group_question"]])
    
    # Merge with initial dataste
    df = df.merge(result, on=parameters["audit_id"], how='left')
    return df

# Node 4
def isnot_survey_in_daytime(df: pd.DataFrame, parameters: Dict,
                            start_time_range: str = "07:00:00", 
                            end_time_range: str = "19:00:00") -> pd.DataFrame:
    """
    Determine if each survey falls within the GMT 7am - 7pm time range.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
    pd.DataFrame: DataFrame with audit_id and a column 'notin_daytime' indicating if the survey is in the range time ( 7am-7pm by default) (0 if in range, 1 otherwise).
    """
    # Ensure the timestamps are in datetime format
    df["date_start"] = pd.to_datetime(df[parameters["start"]], utc=True, unit='ms')
    
    # Define the time range boundaries
    start_time = pd.to_datetime(start_time_range).time()
    end_time = pd.to_datetime(end_time_range).time()
    
    # Group by audit_id to get the first and last timestamps for each survey
    survey_times = df.groupby(parameters["audit_id"]).agg(
        first_start=pd.NamedAgg(column="date_start", aggfunc='min'),
        last_end=pd.NamedAgg(column="date_start", aggfunc='max')
    ).reset_index()
    
    def check_daytime(row):
        start_date = row['first_start'].date()
        end_date = row['last_end'].date()
        start_time_stamp = row['first_start'].time()
        end_time_stamp = row['last_end'].time()
        
        # Check if the survey starts and ends on the same day
        if start_date != end_date:
            return 1
        
        # Check if both the start and end times are within the time range on the same day
        if start_time <= start_time_stamp <= end_time and start_time <= end_time_stamp <= end_time:
            return 0
        return 1
    
    # Apply the function to each row
    survey_times[parameters['notin_daytime']] = survey_times.apply(check_daytime, axis=1)

    # Merge the 'in_daytime' results back into the original DataFrame
    df = df.merge(survey_times[[parameters["audit_id"], parameters['notin_daytime']]], on=parameters["audit_id"], how='left')
    return df

# Node 5
def attach_largest_relative_pace_increase(
    df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """Attach surveys with `largest_relative_pace_increase`.

    This function calculates the largest relative change in median pace between the first n audits of a survey and the remaining len-n audits of the survey, only counting the first encounter per deindexed node.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "largest_relative_pace_increase" containing the largest relative change in median pace.
    """
    intermediate = (
        df.groupby(parameters["audit_id"], as_index=False)
        .apply(_with_largest_relative_pace_increase, parameters, 
            # include_groups=False
            )
        .rename(columns={None: parameters["largest_relative_pace_increase"]})
    )
    return df.merge(intermediate, on=parameters["audit_id"])

# Node 5 -- Inter function
def _with_largest_relative_pace_increase(
    df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """Calculates the largest relative change in median pace between the first n audits of a survey and the remaining len-n audits of the survey, only counting the first encounter per deindexed node.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The largest relative change in median pace.
    """
    first_encounters = df.drop_duplicates(parameters["node_base_path"], keep="first")
    buffer = len(first_encounters) // 4
    relative_pace = first_encounters[parameters["relative_pace"]]
    median_until_now = relative_pace.expanding(buffer).median()
    median_from_now = relative_pace[::-1].expanding(buffer).median()[::-1]
    relative_relative_pace_increase = median_from_now / median_until_now
    return relative_relative_pace_increase.max()

# Node 6
def attach_duration_survey_ignoring_pauses_minutes(
    df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """
    Calculates Total Duration of Survey (in minutes), ignoring times when the survey was paused until the time it was resumed.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        Pandas DataFrame with the duration of survey (in minutes) ignoring pauses.
    """
    duration = df.groupby(parameters["audit_id"]).apply(
        lambda audit: audit.iloc[-1].loc[parameters["start"]]
        - audit.iloc[0].loc[parameters["start"]],
        # include_groups=False,
    )
    time_paused = df.groupby(parameters["audit_id"]).apply(
        lambda audit: audit[parameters["start"]]
        .diff()[
            audit.index[
                audit[parameters["event"]] == parameters["form_resume"]
            ].tolist()
        ]
        .sum(),
        # include_groups=False,
    )
    intermediate = (
        ((duration - time_paused) / 60000)
        .reset_index()
        .rename(columns={0: parameters["duration"]})
    )

    return df.merge(intermediate, on=parameters["audit_id"])

# Node 7
def attach_time_per_question_minutes(
    df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """
    Calculates the time per question (in minutes) for each audit_id in the input dataframe.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        Pandas DataFrame with the time per question (in minutes) for each audit_id.
    """
    count_unique_questions = (
        df.groupby(parameters["audit_id"], as_index=False)
        .apply(lambda x: len(x[parameters["full_path"]].unique()), 
            # include_groups=False
            )
        .rename(columns={None: parameters["num_questions"]})
    )
    df = df.merge(count_unique_questions, on=parameters["audit_id"])
    df.loc[:, parameters["time_per_question"]] = (
        df.loc[:, parameters["duration"]] / df.loc[:, parameters["num_questions"]]
    )
    return df

# Node 8
def attach_nb_value_modifications(
    df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Adds a new column to the DataFrame that counts the number of times enumerator 
    is making question value modification in the survey

    Parameters:
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    
    parameters : Dict
        A dictionary containing the following keys:
        - 'audit_id': The name of the column used for grouping the data (usually a unique identifier).
        - 'old_value': The name of the column containing the old values.
        - 'new_value': The name of the column containing the new values.
        - 'nb_value_modifications': The name of the new column to be added, which will contain the 
          count of non-NaN pairs of old and new values for each group.
    
    Returns:
    -------
    pd.DataFrame
        The original DataFrame with an additional column as specified in the 'nb_value_modifications' 
        parameter, containing the count of non-NaN value pairs for each group defined by 'audit_id'.

    """
    # Compute when both values are not NA
    df['both_non_null'] = df[[parameters['old_value'], parameters['new_value']]].notna().all(axis=1)
    # Count by survey number of TRUE
    df[parameters["nb_value_modifications"]] = df.groupby(parameters['audit_id'])['both_non_null'].transform('sum')
    # Drop intermediate var
    df = df.drop(columns=['both_non_null'])
    return df

# Node 9
def attach_constraint_note_count(
    audit_df: pd.DataFrame, questionnaire_df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """Counts the number of constraint notes that pop up in survey.

    Args:
        audit_df: Pandas DataFrame representing the input data.
        questionnaire_df: Pandas DataFrame representing the questionnaire data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "constraint_note_count" added.
    """
    notes = questionnaire_df[questionnaire_df[parameters["type"]] == parameters["note"]]
    constraint_notes = notes[~notes[parameters["relevant"]].isna()]
    constraint_note_names = constraint_notes[parameters["name"]]

    num_constraints = (
        audit_df.groupby(parameters["audit_id"], as_index=False)
        .apply(
            _count_constraint_notes,
            constraint_note_names,
            parameters,
            # include_groups=False,
        )
        .rename(columns={None: parameters["constraint_note_count"]})
    )

    return audit_df.merge(num_constraints, on=parameters["audit_id"])

# Node 9 -- Inter function
def _count_constraint_notes(
    df: pd.DataFrame, notes: pd.Series, parameters: Dict
) -> int:
    """Counts the number of constraint notes that pop up in survey.

    Args:
        df: Pandas DataFrame representing the input data.
        notes: Series of note names to be checked for in audit file.
        parameters: Dictionary containing parameters for the function.

    Returns:
        int: The number of constraint notes that pop up in survey.
    """
    num_notes = 0
    audit_questions = df[parameters["full_path"]]
    for name in notes:
        num_notes += audit_questions.str.count(name).sum()
    return num_notes

# Node 10
def attach_constraint_backtrack_count(
    audit_df: pd.DataFrame, questionnaire_df: pd.DataFrame, parameters: Dict
) -> pd.DataFrame:
    """
    Calculates the number of times the survey re-visits questions that place constraints
    on other questions. If this happens frequently it could be a sign of many incorrect values being entered
    and having to go back to reference or change the constraining value

    Args:
        audit_df: Pandas DataFrame representing the input data.
        questionnaire_df: Pandas DataFrame representing the questionnaire data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        Pandas DataFrame with the number of times the survey re-visits questions that place constraints
        on other questions.
    """
    question_names = questionnaire_df[parameters["name"]].fillna("NaN")
    constraints = questionnaire_df[parameters["constraint"]]

    reference_matrix = []
    for question in question_names:
        references = list(
            constraints.str.contains("{" + question + "}").fillna(0).astype(int)
        )
        reference_matrix.append(references)

    # The row is the constrained question and the column is the constraining question.
    # So a value of 1 in the row "demo_number_hh" and the column "demo_nb_membre" means that
    # the "demo_number_hh" question comes after the "demo_nb_membre" question and is
    # constrained by it
    constraint_matrix = pd.DataFrame(
        reference_matrix, columns=question_names, index=question_names
    ).transpose()
    constraint_matrix = constraint_matrix.loc[:, (constraint_matrix != 0).any(axis=0)]
    constraint_matrix = constraint_matrix.loc[(constraint_matrix != 0).any(axis=1), :]

    num_backtracks = (
        audit_df.groupby(parameters["audit_id"], as_index=False)
        .apply(
            _constraint_backtrack, constraint_matrix, parameters, 
            # include_groups=False
        )
        .rename(columns={None: parameters["constraint_backtracks_count"]})
    )

    return audit_df.merge(num_backtracks, on=parameters["audit_id"])

# Node 10 -- Inter function
def _constraint_backtrack(
    audit_df: pd.DataFrame, constraint_matrix_df: pd.DataFrame, parameters: Dict
) -> int:
    """
    Calculates the number of times the survey re-visits questions that place constraints
    on other questions. If this happens frequently it could be a sign of many incorrect values being entered
    and having to go back to reference or change the constraining value

    Args:
        audit_df: Pandas DataFrame representing the input data.
        constraint_matrix_df: Pandas DataFrame representing the constraint matrix.
        parameters: Dictionary containing parameters for the function.

    Returns:
        int: The number of times the survey re-visits questions that place constraints
        on other questions.
    """
    audit = audit_df.loc[audit_df[parameters["full_path"]].notna(), :]

    audit.loc[:, parameters["full_path"]] = audit[parameters["full_path"]].apply(
        lambda x: x.split("/")[-1]
    )

    # Look only at nodes that are constrained or constraining questions
    constraint_audit = audit[
        (audit[parameters["full_path"]].isin(constraint_matrix_df.columns))
        | (audit[parameters["full_path"]].isin(constraint_matrix_df.index))
    ]

    # Now filter out questions that were only looked at for one second, this
    # implies that the question was just scrolled past on the way to another question
    # and wasn't referred to or edited.
    threshold = 1000
    constraint_audit_final = constraint_audit[
        (
            constraint_audit[parameters["end"]]
            - constraint_audit[parameters["start"]]
        )
        > threshold
    ]
    constraint_audit_final = constraint_audit_final.reset_index()

    constraint_backtracks = 0
    audit_length = len(constraint_audit_final)
    for i, row in constraint_audit_final.iterrows():
        current_question = row[parameters["full_path"]]
        if current_question in constraint_matrix_df.index:
            ref_mat_row = constraint_matrix_df.loc[current_question]
            for j in range(i + 1, audit_length):
                ref_question = constraint_audit_final[parameters["full_path"]][j]
                if ref_question in ref_mat_row.index:
                    constraint_backtracks += ref_mat_row[ref_question]
    return constraint_backtracks

# Node 11
def attach_resume_count(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Counts the number of times the survey is paused and then later resumed.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        Pandas DataFrame with the resume count for each audit_id.
    """
    # todo: verify results
    resume_count = (
        df.groupby(parameters["audit_id"], as_index=False)
        .apply(
            lambda x: x.event.str.count(parameters["form_resume"]).sum(),
            # include_groups=False,
        )
        .rename(columns={None: parameters["resume_count"]})
    )

    return df.merge(resume_count, on=parameters["audit_id"])

# Node 12
def attach_constraint_error_count(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """Counts number of times the audit event 'constraint error' is reached.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dictionary containing parameters for the function.

    Returns:
        pd.DataFrame: The input DataFrame with a new column "constraint_error_count" added.
    """
    constraint_errors = (
        df.groupby(parameters["audit_id"], as_index=False)
        .apply(
            lambda x: x.event.str.count(parameters["constraint_error"]).sum(),
            # include_groups=False,
        )
        .rename(columns={None: parameters["constraint_error_count"]})
    )

    return df.merge(constraint_errors, on=parameters["audit_id"])

# Node 13
def group_dataframe_by_audit_id(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    return df.groupby(parameters["audit_id"]).first().reset_index()

# Node 14
def attach_duration_minimum_outlier(
    df: pd.DataFrame, parameters: Dict
    ) -> pd.DataFrame:
    """
    Adds a binary indicator column to the input DataFrame that flags rows
    where the 'duration' value is considered an outlier based on the
    Interquartile Range (IQR) method. An outlier is defined as any value
    that is below 1.5 times the IQR below the first quartile (Q1).

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame that contains the column for 'duration'.
    parameters : Dict
        A dictionary containing the following keys:
        - 'duration': The name of the column in the DataFrame containing
          the duration values to analyze.
        - 'duration_minimum_outlier': The name of the new column that will
          store the binary outlier indicator (1 if outlier, 0 otherwise).
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame with an additional column specified in
        `parameters['duration_minimum_outlier']`. This column contains binary
        values where 1 indicates that the 'duration' value is below the
        outlier threshold (1.5 * IQR below Q1).
    """
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    Q1 = df[parameters['duration']].quantile(0.25)
    Q3 = df[parameters['duration']].quantile(0.75)
     # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1
    # Define the lower and upper bounds for outliers (1.5 times the IQR)
    lower_bound = Q1 - 1.5 * IQR
    # Create the indicator feature (1 if duration < lower_bound, 0 otherwise)
    df[parameters['duration_minimum_outlier']] = (df[parameters['duration']] < lower_bound).astype(int)
    return df

# Node 15
def keep_selected_features(df: pd.DataFrame, parameters: Dict) -> pd.DataFrame:
    """
    Returns a Pandas DataFrame with only the specified features.

    Args:
        df: Pandas DataFrame representing the input data.
        parameters: Dict containing the following keys:
            - features: list of str representing the names of the columns to keep.

    Returns:
        Pandas DataFrame with only the specified features.
    """
    return df.loc[:, parameters["features"]]

# Node 16
def remove_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows containing NaN values from the input dataframe.

    Args:
        df: Pandas DataFrame representing the input data.

    Returns:
        Pandas DataFrame with NaN values removed.
    """
    logger.info(f"shape prior to removing NaN: {df.shape}")
    logger.info(f"{df.isna().sum()} NaN values present")
    df_clean = df.dropna().reset_index(drop=True)
    return df_clean

# Node 17
def standard_scaling_input_features(X: pd.DataFrame,  parameters: Dict) -> pd.DataFrame:
    # Remove audit_id column
    X_features = X.drop([parameters['audit_id']], axis=1)
    # Features Standardisation
    scaler = StandardScaler()
    X_features_scaled = scaler.fit_transform(X_features)
    # Concat audit_id col
    X_features_scaled = pd.DataFrame(X_features_scaled, columns=X_features.columns)
    X_features_scaled.insert(0, parameters['audit_id'],  X[parameters['audit_id']])
    return X_features_scaled
