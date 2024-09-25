import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components


def classify_variable_types(df, threshold=20):
    """
    Classify the variables of a DataFrame as continuous or discrete.

    :param df: DataFrame to analyze
    :param threshold: Threshold to distinguish between continuous and discrete variables
    :return: Dictionary with variable names as keys and their type ('continuous' or 'discrete') as values
    """
    variable_types = {}

    for column in df.columns:
        # Check if the column is numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            unique_values = df[column].nunique()
            if unique_values <= threshold:  # Threshold to distinguish continuous from discrete
                variable_types[column] = 'discrete'
            else:
                variable_types[column] = 'continuous'
        else:
            variable_types[column] = 'non-numeric'

    return variable_types


@st.fragment
def univariate_ploting(df, variable, hue, variable_types ) -> None:
     fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
     # assigning a graph to each ax
     if variable_types[variable] == 'discrete':
          # Count plot
          g_bp = sns.boxplot(data = df, x = variable, hue=hue,
                             orient="h", ax=ax_box, legend=False)
          g_hist = sns.countplot(data=df, x=variable, hue= hue, 
                              ax=ax_hist)
          # Remove x axis name for the boxplot
          ax_box.set(xlabel='')

     elif variable_types[variable] == 'non-numeric':  # New case for non-numeric variables
          # Count plot for non-numeric variables
          g_hist = sns.countplot(data=df, x=variable, hue=hue, ax=ax_hist)
          ax_hist.set_xticklabels(ax_hist.get_xticklabels(), rotation=90)
          # Set title and remove boxplot since it doesn't make sense for non-numeric
          # ax_hist.set_title(f"Count plot of {variable}")
          ax_box.axis('off')  # Hide the boxplot
     else:
          # Histogram
          g_bp = sns.boxplot(data = df, x = variable, hue=hue,
          orient="h", ax=ax_box, legend=False)
          g_hist = sns.histplot(data=df, x=variable, hue= hue, 
                              stat="density",  element="step", common_norm=False, 
                              ax=ax_hist)
          # Remove x axis name for the boxplot
          ax_box.set(xlabel='')
     return(fig)


# SHAP Plot
def st_shap(plot, height=None):
     shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
     components.html(shap_html, height=height)

@st.fragment
def id_survey_shap_force_plot(survey_id_var, selected_survey, data, shap_values ) -> None:
     index_survey= data[data[survey_id_var] == selected_survey].index[0]
     shap_index_survey = data.index.get_loc(index_survey)
     fig = shap.force_plot(shap_values[shap_index_survey])
     st_shap(fig)

@st.fragment
def id_survey_shap_bar_plot(survey_id_var, selected_survey, data, shap_values) -> None:

     nb_features = shap_values.data.shape[1]
     index_survey= data[data[survey_id_var] == selected_survey].index[0]
     shap_index_survey = data.index.get_loc(index_survey)
     fig, ax = plt.subplots()
     shap.plots.bar(shap_values[shap_index_survey],  max_display=nb_features, show=False, ax=ax)
     st.pyplot(fig)
