import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt

from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_object_dtype,
)

def freedman_diaconis_rule(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * (len(data) ** (-1/3))
    bins = int((data.max() - data.min()) / bin_width)
    return bins, bin_width

def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Credits: https://blog.streamlit.io/auto-generate-a-dataframe-filtering-ui-in-streamlit-with-filter_dataframe/
    
    Adds a UI on top of a dataframe to let viewers filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    df = df.copy()

    # Try to convert datetimes into a standard format (datetime, no timezone)
    for col in df.columns:
        if is_object_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col])
            except Exception:
                pass

        if is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.tz_localize(None)

    modification_container = st.container()

    with modification_container:
        to_filter_columns = st.multiselect("Filter dataframe on", df.columns)
        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            # Treat columns with < 10 unique values as categorical
            if is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                user_cat_input = right.multiselect(
                    f"Values for {column}",
                    df[column].unique(),
                    default=list(df[column].unique()),
                )
                df = df[df[column].isin(user_cat_input)]
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                _, step = freedman_diaconis_rule(df[column])
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                user_date_input = right.date_input(
                    f"Values for {column}",
                    value=(
                        df[column].min(),
                        df[column].max(),
                    ),
                )
                if len(user_date_input) == 2:
                    user_date_input = tuple(map(pd.to_datetime, user_date_input))
                    start_date, end_date = user_date_input
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df

def make_subheader(_str, font_style="monospace", font_size=18):
    subheader = f"<p style='font-family: {font_style}; font-size: {font_size}px;'>{_str}</p>"
    st.markdown(subheader, unsafe_allow_html=True)
    

def define_binning(df, X, variable_types):
    if variable_types[X] == 'continuous':
        bin_size, bin_width = freedman_diaconis_rule(df[X])
        bin_size = bin_width if bin_width > 0 else (df[X].max() - df[X].min()) / 20
        bin_start = df[X].min()
        bin_end = df[X].max()
    else:
        bin_size = 1  
        bin_start = df[X].min() - 0.5
        bin_end = df[X].max() + 0.5
        
    bins = dict(
                start=bin_start,  
                end=bin_end,      
                size=bin_size     
            )  
    return bins

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

def load_df_excel_explanation():
    if 'variable_names_excel' not in st.session_state:
        st.error("Excel file path not set in session state.")
        
    excel_file_path = st.session_state['variable_names_excel']
    df = pd.read_excel(excel_file_path)
    return df

def load_variable_attribute(var_name, var_attribute):

    df = load_df_excel_explanation()

    mapping = dict(zip(df[var_name], df[var_attribute]))

    if not mapping:
        st.error("Failed to load variable mapping. \
                 Please check your header contains {} and {} as \
                 columns".format(var_name, var_attribute))
        
    return mapping

def define_anomaly_color(key_anom, key_not_anom) -> dict:
    if 'anomaly_color' not in st.session_state:
        st.error("Please define anomaly color map")
    dict_color = {key_anom: st.session_state.anomaly_color[0],  
                   key_not_anom: st.session_state.anomaly_color[1]}
    return dict_color
# Summary tab
def pie_chart_pct_anomalies(df):
    # 1. General indicator - % of detected anomalies
    total_surveys = len(df)
    total_anomalies = df['anomaly_prediction'].sum()
    # Pie chart of detected anomalies
    pie_data = pd.DataFrame({
    "Type": ["Anomaly", "No Anomaly"],
    "Count": [total_anomalies, total_surveys - total_anomalies]
    })
    anom_colors = define_anomaly_color("Anomaly", "No Anomaly")

    fig_pie = px.pie(pie_data, values='Count', names='Type', title="Percentage of detected anomalies",
                     labels={'Type': 'Anomaly status'}, 
                     color = 'Type', color_discrete_map=anom_colors)
    # Display the pie chart
    return fig_pie


def plot_anomaly_count(df):
    # Group the dataframe and count occurrences of anomaly prediction by enumerator
    survey_count = df.groupby(['enum_id', 'anomaly_prediction']).size().unstack(fill_value=0)
    survey_count.columns = ['No Anomaly', 'Anomaly']
    
    # Reset index to use it for Plotly
    survey_count = survey_count.reset_index()

    # Convert from wide to long format to make it compatible with Plotly
    survey_count_long = survey_count.melt(id_vars='enum_id', value_vars=['No Anomaly', 'Anomaly'], 
                                          var_name='Anomaly Type', value_name='Count')
    anom_colors = define_anomaly_color("Anomaly", "No Anomaly")
    
    # Create a Plotly bar chart
    fig = px.bar(survey_count_long, x='enum_id', y='Count', color='Anomaly Type', barmode='group',
                 labels={'enum_id': 'Enumerator ID', 'Count': 'Number of surveys', 'Anomaly Type': 'Anomaly status'},
                 title='Number of surveys per enumerator by anomaly status',
                 color_discrete_map=anom_colors)

    # Ensure the x-axis is ordered by the number of anomalies in descending order
    fig.update_layout(xaxis={'categoryorder':'total descending'})

    # Display the Plotly chart in Streamlit
    return fig, survey_count_long



# Univariate Analysis
@st.fragment
def univariate_plotting(df, X, hue, variable_types, x_label=None) -> None:
    fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})

    if variable_types[X] == 'discrete':
        sns.boxplot(data=df, x=X, hue=hue, orient="h", ax=ax_box, legend=False)
        sns.countplot(data=df, x=X, hue=hue, ax=ax_hist)
        ax_box.set(xlabel='')  # Remove x-axis name for the boxplot
    elif variable_types[X] == 'non-numeric':
        sns.countplot(data=df, x=X, hue=hue, ax=ax_hist)
        ax_hist.set_xticklabels(ax_hist.get_xticklabels(), rotation=90)
        ax_box.axis('off')  # Hide the boxplot
    else:
        sns.boxplot(data=df, x=X, hue=hue, orient="h", ax=ax_box, legend=False)
        sns.histplot(data=df, x=X, hue=hue, stat="density", element="step", common_norm=False, ax=ax_hist)

    # Set the x and y labels
    if x_label:
        ax_hist.set_xlabel(x_label)  # Set the x-axis label to the provided label

    ax_box.set_ylabel('Density')  # Set y-axis label for the boxplot
    ax_hist.set_ylabel('Count')    # Set y-axis label for the histogram

    return fig

@st.fragment
def univariate_plotting_interactive(df, X, hue, variable_types, x_label=None, survey_id=None):
    
    if 'anomaly_color' not in st.session_state:
        st.error("Please define anomaly color map")
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    if hue:
        unique_categories = df[hue].unique()
        colors = [st.session_state.anomaly_color[0], st.session_state.anomaly_color[1]]  
    else:
        unique_categories = [None]  
        colors = [st.session_state.anomaly_color[1]]  

    if variable_types[X] == 'continuous':
        bin_size, bin_width = freedman_diaconis_rule(df[X])
        bin_size = bin_width if bin_width > 0 else (df[X].max() - df[X].min())/20
        bin_start = df[X].min()
        bin_end = df[X].max()
    else:
        bin_size = 1  
        bin_start = df[X].min()-0.5
        bin_end = df[X].max()+0.5
        
    bins = dict(
                start=bin_start,  
                end=bin_end,      
                size=bin_size     
            )  

    for i, category in enumerate(unique_categories):
        if hue:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        box_trace = go.Box(x=subset[X], name=str(category) if hue else "", 
                           boxmean='sd', orientation='h', marker=dict(color=color),
                           legendgroup=str(category) if hue else "All Data",
                           showlegend=False
                            )
        fig.add_trace(box_trace, row=1, col=1)

    for i, category in enumerate(unique_categories):
        if hue:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        hist_trace = go.Histogram(x=subset[X], name=str(category) if hue else "", opacity=0.7, marker=dict(color=color),
                                  histnorm='probability density' if variable_types[X] != 'discrete' else None,
                                  legendgroup=str(category) if hue else "All Data", showlegend=True,  
                                  xbins=bins)
        fig.add_trace(hist_trace, row=2, col=1)

    if survey_id and survey_id in df['audit_id'].values:
        survey_value = df.loc[df['audit_id'] == survey_id, X].values[0]
        highlight_trace = go.Scatter(
            x=[survey_value],
            y=[-1],  
            mode='markers',
            marker=dict(color='red', size=12, symbol='x-dot'),
            name=f"SurveyID value",
            legendgroup="highlight",
            showlegend=True
        )
        fig.add_trace(highlight_trace, row=1, col=1)

    fig.update_layout(
        height=800, width=600,   
        barmode='group' if variable_types[X] == 'discrete' else 'overlay', 
        title='Univariate Plot for {} grouped by Anomaly Prediction'.format(x_label) if hue else 'Univariate Plot for {}'.format(x_label),
        title_x=0.25 if hue else 0.35, legend=dict(title=hue if hue else "All Data")
    )

    if x_label:
        fig.update_xaxes(title_text=x_label, row=2, col=1)
    else:
        fig.update_xaxes(title_text=X, row=2, col=1)

    fig.update_yaxes(title_text="Density", row=1, col=1)  
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    if variable_types[X] == 'discrete':
        unique_vals = df[X].unique()
        tick_vals = sorted(unique_vals)
        tick_text = [str(val) for val in tick_vals]
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, 
                         ticks='outside', tickwidth=3, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)


# Bivariate Analysis 
@st.fragment
def kernel_density_plot(df, X, Y, hue, x_label=None, y_label=None) -> None:
    """
    Generates a kernel density plot using Plotly, allowing for the examination of 
    relationships between two continuous variables. 
    If a grouping variable (hue) is specified, it will color the data accordingly.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - X (str): Column name for the X-axis.
    - Y (str): Column name for the Y-axis.
    - hue (str): Column name for the grouping variable.
    - x_label (str): Label for the X-axis.
    - y_label (str): Label for the Y-axis.
    
    Returns:
    - None: Displays a plot using Streamlit.
    """
    anom_colors = define_anomaly_color(1, 0)

    # If hue is None, plot without grouping by color
    if hue is None:
        fig = px.density_contour(df, x=X, y=Y, marginal_x="violin", marginal_y="violin")
        
    elif hue == "anomaly_prediction":
        fig = px.density_contour(
            df, 
            x=X, 
            y=Y, 
            color=hue, 
            marginal_x="violin", 
            marginal_y="violin", 
            color_discrete_map=anom_colors#{0: st.session_state.anomaly_color[1], 1: st.session_state.anomaly_color[0]}
        )
    else:
        fig = px.density_contour(df, x=X, y=Y, color=hue, marginal_x="violin", marginal_y="violin")

    # Set axis labels if provided
    if x_label:
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

    fig.update_layout(height=600, width=1000)
    # Render the plot using Plotly
    return fig

@st.fragment
def avg_catplot_multicat(df, X, Y, hue, variable_types, x_label=None, y_label=None) -> None:
    """
    Generates a categorical plot (bar plot) using Plotly, allowing comparison of
    multiple categorical or binned numerical variables.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - X (str): Column name for the X-axis.
    - Y (str): Column name for the Y-axis (mean values will be displayed).
    - hue (str): Column name for the grouping variable.
    - variable_types (dict): Dictionary mapping variable names to 'discrete' or 'continuous'.
    - x_label (str): Label for the X-axis.
    - y_label (str): Label for the Y-axis.
    
    Returns:
    - None: Displays a plot using Streamlit.
    """

    # Function to compute mean and confidence interval
    def compute_ci(series, confidence=0.95):
        n = len(series)
        mean = np.mean(series)
        stderr = stats.sem(series)  # Standard error
        h = stderr * stats.t.ppf((1 + confidence) / 2, n - 1)  # Confidence interval
        return mean, h
    
    anom_colors = define_anomaly_color(1, 0)
    #st.write(anom_colors)
    #pilot = {str(key): color for key, color in anom_colors.items()}
    #st.write(pilot)
    # Check if the X variable is discrete or continuous
    if variable_types[X] == 'discrete':     
        if hue is None:
            df_grouped = df.groupby([X]).apply(lambda group: pd.Series({
                'mean': group[Y].mean(),
                'ci': compute_ci(group[Y])[1]
            })).reset_index()
            fig = px.bar(df_grouped, x=X, y='mean', barmode='group', error_y='ci')
        else:
            df_grouped = df.groupby([X, hue]).apply(lambda group: pd.Series({
                'mean': group[Y].mean(),
                'ci': compute_ci(group[Y])[1]
            })).reset_index()
            df_grouped[hue] = df_grouped[hue].astype(str)
            fig = px.bar(df_grouped, x=X, y='mean', color=hue, barmode='group', error_y='ci',
                         color_discrete_map={str(key): color for key, color in anom_colors.items()})
    else:
        # Use Freedman-Diaconis rule for binning and convert intervals to strings
        num_bins, _ = freedman_diaconis_rule(df[X])
        # Create binned data (categorical intervals)
        df['quantitative_var_binned'] = pd.cut(df[X], bins=num_bins)
        # Group by the binned intervals and calculate the mean for Y
        if hue is None:
            df_grouped = df.groupby(['quantitative_var_binned']).apply(lambda group: pd.Series({
                'mean': group[Y].mean(),
                'ci': compute_ci(group[Y])[1]
            })).reset_index()
        else:
            df_grouped = df.groupby(['quantitative_var_binned', hue]).apply(lambda group: pd.Series({
                'mean': group[Y].mean(),
                'ci': compute_ci(group[Y])[1]
            })).reset_index()
            df_grouped[hue] = df_grouped[hue].astype(str)
        
        # Ensure the intervals are ordered correctly
        df_grouped = df_grouped.sort_values(by='quantitative_var_binned', key=lambda x: x.cat.codes)
        # Convert intervals to string after sorting
        df_grouped['quantitative_var_binned'] = df_grouped['quantitative_var_binned'].astype(str)

        # Plot using Plotly
        if hue is None:
            fig = px.bar(df_grouped, x='quantitative_var_binned', y='mean', 
                         barmode='group', error_y='ci')
        else:
            fig = px.bar(
                    df_grouped, x='quantitative_var_binned', y='mean', color=hue, 
                    barmode='group', error_y='ci',
                    color_discrete_map={str(key): color for key, color in anom_colors.items()})

    # Set axis labels if provided
    if x_label:
        fig.update_layout(xaxis_title=x_label, yaxis_title=f'Mean of {y_label}')
    fig.update_layout(height=800, width=1000)
    # Render the Plotly chart
    return fig

@st.fragment
def density_heatmap(df, X, Y, hue, x_label=None, y_label=None) -> None:
    """
    Generates a distribution plot using Plotly, allowing for the visualization 
    of how one variable is distributed across different categories of another variable.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the data.
    - X (str): Column name for the X-axis.
    - Y (str): Column name for the Y-axis.
    - hue (str): Column name for the grouping variable.
    - x_label (str): Label for the X-axis.
    - y_label (str): Label for the Y-axis.
    
    Returns:
    - None: Displays a plot using Streamlit.
    """
    # If hue is None, plot without grouping by color
    if hue is None:
        fig = px.density_heatmap(df, x=X, y=Y, marginal_x="box", marginal_y="box")
    else:
        fig = px.density_heatmap(df, x=X, y=Y, facet_col=hue, marginal_x="box", marginal_y="box")
    
    # Set axis labels if provided
    if x_label:
        fig.update_layout(xaxis_title=x_label, yaxis_title=y_label)

    # Render the Plotly chart
    return fig

## Seaborn version
# @st.fragment
# def kernel_density_plot_seaborn(df, X, Y, hue, x_label=None, y_label=None) -> None:
#     fig = sns.JointGrid(data=df, x=X, y=Y, hue=hue)
#     fig.plot_joint(sns.kdeplot, fill=False, alpha=0.4, common_norm=True, warn_singular=False)
#     fig.plot_joint(sns.rugplot, height=-.02, clip_on=False, alpha=.5)
#     fig.plot_marginals(sns.boxplot)

#     # Set axis labels
#     if x_label:
#         fig.set_axis_labels(x_label, y_label)

#     return fig

# @st.fragment
# def catplot_multicat_seaborn(df, X, Y, hue, variable_types, x_label=None, y_label=None) -> None:
#     if variable_types[X] == 'discrete':
#         fig = sns.catplot(data=df, x=X, y=Y, hue=hue, kind="bar", estimator='mean', errorbar=('ci', 95))
#     else:
#         num_bins, _ = freedman_diaconis_rule(df[X])
#         df['quantitative_var_binned'] = pd.cut(df[X], bins=num_bins)
#         fig = sns.catplot(data=df, x="quantitative_var_binned", y=Y, hue=hue, kind="bar", estimator='mean', errorbar=('ci', 95))

#     if x_label:
#         fig.set(xlabel=x_label, ylabel=y_label)  # Set both x and y axis labels

#     fig.tick_params(axis='x', rotation=90)
#     return fig

# @st.fragment
# def displot_seaborn(df, X, Y, hue, x_label=None, y_label=None) -> None:
#     fig = sns.displot(df, x=X, y=Y, col=hue, rug=True)

#     for ax in fig.axes.flatten():
#         if x_label:
#             ax.set_xlabel(x_label)
#         if y_label:
#             ax.set_ylabel(y_label)

#     return fig



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

@st.fragment
def id_survey_shap_bar_plot_interactive(survey_id_var, selected_survey, data, shap_values) -> None:
    
    def display_top_negative_shap_values(selected_index, shap_values, features, ntop=5) -> None:
        survey_shap_values = shap_values[selected_index].values  
        shap_series = pd.Series(survey_shap_values, index=features.columns)

        negative_shap_df = shap_series[shap_series < 0].nsmallest(ntop)
        
        #st.write(negative_shap_df)
        st.subheader("Top Features Anomaly Detection", divider="gray")
        st.write("The following variable contributed the most to flag the selected \
                survey as an anomaly")

        for feature, value in negative_shap_df.items():
            feature_label = st.session_state.variable_mapping.get(feature, feature)
            feature_comment = st.session_state.variable_meaning.get(feature, "No comment available.")
            full_string = f"- **{feature_label}**: SHAP value = \
                            **{value:.4f}**. {feature_comment}"
            st.markdown(full_string)
        st.markdown("<hr style='height:2px; background-color: gray; border: none;' />", 
                    unsafe_allow_html=True)       
         
    #nb_features = shap_values.data.shape[1]
    index_survey = data[data[survey_id_var] == selected_survey].index[0]
    feature_data = data[data[survey_id_var] == selected_survey]
    shap_index_survey = data.index.get_loc(index_survey)
    
    shap_values_data = shap_values[shap_index_survey].values
    feature_names = data.drop(columns=[survey_id_var]).columns
    
    feature_label = [st.session_state.variable_mapping.get(feature, feature) for feature in feature_names]
    
    shap_df = pd.DataFrame({
        'Feature': feature_label,
        'SHAP Value': shap_values_data
    }).sort_values(by='SHAP Value', ascending=True)

    chart = alt.Chart(shap_df).mark_bar().encode(
        y=alt.Y('Feature:N', sort=None, axis=alt.Axis(labelFontSize=14, labelAngle=0)),  
        x='SHAP Value:Q',
        tooltip=['Feature', 'SHAP Value']
    ).properties(
        title=f'SHAP Values for Survey ID: {selected_survey}',
        width=1200,
        height=800
    ).configure_mark(
        opacity=0.8,
        color='skyblue'
    )

    display_top_negative_shap_values(shap_index_survey, shap_values, 
                                     data.drop(columns=[survey_id_var]), ntop=5)
    st.altair_chart(chart, use_container_width=True)


@st.fragment
def id_survey_shap_bar_plot_interactive2(survey_id_var, selected_survey, data, shap_values) -> None:
    
    def display_top_negative_shap_values2(selected_index, shap_values, data, ntop=5) -> None:
            features = data.drop(columns=[survey_id_var])
            feature_data = data[data[survey_id_var] == selected_survey]
            survey_shap_values = shap_values[selected_index].values  
            shap_series = pd.Series(survey_shap_values, index=features.columns)

            negative_shap_df = shap_series[shap_series < 0].nsmallest(ntop)
            
            st.subheader("Top Features Anomaly Detection", divider="gray")
            st.write("The following variable contributed the most to flag the selected \
                    survey as an anomaly")

            for feature, shap_value in negative_shap_df.items():
                feature_label = st.session_state.variable_mapping.get(feature, feature)
                feature_comment = st.session_state.variable_meaning.get(feature, "No comment available.")
                feature_value = feature_data[feature].values[0]
                full_string = f"- **{feature_label} = {feature_value:.0f}** \
                              (**SHAP value = {shap_value:.2f}**) \
                              {feature_comment}: is below/above its average \
                              **\u03BC = {int(data[feature].mean())} +/- {int(data[feature].std())}**."
                st.markdown(full_string)
            st.markdown("<hr style='height:2px; background-color: gray; border: none;' />", 
                        unsafe_allow_html=True)
      
         
    #nb_features = shap_values.data.shape[1]
    index_survey = data[data[survey_id_var] == selected_survey].index[0]
    shap_index_survey = data.index.get_loc(index_survey)
    
    shap_values_data = shap_values[shap_index_survey].values
    feature_names = data.drop(columns=[survey_id_var]).columns
    
    feature_label = [st.session_state.variable_mapping.get(feature, feature) for feature in feature_names]
    
    shap_df = pd.DataFrame({
        'Feature': feature_label,
        'SHAP Value': shap_values_data
    }).sort_values(by='SHAP Value', ascending=True)

    chart = alt.Chart(shap_df).mark_bar().encode(
        y=alt.Y('Feature:N', sort=None, axis=alt.Axis(labelFontSize=14, labelAngle=0)),  
        x='SHAP Value:Q',
        tooltip=['Feature', 'SHAP Value']
    ).properties(
        title=f'SHAP Values for Survey ID: {selected_survey}',
        width=1200,
        height=800
    ).configure_mark(
        opacity=0.8,
        color='skyblue'
    )

    display_top_negative_shap_values2(shap_index_survey, shap_values, data, ntop=5)
    st.altair_chart(chart, use_container_width=True)    


def generate_palette_colors(num_colors):
    """Generate a list of random hex colors."""
    colors_array = np.random.randint(0, 256, size=(num_colors, 3))
    palette = ["#{:02x}{:02x}{:02x}".format(r, g, b) for r, g, b in colors_array]
    return palette

@st.fragment
def univariate_plotting_interactive_enum(df, X, hue, variable_types, x_label=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    if hue:
        unique_categories = df[hue].unique()
        if hue == "enum_id":
            selected_categories = st.multiselect("Select categories for hue:", 
                                                 unique_categories, key='multi_select_enum_id')
            if 'palette_colors' not in st.session_state:
                st.session_state.palette_colors = generate_palette_colors(len(unique_categories))

            colors = [st.session_state.palette_colors[unique_categories.tolist().index(cat)] for cat in selected_categories] if selected_categories else []
        elif hue == "anomaly_prediction":
            selected_categories = unique_categories
            colors = [st.session_state.anomaly_color[0], st.session_state.anomaly_color[1]]
    else:
        selected_categories = [None]
        colors = [st.session_state.anomaly_color[1]]
        
    bins = define_binning(df, X, variable_types)

    for i, category in enumerate(selected_categories):
        if hue and category:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        box_trace = go.Box(x=subset[X], name=str(category) if hue else "", 
                           boxmean='sd', orientation='h', marker=dict(color=color),
                           legendgroup=str(category) if hue else "All Data",
                           showlegend=False
                            )
        fig.add_trace(box_trace, row=1, col=1)

    for i, category in enumerate(selected_categories):
        if hue and category:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        hist_trace = go.Histogram(x=subset[X], name=str(category) if hue else "", opacity=0.7, 
                                  marker=dict(color=color),
                                  histnorm='probability density' if variable_types[X] != 'discrete' else None,
                                  legendgroup=str(category) if hue else "All Data", showlegend=True,  
                                  xbins=bins)
        fig.add_trace(hist_trace, row=2, col=1)

    fig.update_layout(
        height=800, width=600,   
        barmode='group' if variable_types[X] == 'discrete' else 'overlay', 
        title='Univariate Plot for {} grouped by Anomaly Prediction'.format(x_label) if hue else 'Univariate Plot for {}'.format(x_label),
        title_x=0.25 if hue else 0.35, legend=dict(title=hue if hue else "All Data")
    )

    if x_label:
        fig.update_xaxes(title_text=x_label, row=2, col=1)
    else:
        fig.update_xaxes(title_text=X, row=2, col=1)

    fig.update_yaxes(title_text="Density", row=1, col=1)  
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    if variable_types[X] == 'discrete':
        unique_vals = df[X].unique()
        tick_vals = sorted(unique_vals)
        tick_text = [str(val) for val in tick_vals]
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, 
                         ticks='outside', tickwidth=3, row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

@st.fragment
def univariate_plotting_interactive_enum_anomaly(df, X, hue, variable_types, selected_enums=None, x_label=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])
    bins = define_binning(df, X, variable_types)

    unique_enums = df['enum_id'].unique()
    filtered_df = df[df['enum_id'].isin(selected_enums)] if selected_enums else df

    if 'enum_color_map' not in st.session_state:
        st.session_state.enum_color_map = {enum: color for enum, color in zip(unique_enums, generate_palette_colors(len(unique_enums)))}
    #st.write(len(unique_enums))
    if 'anomaly_color' not in st.session_state:
        st.error("Please define anomaly color map")
    
    if selected_enums:
        st.write("Descriptive Statistics for Selected Enumerators:")
        df_display = filtered_df.groupby(filtered_df['enum_id'])[X].describe()
        st.dataframe(df_display, height=200 if len(selected_enums) > 5 else len(selected_enums)*48)

    colors = [st.session_state.enum_color_map.get(enum) for enum in unique_enums] if selected_enums else []
    #colors = [st.session_state.enum_color_map[enum] for enum in selected_enums]
    #st.write(len(colors))

    if hue == "anomaly_prediction":
        unique_categories = filtered_df[hue].unique()
        mapped_categories = ["No Anomaly" if x == 0 else "Anomaly" for x in unique_categories]
        selected_categories = mapped_categories
        colors = st.session_state.anomaly_color
    else:
        selected_categories = selected_enums

    for i, category in enumerate(selected_categories):
        if hue == "anomaly_prediction":
            subset = filtered_df[filtered_df[hue] == (1 if category == "Anomaly" else 0)]
            color = colors[0] if category == "Anomaly" else colors[1]
        else:
            subset = filtered_df[filtered_df['enum_id'] == category]
            color = colors[i] if len(colors) > 1 else st.session_state.enum_color_map.get(unique_enums[0], '#636EFA')
        box_trace = go.Box(
            x=subset[X], 
            name=str(category), 
            boxmean='sd', 
            orientation='h', 
            marker=dict(color=color),
            legendgroup=str(category),
            showlegend=True
        )
        fig.add_trace(box_trace, row=1, col=1)

        hist_trace = go.Histogram(
            x=subset[X], 
            name=str(category), 
            opacity=0.7, 
            marker=dict(color=color),
            histnorm='probability density' if variable_types[X] != 'discrete' else None,
            legendgroup=str(category), 
            showlegend=False,  
            xbins=bins
        )
        fig.add_trace(hist_trace, row=2, col=1)

    fig.update_layout(
        height=800, width=600,   
        barmode='group' if variable_types[X] == 'discrete' else 'overlay', 
        title=f'Univariate Plot for {x_label}<br>filtered by Enumerator ID: {", ".join(selected_enums)}<br>' + 
              ('grouped by Anomaly Prediction' if hue == "anomaly_prediction" else ''),
        title_x=0.3, legend=dict(title=hue)
    )

    fig.update_xaxes(title_text=x_label if x_label else X, row=2, col=1)
    fig.update_yaxes(title_text="Density", row=1, col=1)  
    fig.update_yaxes(title_text="Count", row=2, col=1)

    if variable_types[X] == 'discrete':
        unique_vals = filtered_df[X].unique()
        tick_vals = sorted(unique_vals)
        tick_text = [str(val) for val in tick_vals]
        fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, ticks='outside', tickwidth=3, row=2, col=1)

    # Use a unique key to avoid duplicate IDs
    st.plotly_chart(fig, use_container_width=True, key=f"univariate_plot_{x_label}_{'_'.join(selected_enums) if selected_enums else 'all'}")
