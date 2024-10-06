import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import shap
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import altair as alt


def make_subheader(_str, font_style="monospace", font_size=18):
    subheader = f"<p style='font-family: {font_style}; color: white; font-size: {font_size}px;'>{_str}</p>"
    st.markdown(subheader, unsafe_allow_html=True)
    


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

def freedman_diaconis_rule(data):
    q25, q75 = np.percentile(data, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr * len(data) ** (-1/3)
    bins = int((data.max() - data.min()) / bin_width)
    return bins

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
def univariate_plotting2(df, X, hue, variable_types, x_label=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    if hue:
        unique_categories = df[hue].unique()
        colors = ["#FFA500", "#636EFA"]  
    else:
        unique_categories = [None]  
        colors = ['#636EFA']  

    if variable_types[X] == 'continuous':
        bin_size = (df[X].max() - df[X].min())/20 
        #freedman_diaconis_rule(df[X])
        bin_start = df[X].min()
        bin_end = df[X].max()
    else:
        bin_size = 2  
        bin_start = df[X].min()-1
        bin_end = df[X].max()+1
        
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
        fig.update_xaxes(tickvals=df[X].unique(), ticktext=df[X].unique(), ticks='outside', tickwidth=3, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


# Bivariate Analysis 
@st.fragment
def kernel_density_plot(df, X, Y, hue, x_label=None, y_label=None) -> None:
    fig = sns.JointGrid(data=df, x=X, y=Y, hue=hue)
    fig.plot_joint(sns.kdeplot, fill=False, alpha=0.4, common_norm=True, warn_singular=False)
    fig.plot_joint(sns.rugplot, height=-.02, clip_on=False, alpha=.5)
    fig.plot_marginals(sns.boxplot)

    # Set axis labels
    if x_label:
        fig.set_axis_labels(x_label, y_label)

    return fig

@st.fragment
def catplot_multicat(df, X, Y, hue, variable_types, x_label=None, y_label=None) -> None:
    if variable_types[X] == 'discrete':
        fig = sns.catplot(data=df, x=X, y=Y, hue=hue, kind="bar", estimator='mean', errorbar=('ci', 95))
    else:
        num_bins = freedman_diaconis_rule(df[X])
        df['quantitative_var_binned'] = pd.cut(df[X], bins=num_bins)
        fig = sns.catplot(data=df, x="quantitative_var_binned", y=Y, hue=hue, kind="bar", estimator='mean', errorbar=('ci', 95))

    if x_label:
        fig.set(xlabel=x_label, ylabel=y_label)  # Set both x and y axis labels

    fig.tick_params(axis='x', rotation=90)
    return fig

@st.fragment
def displot(df, X, Y, hue, x_label=None, y_label=None) -> None:
    fig = sns.displot(df, x=X, y=Y, col=hue, rug=True)

    for ax in fig.axes.flatten():
        if x_label:
            ax.set_xlabel(x_label)
        if y_label:
            ax.set_ylabel(y_label)

    return fig

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


