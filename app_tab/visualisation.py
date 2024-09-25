import streamlit as st
from app_utils import plot_tools as pt


# Params
THRESHOLD_QUANTITATIVE_TYPE = 20


def display():

     predic_data = st.session_state.ETL_output['features_prediction_score']
     variable_types = pt.classify_variable_types(predic_data, THRESHOLD_QUANTITATIVE_TYPE)
     hue_var_list = ['anomaly_prediction']

     # Exclude the 'enum_id', 'anomaly_score' and anomaly_prediction columns
     features_list = [col for col in predic_data.columns if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]

     col1, col2 = st.columns([1, 5])
     with col1:
          variable = st.selectbox("Choose a variable:", features_list, index=None)
          hue_variable = st.selectbox("Choose a grouping variable:", hue_var_list, index=None)
     with col2:
          if variable:
               fig = pt.univariate_ploting(df=predic_data, variable=variable, hue=hue_variable, variable_types=variable_types)
               st.pyplot(fig)
