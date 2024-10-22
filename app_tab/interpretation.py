import streamlit as st
import pandas as pd
import numpy as np
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from app_utils import plot_tools as pt
import plotly.express as px

# Params
SURVEY_ID_VAR = 'audit_id'
ENUMERATOR_ID_VAR = 'enum_id'

def display():
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Local Interpretation", "Global Interpretation", "Enumerator No Anomalies"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
    if "variable_mapping" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return

    if st.session_state.ETL_output:
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        shap_values = interpretation_data['shap_values']  
        features_label = interpretation_data['features']
        features = interpretation_data['features'].iloc[:, 1:]

        all_enumerators = predic_data[ENUMERATOR_ID_VAR].unique()
        enumerators_with_anomalies = predic_data[predic_data.anomaly_prediction == 1][ENUMERATOR_ID_VAR].unique()
        enumerators_without_anomalies = np.setdiff1d(all_enumerators, enumerators_with_anomalies)

        with sub_tab1:

            col1, col2 = st.columns([1, 3])

            with col1:
                selected_enumerator = st.multiselect("Filter by Enumerator", enumerators_with_anomalies)

            with col2:
                if len(selected_enumerator) == 0:
                    survey_id = predic_data[predic_data.anomaly_prediction == 1][SURVEY_ID_VAR].unique()
                else:
                    survey_id = predic_data[(predic_data.anomaly_prediction == 1) & 
                                            (predic_data[ENUMERATOR_ID_VAR].isin(selected_enumerator))][SURVEY_ID_VAR].unique()

                if len(survey_id) == 0:
                    st.warning("No surveys available for the selected enumerator(s) with anomalies.")

                if len(survey_id) > 0:
                    selected_survey = st.selectbox("Select a survey ID", survey_id)
                else:
                    selected_survey = None  
                    st.warning("Please select a valid enumerator or reset the filters.")

            if selected_survey is not None:
                df_display = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey]

                if len(selected_enumerator) > 0:
                    df_display = df_display[df_display[ENUMERATOR_ID_VAR].isin(selected_enumerator)]

                df_display = df_display.drop(columns=['anomaly_prediction', 'anomaly_score']).set_index(SURVEY_ID_VAR)
                df_display.rename(index=st.session_state.variable_mapping, columns=st.session_state.variable_mapping, inplace=True)

                col1, col2 = st.columns(2)
                with col1:
                    pt.make_subheader(f"Anomaly Prediction Rank: {predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_prediction'].index[0] + 1}")

                with col2:
                    pt.make_subheader(f"Anomaly Score: {round(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_score'].values[0], 3)}")

                st.dataframe(df_display)

                sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
                alt_plt_flag = False
                with sub_sub2_tab1:
                    pt.id_survey_shap_bar_plot_interactive(SURVEY_ID_VAR, selected_survey, features_label, 
                                                           shap_values, alt_plot=alt_plt_flag)
                with sub_sub2_tab2:
                    pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, selected_survey=selected_survey, 
                                                 data=features_label, shap_values=shap_values)

        with sub_tab2:
            pt.make_global_shap(shap_values, features, alt_plot=alt_plt_flag)

        with sub_tab3:
            pt.make_subheader(f"Enumerators without anomalies: {', '.join(map(str, enumerators_without_anomalies))}")

