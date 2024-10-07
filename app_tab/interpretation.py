import streamlit as st
import pandas as pd
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
from app_utils import plot_tools as pt

# Params
SURVEY_ID_VAR = 'audit_id'

def display():
    sub_tab1, sub_tab2 = st.tabs(["Local Interpretation", "Global Interpretation"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
    if "variable_mapping" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
        
    if st.session_state.ETL_output:
        # Extract Interpretation data
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        shap_values = interpretation_data['shap_values']  
        features_label = interpretation_data['features']
        features = interpretation_data['features'].iloc[:, 1:]

        with sub_tab1:

            #st.write(predic_data.columns)
            survey_id = predic_data[predic_data.anomaly_prediction == 1][SURVEY_ID_VAR]
            selected_survey = st.selectbox("Select a survey ID", survey_id)
            df_display = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey].drop(columns=['anomaly_prediction', 'anomaly_score']).set_index(SURVEY_ID_VAR)
            df_display.rename(index=st.session_state.variable_mapping, columns=st.session_state.variable_mapping, inplace=True)

            col1, col2 = st.columns(2)
            with col1:
                #st.subheader("Anomaly prediction")
                #st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_prediction'])
                pt.make_subheader("Anomaly Prediction Rank: {}".format(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_prediction'].index[0]+1))

            with col2:
                #st.subheader("Anomaly Score")
                #pt.st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_score'])
                pt.make_subheader("Anomaly Score: {}".format(round(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_score'].values[0], 3)))

            st.dataframe(df_display)
            #selected_index = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey].index[0]
            #pt.display_top_negative_shap_values(selected_index, shap_values, features)

            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
            with sub_sub2_tab1:
                pt.id_survey_shap_bar_plot_interactive(SURVEY_ID_VAR, 
                                           selected_survey, 
                                           features_label, 
                                           shap_values)
                # pt.id_survey_shap_bar_plot(SURVEY_ID_VAR, 
                #                            selected_survey, 
                #                            features_label, 
                #                            shap_values)
            with sub_sub2_tab2:
                pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, 
                                             selected_survey=selected_survey, 
                                             data=features_label,
                                             shap_values=shap_values)

        with sub_tab2:
            nb_features = len(features.columns)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
            st.pyplot(fig)

