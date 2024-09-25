import streamlit as st
import shap
import streamlit.components.v1 as components
import matplotlib.pyplot as plt

# Params
SURVEY_ID_VAR = 'audit_id'

# Functions 
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

def display():
     sub_tab1, sub_tab2 = st.tabs(["Global Interpretation", "Local Interpretation"])
     if st.session_state.ETL_output:
          # Extract Interpretation data
          interpretation_data =  st.session_state.ETL_output['SHAP_interpretation']
          predic_data = st.session_state.ETL_output['features_prediction_score']
          # model = interpretation_data['model']
          # explainer = interpretation_data['explainer']
          shap_values = interpretation_data['shap_values']
          features = interpretation_data['features']
          shap_data = features.drop(SURVEY_ID_VAR, axis=1)
     
          with sub_tab1:
               nb_features = len(shap_data.columns)
               fig, ax = plt.subplots()
               shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
               st.pyplot(fig)
          with sub_tab2:
               survey_id = features[SURVEY_ID_VAR]
               selected_survey = st.selectbox("Select an survey ID", survey_id)

               col1, col2 = st.columns(2)
               with col1:
                    st.subheader("Anomaly prediction")
                    st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_prediction'])
               with col2:
                    st.subheader("Anomaly Score")
                    st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_score'])

               sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
               with sub_sub2_tab1:
                    id_survey_shap_bar_plot(SURVEY_ID_VAR, 
                                             selected_survey, 
                                             features, 
                                             shap_values)
               with sub_sub2_tab2:
                    id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, 
                                                  selected_survey=selected_survey, 
                                                  data=features,
                                                  shap_values=shap_values)
