import streamlit as st
import yaml
import os
from app_utils import data_io
from app_utils import plot_tools as pt
from app_tab import welcome, summary, visualisation, interpretation

import pickle
import logging

# Set Logging Level
# Kedro
logging.getLogger('kedro').setLevel(logging.INFO)
# # Dask logger
logging.getLogger('distributed').setLevel(logging.ERROR)
# # Streamlit
logging.getLogger('streamlit').setLevel(logging.INFO)

# os.environ['KEDRO_LOGGING_CONFIG'] = os.path.join("conf", "logging.yml")


### APP PARAMETERS
# Set the page to open in wide layout mode
st.set_page_config(page_title="Audit Anomaly Detection", layout="wide")

# Output files path to export
MODEL_OUTPUT_PATH = os.path.join("data", "05_model_output", "output_model.pkl")
PREDICTION_PATH = os.path.join("data", "05_model_output", "raw_features_predictions_and_scores.xlsx")

# Init counter click button
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0
    
if 'variable_names_excel' not in st.session_state:
    st.session_state['variable_names_excel'] = os.path.join("conf", "base", 
                                                            "variable_map_explanation.xlsx")
if "variable_mapping" not in st.session_state:
    st.session_state["variable_mapping"] = pt.load_variable_attribute("var_name", "var_label")

if "variable_meaning" not in st.session_state:
    st.session_state["variable_meaning"] = pt.load_variable_attribute("var_name", "str_comment")


if 'ETL_output' not in st.session_state:
    st.session_state.ETL_output = {}
    if os.path.exists(MODEL_OUTPUT_PATH):
        with open(MODEL_OUTPUT_PATH, 'rb') as pickle_file:
            ETL_output = pickle.load(pickle_file)
        st.session_state.ETL_output = ETL_output

st.title("Anomaly Detection")
st.header("User Guide", divider="gray")
st.write("""\
        The app will allow you to easily download, process and analyse the audit files collected
        for your project. It will run a anomaly detection solution to flag suspicios surveys.\
        Please follow the instructions below for running the solution or import its results. \
        """)

if st.button("Clear Tabs/Outputs"):
    data_io.clear_outputs()
# if st.session_state.click_count == 0:  
#     data_io.delete_file(MODEL_OUTPUT_PATH)
#     data_io.delete_file(PREDICTION_PATH)

# Menu/Tabs app
list_tab = ["Welcome", "Summary", "Interpretation", "Visualization"]
welcome_tab, summary_tab, interpretation_tab, visualization_tab= st.tabs(list_tab)

with welcome_tab:
    welcome.display()

if st.session_state.ETL_output:
    with summary_tab: 
        summary.display()

    with interpretation_tab: 
        interpretation.display()

    with visualization_tab: 
        visualisation.display()
else:
    with summary_tab:
        st.write("No data available.  \
                 Please run the Anomaly Detection analysis on the welcome page or  \
                 import new results.")
    with interpretation_tab:
        st.write("No data available.  \
                 Please run the Anomaly Detection analysis on the welcome page or  \
                 import new results.")
    with visualization_tab:
        st.write("No data available.  \
                 Please run the Anomaly Detection analysis on the welcome page or  \
                 import new results.")
    

