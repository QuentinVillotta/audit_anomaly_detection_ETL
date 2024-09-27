import streamlit as st
from app_utils import data_io
import os
import yaml
import pickle

# Parameters
# Load the initial YAML file from the specified path
INPUT_FILE_PATH = os.path.join("conf", "base", "globals_template.yml")
OUTPUT_FILE_PATH = os.path.join("conf", "base", "globals.yml")
MODEL_OUTPUT_PATH = os.path.join("data", "05_model_output", "output_model.pkl")
PREDICTION_PATH = os.path.join("data", "05_model_output", "raw_features_predictions_and_scores.xlsx")


# Mandatory fields
MANDATORY_FIELDS = ["asset_uid", "project_short", "kobo_credentials", 
                    "raw_data_columns.enum_id.mapping", "raw_data_columns.enum_id.dtype",
                    "raw_data_columns.audit_id.mapping", "raw_data_columns.audit_id.dtype"]

def display():

    with open(INPUT_FILE_PATH, 'r') as file:
        config = yaml.safe_load(file)
     
    # Welcome page with user guide and form
    st.title("Welcome to the Audit Anomaly Detection App")
    col1, col2 = st.columns([2, 1])

    # Left column: User guide
    with col1:
        st.header("Configuration Form")

        updated_config = data_io.render_form(config)

        missing_fields = []
        for field in MANDATORY_FIELDS:
            keys = field.split('.')  # Split the field into a list of keys
            current_dict = updated_config
            for key in keys:
                current_dict = current_dict.get(key)  # Access each level
                if current_dict is None:
                    break
            if not current_dict:
                missing_fields.append(field)

        if missing_fields:
            st.error(f"Mandatory fields missing: {', '.join(missing_fields)}")

        # Generate URLs in the output, do not display them in the form
        updated_config['url'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/data/?format=json"
        updated_config['url_questionnaire'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/?format=json"

    with col2:
        st.header("User Guide")
        st.write("""
        1. Complete the form on the left.
        2. Click the "Go" button to start the analysis.
        3. Once the analysis is complete, the results will appear in the Summary, Visualization, and Interpretation tabs.
        """)

        st.header("Run Anomaly Detection")
        if st.button("Go"):
            st.session_state.click_count += 1
            if not missing_fields:
                with open(OUTPUT_FILE_PATH, 'w') as file:
                    yaml.dump(updated_config, file, default_flow_style=False)

                # Run kedro and display logs
                # data_io.run_kedro()
                data_io.run_kedro_pipeline()

                # Load Pickle file
                if os.path.exists(MODEL_OUTPUT_PATH):
                    with open(MODEL_OUTPUT_PATH, 'rb') as pickle_file:
                        ETL_output = pickle.load(pickle_file) 
                    st.session_state.ETL_output = ETL_output
                else:
                    st.error("Please re run the pipeline")

            else:
                st.error("Please fill out all mandatory fields.")