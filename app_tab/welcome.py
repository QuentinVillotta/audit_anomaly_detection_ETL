import streamlit as st
import os
import yaml
import pickle
from app_utils import data_io
from app_utils import kobo_tools

# Parameters
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
     
    st.title("Welcome to the Audit Anomaly Detection App")
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Configuration Form")
        updated_config = data_io.render_form(config)

        missing_fields = []
        for field in MANDATORY_FIELDS:
            keys = field.split('.')  
            current_dict = updated_config
            for key in keys:
                current_dict = current_dict.get(key)
                if current_dict is None:
                    break
            if not current_dict:
                missing_fields.append(field)

        if missing_fields:
            st.error(f"Mandatory fields missing: {', '.join(missing_fields)}")

        updated_config['url'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/data/?format=json"
        updated_config['url_questionnaire'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/?format=json"

    with col2:
        st.header("User Guide")
        st.write("""
        1. Complete the form on the left.
        2. Click the "Go" button to check the Kobo API connection and start the analysis.
        3. The results will appear in the appropriate tabs after the analysis.
        """)

        st.header("Run Anomaly Detection")
        
        if st.button("Go"):
            if not missing_fields:
                
                # Then, check Kobo API connectivity after saving
                credentials = updated_config.get('kobo_credentials', {})
                status, message = kobo_tools.HTTP_check_kobo_api(updated_config['url'], credentials)
                
                if status == "Success":              
                    # Save the updated config to YAML first
                    with open(OUTPUT_FILE_PATH, 'w') as file:
                        yaml.dump(updated_config, file, default_flow_style=False)

                    # Run Kedro pipeline and display logs
                    data_io.run_kedro_pipeline()

                    # Load Pickle file if the pipeline ran successfully
                    if os.path.exists(MODEL_OUTPUT_PATH):
                        with open(MODEL_OUTPUT_PATH, 'rb') as pickle_file:
                            ETL_output = pickle.load(pickle_file) 
                        st.session_state.ETL_output = ETL_output
                    else:
                        st.error("Pipeline error: output model file not found.")
                
                else:
                    st.error(message)
            else:
                st.error("Please fill out all mandatory fields.")