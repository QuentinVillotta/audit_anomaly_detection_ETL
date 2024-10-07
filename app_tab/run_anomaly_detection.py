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
MANDATORY_FIELDS = ["asset_uid", "kobo_credentials", 
                    "raw_data_columns.enum_id.mapping", "raw_data_columns.enum_id.dtype",
                    "raw_data_columns.audit_id.mapping", "raw_data_columns.audit_id.dtype"]
EXPORT_FLAG = False

def display():

    global EXPORT_FLAG  

    # Choix de l'utilisateur : Importer les résultats ou utiliser le formulaire de configuration
    st.header("Choose an action", divider="gray")
    action = st.radio("", ("Run anomaly detection pipeline", "Import previous Results"), index=0)

    if action == "Import previous Results":
        # Importer les résultats
        st.header("Import Results")
        st.write("""\
        1. Upload the model results exported previously
        2. Make sure it is a pickle file
        3. The results will appear in the appropriate tabs after the uploading.
        """)
        import_results = st.file_uploader("Upload a Pickle file", type=['pkl'])

        if import_results is not None:
            if not import_results.name.endswith('.pkl'):
                st.error("Error: Filename must be a pickle and end with '.pkl'")
            else:
                ETL_output = pickle.load(import_results)
                st.session_state.ETL_output = ETL_output
                st.success("Data imported successfully!")
    
    elif action == "Run anomaly detection pipeline":
        # Utiliser le formulaire de configuration
        with open(INPUT_FILE_PATH, 'r') as file:
            config = yaml.safe_load(file)
        
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

            updated_config['url'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/data/?format=json"
            updated_config['url_questionnaire'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/?format=json"

        with col2:
            st.header("Run Anomaly Detection", divider="gray")
            st.write("""\
            1. Complete the form on the left.
            2. Click the "Go" button to check the Kobo API connection and start the analysis.
            3. The results will appear in the appropriate tabs after the analysis.
            """)
            output_dict = {}                    
            output_dict["export_name"] = st.text_input("Enter filename for export:", value="model_output.pkl")

            if output_dict["export_name"].endswith('.pkl'):
                EXPORT_FLAG = True
            else:
                st.error("Error: Please enter a valid filename ending with '.pkl'")
            
            if st.button("Go"):
                if not missing_fields and EXPORT_FLAG:
                    credentials = updated_config.get('kobo_credentials', {})
                    http_check_status, http_check_message = kobo_tools.HTTP_check_kobo_api(updated_config['url'], credentials)
                    
                    if http_check_status == "Success":
                        col_check_status, col_check_message = kobo_tools.check_kobo_columns(url=updated_config['url'],
                                                                                            raw_data_columns=updated_config['raw_data_columns'],
                                                                                            credentials=credentials)      
                        if col_check_status == "Success":
                            with open(OUTPUT_FILE_PATH, 'w') as file:
                                yaml.dump(updated_config, file, default_flow_style=False)

                            data_io.run_kedro_pipeline()

                            if os.path.exists(MODEL_OUTPUT_PATH):
                                with open(MODEL_OUTPUT_PATH, 'rb') as pickle_file:
                                    ETL_output = pickle.load(pickle_file) 
                                    
                                st.session_state.ETL_output = ETL_output
                                st.success("Data processing complete! Your download is now ready. \
                                            Press the download button to save the results.")
                                st.download_button(label='Download Results',
                                                   data=pickle.dumps(ETL_output),
                                                   file_name=output_dict["export_name"],
                                                   mime='application/octet-stream',
                                                   key="export_button")

                            else:
                                st.error("Pipeline error: output model file not found.")
                        else:
                            st.error(col_check_message)
                    else:
                        st.error(http_check_message)
                else:
                    st.error(f"Please complete all mandatory fields. The following are missing: **{', '.join(missing_fields)}**")
