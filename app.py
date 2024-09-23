import streamlit as st
import yaml
import subprocess
import os

# App Parameters
# Set the page to open in wide layout mode
st.set_page_config(layout="wide")

# Load the initial YAML file from the specified path
input_file_path = os.path.join("conf", "base", "globals_template.yml")
output_file_path = os.path.join("conf", "base", "globals.yml")

# Output files path to export
model_output_path = os.path.join("data", "05_model_output", "output_model.pkl")
prediction_path = os.path.join("data", "05_model_output", "raw_features_predictions_and_scores.xlsx")

def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        else:
            st.write(f"File not found: {file_path}")
    except Exception as e:
        st.write(f"Error deleting file {file_path}: {e}")


# Init counter click button
if 'click_count' not in st.session_state:
    st.session_state.click_count = 0

if st.session_state.click_count == 0:  
    delete_file(model_output_path)
    delete_file(prediction_path)


with open(input_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Application title
st.title("Configuration Form")


# Function to display and modify a dictionary in a form
def render_form(config_dict, parent_key='', inside_expander=False):
    updated_dict = {}
    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key  # Unique key for each field
        
        if key in ['url', 'url_questionnaire']:  # Skip these parameters
            continue
        
        if key in ['raw_data_columns', 'audit_columns', 'questionnaire_columns'] and not inside_expander:
            with st.expander(f"{key}:"):
                updated_dict[key] = render_form(value, full_key, inside_expander=True)
        else:
            if isinstance(value, dict) and 'mapping' in value and 'dtype' in value:
                st.write(f"{key}")
                
                # Input field for 'mapping'
                updated_dict[key] = {}
                updated_dict[key]['mapping'] = st.text_input(f"{key} - Variable Name", value=value.get('mapping', ''))
                
                # Selectbox for 'dtype'
                updated_dict[key]['dtype'] = st.selectbox(f"{key} - Variable Type", ['str', 'float'], index=0 if value.get('dtype') == 'str' else 1)

            elif key == "kobo_server":
                updated_dict[key] = st.selectbox(
                    f"{key}", 
                    ['kobo.impact-initiatives.org', 'eu.kobotoolbox.org', 'Other'], 
                    index=0,
                    key=full_key
                )
                if updated_dict[key] == 'Other':
                    updated_dict[key] = st.text_input("Enter custom kobo_server:", key=f"custom_{full_key}")

            elif key == "kobo_credentials":
                uploaded_file = st.file_uploader(f"Upload {key} YAML", type="yml", key=full_key)
                if uploaded_file is not None:
                    yaml_data = yaml.safe_load(uploaded_file)
                    updated_dict[key] = yaml_data.get("kobo_credentials", "")

            elif value is None or isinstance(value, str):
                updated_dict[key] = st.text_input(f"{key}", value="" if value is None else value, key=full_key)
            
            elif isinstance(value, float):
                updated_dict[key] = st.number_input(f"{key}", value=value, key=full_key)

            elif isinstance(value, int):
                updated_dict[key] = st.number_input(f"{key}", value=value, step=1, key=full_key)

            else:
                updated_dict[key] = st.text_input(f"{key}", value=str(value), key=full_key)

    return updated_dict

# Mandatory fields
mandatory_fields = ["asset_uid", "project_short", "kobo_credentials", 
                    "raw_data_columns.enum_id.mapping", "raw_data_columns.enum_id.dtype",
                    "raw_data_columns.audit_id.mapping", "raw_data_columns.audit_id.dtype"]

col1, col2 = st.columns(2) 

with col1:
    st.header("Parameters")
    updated_config = render_form(config)

    missing_fields = []
    for field in mandatory_fields:
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

# Function to run the Kedro pipeline and capture logs
def run_kedro():
    with st.spinner("Running the pipeline, this may take a few minutes. Please wait."):
        process = subprocess.Popen(['kedro', 'run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Use a placeholder for logs
        log_placeholder = st.empty()

        # List to store the last few log lines
        log_lines = []

        # Read the logs in real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                # Add the line to the logs
                log_lines.append(output.strip())

                # Keep only the last 20 lines
                if len(log_lines) > 17:
                    log_lines.pop(0)

                # Display the last 5 lines in the placeholder
                log_placeholder.text("\n".join(log_lines))

        # Check for stderr output (errors)
        stderr_output = process.stderr.read()
        if stderr_output:
            log_placeholder.error(stderr_output.strip())
            st.error("An error occurred while running the pipeline. Please check the logs.")
        else:
            st.success("Pipeline run completed successfully!")

with col2:
    st.header("Run")

    if st.button("Run Anomaly Detection"):
        st.session_state.click_count += 1
        if not missing_fields:
            with open(output_file_path, 'w') as file:
                yaml.dump(updated_config, file, default_flow_style=False)

            # Run kedro and display logs
            run_kedro()
        else:
            st.error("Please fill out all mandatory fields.")

    st.header("Outputs")  
    if os.path.exists(model_output_path):
        with open(model_output_path, 'rb') as f:
            st.download_button('Download output file', f, file_name='output_model.pkl')
    if os.path.exists(prediction_path):
        with open(prediction_path, 'rb') as f:
            st.download_button('Download prediction file', f, file_name='prediction.xlsx')

