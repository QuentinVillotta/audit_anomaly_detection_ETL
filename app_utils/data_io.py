import pandas as pd
import yaml
import os
import io
import streamlit as st
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from app_utils import kobo_tools

# Welcome Page - Configuration Form 
def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.write(f"Error deleting file {file_path}: {e}")

## Function to display and modify a dictionary in a form
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
                uploaded_file = st.file_uploader(f"Upload Kobo credentials YAML file", type=["yml", "yaml"], key=full_key)
                if uploaded_file:
                    try:
                        # Load the YAML data from the uploaded file
                        yaml_data = yaml.safe_load(uploaded_file)
                        # Check Kobo token format
                        valid, message = kobo_tools.check_kobo_credentials_format(yaml_data)
                        if valid:
                            updated_dict[key] = yaml_data.get("kobo_credentials", "")
                        else:
                            # st.error(message)
                            st.image("www/wrong_format_credentials.gif") 
                    except yaml.YAMLError as e:
                        st.error(f"YAML parsing error: {e}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                    
            elif value is None or isinstance(value, str):
                updated_dict[key] = st.text_input(f"{key}", value="" if value is None else value, key=full_key)
            
            elif isinstance(value, float):
                updated_dict[key] = st.number_input(f"{key}", value=value, key=full_key)

            elif isinstance(value, int):
                updated_dict[key] = st.number_input(f"{key}", value=value, step=1, key=full_key)

            else:
                updated_dict[key] = st.text_input(f"{key}", value=str(value), key=full_key)

    return updated_dict

## Kedro ETL manager
def run_kedro_pipeline():

    bootstrap_project(os.getcwd())
    # Initialize Kedro session (assuming you're in a Kedro project directory)
    with KedroSession.create() as session:
        placeholder = st.empty()
        with placeholder.container():
        # Run the pipeline (use the name of the pipeline you want to execute, or default)
            st.info("Running the full pipeline...")
            session.run(pipeline_name="__default__")  # Or specify your custom pipeline name

        placeholder.empty()
        st.success("Pipeline has finished")


# Function to save plots as PNG and return bytes
def save_plot_as_png(fig, format="eps", dpi=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf