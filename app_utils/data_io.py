import pandas as pd
import yaml
import os
import subprocess
import io
import streamlit as st
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project


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

## Kedro ETL manager
# def run_kedro():
#     with st.spinner("Running the pipeline, this may take a few minutes. Please wait."):
#         process = subprocess.Popen(['kedro', 'run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

#         # Use a placeholder for logs
#         log_placeholder = st.empty()

#         # List to store the last few log lines
#         log_lines = []
#         # Dask Daskboard
#         dashboard_link = "http://127.0.0.1:8787"
#         st.components.v1.iframe(src=dashboard_link, height=600)
#         # Read the logs in real-time
#         while True:
#             output = process.stdout.readline()
#             if output == '' and process.poll() is not None:
#                 break
#             if output:
#                 # Add the line to the logs
#                 log_lines.append(output.strip())

#                 # Keep only the last 20 lines
#                 if len(log_lines) > 12:
#                     log_lines.pop(0)

#                 # Display the last 5 lines in the placeholder
#                 log_placeholder.text("\n".join(log_lines))
#         # Check for stderr output (errors)
#         stderr_output = process.stderr.read()
#         if stderr_output:
#             log_placeholder.error(stderr_output.strip())
#             st.error("An error occurred while running the pipeline. Please check the logs.")
#         else:
#             st.success("Pipeline run completed successfully!")

def run_kedro_pipeline():

    bootstrap_project(os.getcwd())

    # Initialize Kedro session (assuming you're in a Kedro project directory)
    with KedroSession.create() as session:
        placeholder = st.empty()
        with placeholder.container():
        # Run the pipeline (use the name of the pipeline you want to execute, or default)
            st.info("Running the full pipeline...")
            session.run(pipeline_name="__default__")  # Or specify your custom pipeline name

        # #Access the Kedro context and catalog after the pipeline has run
        # context = session.load_context()
        # catalog = context.catalog
        
        # # Retrieve objects from the catalog
        # output_data = catalog.load("01_raw_data")  # Replace with the name of your dataset

        # output_data = context.run(pipeline_name='data_download')
        # raw_data = output_data['01_raw_data']
        #Display the results in Streamlit
            
        # st.dataframe(output_data)
        placeholder.empty()
        st.success("Pipeline has finished")



# Function to save plots as PNG and return bytes
def save_plot_as_png(fig, format="eps", dpi=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf