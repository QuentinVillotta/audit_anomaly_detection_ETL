import pandas as pd
import yaml
import os
import io
import streamlit as st
from kedro.framework.session import KedroSession
from kedro.framework.startup import bootstrap_project
from app_utils import kobo_tools

#doesn't remove them after refresh
def clear_outputs():
    st.session_state.ETL_output = {}


def format_string(input_string):
    return input_string.replace('_', ' ').title()

# Welcome Page - Configuration Form 
def delete_file(file_path):
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        st.write(f"Error deleting file {file_path}: {e}")

def render_form(config):
    updated_dict = {}
    
    # Asset UID Section
    updated_dict['asset_uid'] = st.text_input("**Enter KoBo Project ID (asset_uid)**", value='a7Ttgjamsjep2KJxXZ4vbt')
    
    # Kobo Server Section
    kobo_servers = ['kobo.impact-initiatives.org', 'eu.kobotoolbox.org', 'Other']
    selected_server = st.selectbox("**Select KoBo server**", kobo_servers, index=0)
    if selected_server == 'Other':
        updated_dict['kobo_server'] = st.text_input("Enter custom KoBo server host:", value="")
    else:
        updated_dict['kobo_server'] = selected_server
    
    # Initialize attempt counter in session state if it doesn't exist
    if 'login_counter' not in st.session_state:
        st.session_state.login_counter = 0

    # Upload Kobo credentials
    uploaded_file = st.file_uploader("**Upload Kobo credentials (YAML file)**", type=["yml", "yaml"])
    
    if uploaded_file:
        try:
            # Load the YAML data from the uploaded file
            yaml_data = yaml.safe_load(uploaded_file)
            # Check Kobo token format
            valid, message = kobo_tools.check_kobo_credentials_format(yaml_data)
            
            if valid:
                updated_dict['kobo_credentials'] = yaml_data.get("kobo_credentials", "")
                st.session_state.login_counter = 0
                st.success("Credentials are valid!")
            else:
                st.session_state.login_counter += 1
                st.warning(message)

                #More than 3 failed attempts
                if st.session_state.login_counter == 3:
                    st.image("www/wrong_format_credentials.gif") 
                    st.warning("You have entered invalid credentials three times. Please check your input.")
                if st.session_state.login_counter >= 5:
                    st.image("www/wrong_format_credentials_jp.gif") 
                    st.error("Invalid credential: check your token format and value on your Kobo personal settings.")

        except yaml.YAMLError as e:
            st.error(f"YAML parsing error: {e}")
        except Exception as e:
            st.error(f"Unexpected error: {e}")

    # Raw Data Columns Section (in an expander with name and type in two columns)
    name_mapping = {
        "enum_id": "enumerator ID",
        "audit_id": "survey ID",
        "start_time": "start"
    }
    # Expander pour les colonnes des raw data
  
    for key, value in config.items():
        updated_dict[key] = {}
        # Définir le nom de l'expander et son état (ouvert ou fermé par défaut)
        expander_name = f"**{format_string(key)}**"
        expanded_flag = (key == "raw_data_columns")
        # Expander pour chaque section (Raw Data ou autre)
        with st.expander(expander_name, expanded=expanded_flag):
            # Parcourir chaque colonne et ses propriétés
            for column, properties in value.items():
                display_name = name_mapping.get(column, column)  # Utiliser le nom personnalisé si disponible
                updated_dict[key][column] = {}
                # Deux colonnes pour organiser le formulaire (nom et type)
                col1, col2 = st.columns(2)
                with col1:
                    # Champ pour le mapping (nom de la variable dans le formulaire Kobo)
                     updated_dict[key][column]['mapping'] = st.text_input(
                        f"**{display_name}** : Variable name in Kobo form", 
                        value=properties.get('mapping', ''),
                        key=f"mapping_{column}"
                    )
                with col2:
                    # Sélectionner le type de variable (dtype)
                     updated_dict[key][column]['dtype'] = st.selectbox(
                        f"Type", ['str', 'float'], 
                        index=0 if properties.get('dtype') in ['str', None] else 1,
                        key=f"dtype_{column}"
                    )
    return updated_dict

# ## Kedro ETL manager
def run_kedro_pipeline():

    bootstrap_project(os.getcwd())
    # Initialize Kedro session (assuming you're in a Kedro project directory)
    placeholder = st.empty()
    with placeholder.container():
        with KedroSession.create() as session:
            # Run the pipeline (use the name of the pipeline you want to execute, or default)
            st.info("Running the pipeline, please wait ...")
            session.run(pipeline_name="__default__")
    placeholder.empty()
    st.success("Pipeline has finished")
        


# Function to save plots as PNG and return bytes
def save_plot_as_png(fig, format="eps", dpi=1000):
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi)
    buf.seek(0)
    return buf