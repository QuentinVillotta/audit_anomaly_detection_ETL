import streamlit as st
import yaml
import subprocess
import os

# Charger le fichier YAML initial depuis le chemin spécifié
input_file_path = "conf/base/globals_generic.yml"
output_file_path = "conf/base/globals.yml"

with open(input_file_path, 'r') as file:
    config = yaml.safe_load(file)

# Titre de l'application
st.title("Configuration Form")

# Fonction pour afficher et modifier un dictionnaire dans un formulaire
def render_form(config_dict, parent_key='', inside_expander=False):
    updated_dict = {}
    for key, value in config_dict.items():
        full_key = f"{parent_key}.{key}" if parent_key else key  # Unique key for each field
        
        if key in ['url', 'url_questionnaire']:  # Ne pas afficher ces paramètres
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

# Champs obligatoires
mandatory_fields = ["asset_uid", "project_short", "kobo_credentials", 
                    "raw_data_columns.enum_id.mapping", "raw_data_columns.enum_id.dtype",
                    "raw_data_columns.audit_id.mapping", "raw_data_columns.audit_id.dtype"]

col1, col2 = st.columns(2) 

with col1:
    st.header("Parameters")
    updated_config = render_form(config)

    missing_fields = []
    for field in mandatory_fields:
        keys = field.split('.')  # Divisez les champs en une liste de clés
        current_dict = updated_config
        for key in keys:
            current_dict = current_dict.get(key)  # Accédez à chaque niveau
            if current_dict is None:
                break
        if not current_dict:
            missing_fields.append(field)


    if missing_fields:
        st.error(f"Mandatory fields missing: {', '.join(missing_fields)}")

    # Générer les URLs dans l'output, ne pas les afficher dans le formulaire
    updated_config['url'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/data/?format=json"
    updated_config['url_questionnaire'] = f"https://{updated_config['kobo_server']}/api/v2/assets/{updated_config['asset_uid']}/?format=json"


def run_kedro():
    with st.spinner("Running the pipeline, this may take a few minutes. Please wait."):
        process = subprocess.Popen(['kedro', 'run'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Utilisation d'un espace réservé pour les logs
        log_placeholder = st.empty()

        # Lire les logs en temps réel
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                log_placeholder.text(output.strip())

        # Vérifier les erreurs de sortie
        stderr_output = process.stderr.read()
        if stderr_output:
            log_placeholder.error(stderr_output.strip())
            st.error("An error occurred while running pipeline. Please check the logs.")
        else:
            st.success("Pipeline run completed successfully!")

with col2:
    st.header("Output")
    if st.button("Save and Run Pipeline"):
        if not missing_fields:
            with open(output_file_path, 'w') as file:
                yaml.dump(updated_config, file, default_flow_style=False)
            # st.success(f"File saved as '{output_file_path}'.")

            # Exécutez kedro run et montrez les logs
            run_kedro()

            # Affichez le bouton de téléchargement
            if os.path.exists('data/05_model_output/output_model.pkl'):
                with open('data/05_model_output/output_model.pkl', 'rb') as f:
                    st.download_button('Download output file', f, file_name='output_model.pkl')

        else:
            st.error("Please fill out all mandatory fields.")
