import requests
import re
import pandas as pd

def HTTP_check_kobo_api(url, credentials):
    headers = {
        "Authorization": f"{credentials}"
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return "Success", "Successfully connected to the Kobo API!"
        elif response.status_code == 401:
            return "Error", "Error 401: Unauthorized.  Please check the kobo server you have chosen and your token."
        elif response.status_code == 403:
            return "Error", "Error 403: Access forbidden. Please check your permissions on the kobo project and your token."
        elif response.status_code == 404:
            return "Error", "Error 404: Resource not found. Please check the `asset_uid`, your permissions on the kobo project and your token."
        else:
            return "Error", f"Error {response.status_code}: Problem connecting to the API."
    except requests.RequestException as e:
        return "Error", f"Connection error: {e}"

def check_kobo_credentials_format(yaml_data):
    # Check if the file contains the required 'kobo_credentials' key
    if 'kobo_credentials' not in yaml_data:
        return False, "The uploaded file is missing the 'kobo_credentials' field."
    
    # Retrieve the token value
    token = yaml_data['kobo_credentials']
    
    # Validate the token format using a regex (for a typical API token structure)
    token_pattern = r"^Token\s[a-fA-F0-9]{40}$"  # Adjust the regex as needed for your specific token format
    if not re.match(token_pattern, token):
        return False, "Invalid token format. Expected format: 'Token [40-character alphanumeric token]'."
    
    # If all checks pass
    return True, "The 'kobo_credentials' format is valid."

def check_kobo_columns(url, raw_data_columns, credentials):
    """
    Downloads the Kobo metadata from the provided URL and verifies if the column names 
    specified in raw_data_columns are present.

    Parameters:
    - url (str): The URL to download the Kobo data in JSON format.
    - raw_data_columns (dict): Dictionary containing raw data fields with 'mapping'.
    - credentials (str): Authorization token or credentials for accessing the Kobo form.

    Returns:
    - tuple: ("Error", message) if columns are missing or an issue occurs, otherwise ("Success", message).
    """
    
    # Step 1: Download the Kobo form (JSON) from the API using credentials
    headers = {
        "Authorization": f"{credentials}"
    }
    # Extract only 1 row to get the metadata
    url_metadata = f"{url}&limit=1"
    try:
        response = requests.get(url_metadata, headers=headers)
        response.raise_for_status()  # Check if the request was successful
        data = response.json()['results']
       
    except requests.exceptions.RequestException as e:
        return "Error", f"Failed to download Kobo data: {e}"
        
    # Step 2: Extract the 'survey' fields and convert them into a DataFrame
    try:
        columns = pd.DataFrame(data).columns
        print(columns)
    except KeyError:
        return "Error", "Unexpected format in the Kobo data response."

    # Step 3: Verify that the specified columns in raw_data_columns exist in the Kobo form
    missing_columns = []
    for column, properties in raw_data_columns.items():
        mapping_name = properties.get('mapping', '')
        if mapping_name not in columns:
            missing_columns.append(mapping_name)

    # Step 4: Return an error message if columns are missing, or success if all are present
    if missing_columns:
        missing_fields = '\n'.join([f"- {col}" for col in missing_columns])
        return "Error", f"The following fields are missing in the Kobo data:\n\n{missing_fields}\n\nPlease check that you have typed the variable(s) name(s) correctly."
    else:
        return "Success", "All specified fields are present in the Kobo form."