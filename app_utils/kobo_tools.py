import requests
from requests.auth import HTTPBasicAuth
import yaml
import re

def HTTP_check_kobo_api(url, credentials):
    headers = {
        "Authorization": f"{credentials}"
    }
    try:
        # Tenter de se connecter à l'API Kobo avec le token fourni
        response = requests.get(url, headers=headers)
        print(response)
        if response.status_code == 200:
            return "Success", "Successfully connected to the Kobo API!"
        elif response.status_code == 401:
            return "Error", "Error 401: Unauthorized. The provided token is invalid."
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
