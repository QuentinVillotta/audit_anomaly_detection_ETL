# Audit Anomaly Detection App

This application detects anomalies in survey data based on Kobo audit files. Using a machine learning approach, it automatically identifies potentially fraudulent surveys by analyzing abnormal patterns. The project is structured as a Kedro pipeline that downloads audit files from the Kobo API, processes the data, and predicts anomalies. The results are accessible through a Streamlit web application, allowing users to configure and explore them interactively.

## Installation Guide

### Prerequisites

To run the application, you need to have Docker installed on your machine. Below is a step-by-step guide to install Docker:

#### For Windows:

1. **Download Docker Desktop for Windows**  
   Visit [Docker for Windows](https://docs.docker.com/desktop/install/windows-install/) and download Docker Desktop.

2. **Install Docker Desktop**  
   Double-click the downloaded `.exe` file and follow the installation instructions.

3. **Launch Docker**  
   After installation, launch Docker Desktop. Ensure that it is running by checking the Docker icon in your system tray.

4. **Verify Installation**  
   Open a terminal (or PowerShell) and run the following command:
   ```bash
   docker --version
    ```

For more detailed instructions, refer to the [official Docker documentation](https://docs.docker.com/get-docker/).

### Pulling the Docker Image

To download the Docker image from Docker Hub, run the following command:
```bash
docker pull qvillo/audit_anomaly_detection_app
```

## Running the Application

After pulling the image, you can run the application using Docker. Simply run:
```bash
docker run -p 8501:8501 -p 8787:8787 qvillo/audit_anomaly_detection_app
```

## Access the Application

Once the container is running, open your web browser and go to: [http://localhost:8501](http://localhost:8501) to access the application.

## License

This project is licensed under the MIT License.