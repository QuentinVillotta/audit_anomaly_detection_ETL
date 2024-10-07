import streamlit as st
from app_utils import data_io

def display():
    st.header("User Guide", divider="gray")
    st.write("""\
        Welcome to the Audit File Anomaly Detection Application! This app allows you to easily:
        - **Download**: Retrieve the audit files collected for your project directly from KoboToolbox.
        - **Process**: Run the anomaly detection solution to analyze the data and flag suspicious surveys.
        - **Analyze**: Review the flagged surveys and gain insights from the results.
        ### How to Use the Application:
        1. **Go to Run Anomaly Detection tab and choose an action**:
            - **Run anomaly detection pipeline**: Fill out the necessary configuration fields and run the anomaly detection pipeline.
            - **Import previous Results**: Upload previously exported anomaly detection results.
        2. **Explore the results**: 
             * After running the analysis, navigate through the different tabs to examine your findings:
                - **Summary**: Overview of flagged surveys and key statistics.
                - **Interpretation**: Detailed analysis of the results and what they mean.
                - **Visualization**: Graphical representation of the data for better insights.
            """)

    # if st.button("Clear Tabs/Outputs"):
    #     data_io.clear_outputs()