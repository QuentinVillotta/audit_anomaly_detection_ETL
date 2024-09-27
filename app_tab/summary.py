import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# To order by nb d'annomaly

def display():
     df = st.session_state.ETL_output['features_prediction_score']
     sub_tab1, sub_tab2 = st.tabs(["General", "Enumerator"])
 
     with sub_tab1:
          col1, col2 = st.columns([2, 4])
          with col1:
          # 1. General indicator - % of detected anomalies
               total_surveys = len(df)
               total_anomalies = df['anomaly_prediction'].sum()
               # Pie chart of detected anomalies
               pie_data = pd.DataFrame({
               "Type": ["Anomalies", "Non-Anomalies"],
               "Count": [total_anomalies, total_surveys - total_anomalies]
               })
               fig_pie = px.pie(pie_data, values='Count', names='Type', title="Percentage of Detected Anomalies")
               # Display the pie chart
               st.plotly_chart(fig_pie)
          with col2:
               # summary_stats = df.describe()
               st.dataframe(df)

     with sub_tab2:
          # fig, ax = plt.subplots()
          # import pandas as pd
          # import numpy as np
          # chart_data = pd.DataFrame(np.random.randn(20, 3), columns=["a", "b", "c"])
          survey_count = df.groupby(['enum_id', 'anomaly_prediction']).size().unstack(fill_value=0)
          survey_count.columns = ['No Anomaly', 'Anomaly']
          st.bar_chart(survey_count)
          # sns.countplot(data=df, x="enum_id", hue="anomaly_prediction", ax=ax)
          # ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
          # st.pyplot(fig)
