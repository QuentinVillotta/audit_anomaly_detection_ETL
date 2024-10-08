import streamlit as st
import pandas as pd
from app_utils import plot_tools as pt
import plotly.express as px

def display():
     df = st.session_state.ETL_output['features_prediction_score']
     sub_tab1, sub_tab2 = st.tabs(["General", "Enumerator"])
     with sub_tab1:
          col1, col2 = st.columns([1, 5])
          with col1:
               fig_pie = pt.pie_chart_pct_anomalies(df)
               st.plotly_chart(fig_pie)
          with col2:
               st.dataframe(df.set_index('audit_id'), hide_index= False)

     with sub_tab2:
          anomaly_count_enum_plot, anomaly_count_enum_df = pt.plot_anomaly_count(df)
          st.plotly_chart(anomaly_count_enum_plot)
          # Sort DF
          anomaly_df = anomaly_count_enum_df[anomaly_count_enum_df['Anomaly Type'] == 'Anomaly']
          # Sort the filtered DataFrame by 'Count' in descending order
          sorted_anomaly_df = anomaly_df.sort_values(by='Count', ascending=False)
          # Combine back with the rest of the DataFrame (No Anomaly rows)
          df_sorted = pd.concat([sorted_anomaly_df, anomaly_count_enum_df[anomaly_count_enum_df['Anomaly Type'] == 'No Anomaly']])
          # Rename columns 
          new_col_names = {'enum_id': 'Enumerator ID', 'Count': 'Number of surveys', 'Anomaly Type': 'Anomaly status'}
          df_sorted_renamed = df_sorted.rename(columns=new_col_names)
          st.dataframe(df_sorted_renamed, hide_index= True)
