import streamlit as st
from app_utils import plot_tools as pt


# To order by nb d'annomaly

def display():
     df = st.session_state.ETL_output['features_prediction_score']
     sub_tab1, sub_tab2 = st.tabs(["General", "Enumerator"])
 
     with sub_tab1:
          col1, col2 = st.columns([2, 4])
          with col1:
               pt.pie_chart_pct_anomalies(df)
          with col2:
               # summary_stats = df.describe()
               st.dataframe(df.set_index('audit_id'), hide_index= False)

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
