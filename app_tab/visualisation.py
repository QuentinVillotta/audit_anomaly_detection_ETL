import streamlit as st
from app_utils import plot_tools as pt


# Params
THRESHOLD_QUANTITATIVE_TYPE = 20


def display():
     sub_tab1, sub_tab2 = st.tabs(["Univariate", "Bivariate"])
     predic_data = st.session_state.ETL_output['features_prediction_score']
     variable_types = pt.classify_variable_types(predic_data, THRESHOLD_QUANTITATIVE_TYPE)
     hue_var_list = ['anomaly_prediction']

     # Exclude the 'enum_id', 'anomaly_score' and anomaly_prediction columns
     features_list = [col for col in predic_data.columns if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]
     with sub_tab1:
          col1, col2 = st.columns([1, 5])
          with col1:
               variable = st.selectbox("Choose a variable:", features_list, index=None)
               hue_univariate = st.selectbox("Choose a grouping variable:", hue_var_list, key="HUE_univariate", index=0)
          with col2:
               if variable:
                    fig = pt.univariate_ploting(df=predic_data, X=variable, hue=hue_univariate, variable_types=variable_types)
                    st.pyplot(fig)
     with sub_tab2:
                col1, col2 = st.columns([1, 5])
                with col1:
                    X = st.selectbox("Choose X-axis variable", features_list, key="X", index = None)
                    Y = st.selectbox("Choose Y-axis variable", features_list, key="Y", index = None)
                    hue_bivariate = st.selectbox("a grouping variable:", hue_var_list, key="HUE_bivariate", index=0)
                with col2:
                    if X and Y:
                         if hue_bivariate:
                              kernel_plot = pt.kernel_density_plot(df=predic_data, X=X, Y=Y, hue=hue_bivariate)
                              catplot = pt.catplot_multicat(df=predic_data, X=X, Y=Y, hue=hue_bivariate, variable_types=variable_types)
                              distplot = pt.displot(df=predic_data, X=X, Y=Y, hue=hue_bivariate)
                         else:
                              kernel_plot = pt.kernel_density_plot(df=predic_data, X=X, Y=Y, hue=None)
                              catplot = pt.catplot_multicat(df=predic_data, X=X, Y=Y, hue=None, variable_types=variable_types)
                              distplot = pt.displot(df=predic_data, X=X, Y=Y, hue=None)
                         kernel_tab, catplot_tab, displot_tab = st.tabs(["Kernel Density Plot", "Catplot", "Distplot"])
                         with kernel_tab: 
                              st.pyplot(kernel_plot)
                         with catplot_tab: 
                              st.pyplot(catplot)
                         with displot_tab: 
                              st.pyplot(distplot)

