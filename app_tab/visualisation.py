import streamlit as st
from app_utils import plot_tools as pt

# Params
THRESHOLD_QUANTITATIVE_TYPE = 20

def display():
    
    if "variable_mapping" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
    #st.write(st.session_state.variable_mapping)
    features_list = list(st.session_state.variable_mapping.keys())
    features_labels = list(st.session_state.variable_mapping.values())

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Univariate", "Bivariate", "Enumerators"])
    predic_data = st.session_state.ETL_output['features_prediction_score']
    variable_types = pt.classify_variable_types(predic_data, THRESHOLD_QUANTITATIVE_TYPE)
    hue_var_list = ["", 'anomaly_prediction']
    labels_hue = ["None", "Anomaly Prediction"]
    hue_mapping = dict(zip(hue_var_list, labels_hue))
    reverse_hue_mapping = {v: k for k, v in hue_mapping.items()}

    # Exclude certain columns from features_list
    filtered_features_list = [col for col in features_list if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]
    filtered_labels_list = [label for col, label in zip(features_list, features_labels) if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]
    
    with sub_tab1:
        col1, col2 = st.columns([1, 5])
        with col1:
            variable = st.selectbox("Choose a variable:", filtered_labels_list, index=None)
            hue_univariate_label = st.selectbox("Choose a grouping variable:", labels_hue, key="HUE_univariate", index=0)
            hue_univariate = reverse_hue_mapping[hue_univariate_label] 
            if hue_univariate == "":
                hue_univariate = None
        with col2:
            if variable:
                variable_name = filtered_features_list[filtered_labels_list.index(variable)]
                pt.univariate_plotting_interactive(df=predic_data, X=variable_name, hue=hue_univariate, variable_types=variable_types, x_label=variable)
                fig = pt.univariate_plotting(df=predic_data, X=variable_name, hue=hue_univariate, variable_types=variable_types, x_label=variable)
                st.pyplot(fig)

    with sub_tab2:
        col1, col2 = st.columns([1, 5])
        with col1:
            X = st.selectbox("Choose X-axis variable", filtered_labels_list, key="X", index=None)
            Y = st.selectbox("Choose Y-axis variable", filtered_labels_list, key="Y", index=None)
            hue_bivariate_label = st.selectbox("Choose a grouping variable:", labels_hue, key="HUE_bivariate", index=0)
            hue_bivariate = reverse_hue_mapping[hue_bivariate_label]
            if hue_bivariate == "":
                hue_bivariate = None
        with col2:
            if X and Y:
                X_name = filtered_features_list[filtered_labels_list.index(X)]
                Y_name = filtered_features_list[filtered_labels_list.index(Y)]
                
                kernel_plot = pt.kernel_density_plot(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, x_label=X, y_label=Y)
                catplot = pt.catplot_multicat(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, variable_types=variable_types, x_label=X, y_label=Y)
                distplot = pt.displot(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, x_label=X, y_label=Y)

                kernel_tab, catplot_tab, displot_tab = st.tabs(["Kernel Density Plot", "Catplot", "Distplot"])
                with kernel_tab:
                    st.pyplot(kernel_plot)
                with catplot_tab:
                    st.pyplot(catplot)
                with displot_tab:
                    st.pyplot(distplot)
    with sub_tab3:
        col1, col2 = st.columns([1, 5])
        with col1:
            variable_enum = st.selectbox("Choose a variable:", filtered_labels_list, index=None, 
                                         key="enum_variable")

            hue_enum_label = st.selectbox("Choose a grouping variable:", labels_hue, 
                                          key="HUE_enum", index=0)
            hue_enum = reverse_hue_mapping[hue_enum_label] 
            if hue_enum == "":
                hue_enum = None

        with col2:
            if variable_enum:
                variable_name_enum = filtered_features_list[filtered_labels_list.index(variable_enum)]
                pt.univariate_plotting_interactive_enum_anomaly(df=predic_data, X=variable_name_enum, 
                                                                hue=hue_enum, variable_types=variable_types, 
                                                                x_label=variable_enum)



