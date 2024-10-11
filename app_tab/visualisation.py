import streamlit as st
from app_utils import plot_tools as pt

# Params
THRESHOLD_QUANTITATIVE_TYPE = 20
SURVEY_ID_VAR = 'audit_id'
ENUMERATOR_ID_VAR = 'enum_id'

def display():
    if "variable_mapping" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return

    features_list = list(st.session_state.variable_mapping.keys())
    features_labels = list(st.session_state.variable_mapping.values())

    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Enumerators", "Univariate", "Bivariate"])
    predic_data = st.session_state.ETL_output['features_prediction_score']
    variable_types = pt.classify_variable_types(predic_data, THRESHOLD_QUANTITATIVE_TYPE)
    hue_var_list = ["", 'anomaly_prediction']
    labels_hue = ["None", "Anomaly Prediction"]
    hue_mapping = dict(zip(hue_var_list, labels_hue))
    reverse_hue_mapping = {v: k for k, v in hue_mapping.items()}

    filtered_features_list = [col for col in features_list if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]
    filtered_labels_list = [label for col, label in zip(features_list, features_labels) if col not in ['audit_id', 'enum_id', 'anomaly_prediction']]


    with sub_tab1:
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
            
            unique_enums = st.session_state.ETL_output['features_prediction_score'][ENUMERATOR_ID_VAR].unique()
            selected_enums = st.multiselect("Select Enumerator(s):", unique_enums, key="enum_second_filter", 
                                            #default=unique_enums[0] if len(unique_enums) > 0 else st.error("No Enumerators"),
                                            placeholder="Enumerator ID(s)")

            if variable_enum:
                variable_name_enum = filtered_features_list[filtered_labels_list.index(variable_enum)]
                pt.univariate_plotting_interactive_enum_anomaly(df=predic_data, X=variable_name_enum, 
                                                                hue=hue_enum, variable_types=variable_types, 
                                                                x_label=variable_enum, 
                                                                selected_enums=selected_enums
                                                                )

                
    with sub_tab2:
        col1, col2 = st.columns([1, 5])
        with col1:
            variable = st.selectbox("Choose a variable:", filtered_labels_list, index=None)
            hue_univariate_label = st.selectbox("Choose a grouping variable:", labels_hue, key="HUE_univariate", index=0)
            hue_univariate = reverse_hue_mapping[hue_univariate_label]
            if hue_univariate == "":
                hue_univariate = None
        with col2:

            if len(selected_enums) > 0:  
                #available_survey_ids = st.session_state.ETL_output['features_prediction_score'].loc[
                #    st.session_state.ETL_output['features_prediction_score'][ENUMERATOR_ID_VAR].isin(selected_enums), 
                #    SURVEY_ID_VAR].unique()

                #Display only anomaly surveys
                available_survey_ids = predic_data[(predic_data.anomaly_prediction == 1) & 
                                            (predic_data[ENUMERATOR_ID_VAR].isin(selected_enums))][SURVEY_ID_VAR].unique()
            else:
                available_survey_ids = []

            if len(available_survey_ids) > 0:
                survey_id_input = st.selectbox("Select Survey ID to highlight:", available_survey_ids, 
                                            key="survey_id", index=None,
                                            placeholder="Survey ID")
            else:
                st.warning("No surveys available. Please select an enumerator with flagged surveys in the previous tab. \
                            It allows to display the selected feature values and the available surveys \
                            for a given enumerator.")
                survey_id_input = None

            if variable:
                variable_name = filtered_features_list[filtered_labels_list.index(variable)]
                pt.univariate_plotting_interactive(df=predic_data, X=variable_name, 
                                                   hue=hue_univariate, variable_types=variable_types, 
                                                   x_label=variable, 
                                                   survey_id=survey_id_input)

    with sub_tab3:
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
                avg_catplot_multicat = pt.avg_catplot_multicat(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, variable_types=variable_types, x_label=X, y_label=Y)
                density_heatmap = pt.density_heatmap(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, x_label=X, y_label=Y)

                # Seaborn version
                # kernel_plot_seaborn = pt.kernel_density_plot_seaborn(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, x_label=X, y_label=Y)
                # catplot_seaborn = pt.catplot_multicat_seaborn(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, variable_types=variable_types, x_label=X, y_label=Y)
                # distplot_seaborn = pt.displot_seaborn(df=predic_data, X=X_name, Y=Y_name, hue=hue_bivariate, x_label=X, y_label=Y)


                kernel_tab, catplot_tab, displot_tab = st.tabs(["Kernel Density Plot", "Average Plot by Categories", "Density Heatmap"])
                # if X_name != X_name:
                with kernel_tab:
                        st.plotly_chart(kernel_plot)
                        # st.pyplot(kernel_plot_seaborn)
                with catplot_tab:
                    st.plotly_chart(avg_catplot_multicat)
                    # st.pyplot(catplot_seaborn)
                with displot_tab:
                   st.plotly_chart(density_heatmap)

