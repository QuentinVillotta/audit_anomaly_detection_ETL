def univariate_plotting_semigood(df, X, hue, variable_types, x_label=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    if hue:
        unique_categories = df[hue].unique()
        colors = ["#FFA500", "#636EFA"] 
    else: 
        unique_categories = [None]  
        colors = ['#636EFA']  

    for i, category in enumerate(unique_categories):
        if hue:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        box_trace = go.Box(
            x=subset[X], 
            name=str(category) if hue else "All Data", 
            boxmean='sd', 
            orientation='h',
            marker=dict(color=color),
            legendgroup=str(category) if hue else "All Data",
            showlegend=True
        )
        fig.add_trace(box_trace, row=1, col=1)

    for i, category in enumerate(unique_categories):
        if hue:
            subset = df[df[hue] == category]
            color = colors[i % len(colors)]
        else:
            subset = df
            color = colors[0]
        
        hist_trace = go.Histogram(
            x=subset[X], 
            name=str(category) if hue else "", 
            opacity=0.7,
            histnorm='probability density' if variable_types[X] != 'discrete' else None,
            marker=dict(color=color),
            showlegend=False  
        )
        fig.add_trace(hist_trace, row=2, col=1)

    fig.update_layout(
        height=800,  
        barmode='overlay' if hue else 'group',  
        title=f'Univariate Plot for {X}' + (f' grouped by {hue}' if hue else ''), 
        legend=dict(title=hue if hue else "All Data"), 
    )

    if x_label:
        fig.update_xaxes(title_text=x_label, row=2, col=1)
    else:
        fig.update_xaxes(title_text=X, row=2, col=1)

    fig.update_yaxes(title_text="Density", row=1, col=1)  
    fig.update_yaxes(title_text="Count", row=2, col=1)   
    if variable_types[X] == 'discrete':
        fig.update_xaxes(tickvals=df[X].unique(), ticktext=df[X].unique(), ticks='outside', tickwidth=3, row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)


def univariate_plotting3(df, X, hue, variable_types, x_label=None):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    if variable_types[X] == 'discrete' or variable_types[X] == 'non-numeric':
        for category in df[hue].unique():
            box_trace = go.Box(x=df[df[hue] == category][X], name=str(category), boxmean='sd', orientation='h')
            fig.add_trace(box_trace, row=1, col=1)
    else:
        for category in df[hue].unique():
            box_trace = go.Box(x=df[df[hue] == category][X], name=str(category), boxmean='sd', orientation='h')
            fig.add_trace(box_trace, row=1, col=1)

    if variable_types[X] == 'discrete':
        for category in df[hue].unique():
            hist_trace = go.Histogram(x=df[df[hue] == category][X], name=str(category), opacity=0.7, barmode='overlay')
            fig.add_trace(hist_trace, row=2, col=1)
    elif variable_types[X] == 'non-numeric':
        for category in df[hue].unique():
            hist_trace = go.Histogram(x=df[df[hue] == category][X], name=str(category), opacity=0.7, barmode='overlay')
            fig.add_trace(hist_trace, row=2, col=1)
    else:
        # Continuous variable histogram (normalized to density) with grouping (hue)
        for category in df[hue].unique():
            hist_trace = go.Histogram(x=df[df[hue] == category][X], name=str(category), opacity=0.7, histnorm='probability density')
            fig.add_trace(hist_trace, row=2, col=1)

    # Update layout and axes labels
    fig.update_layout(
        height=600,  # Set the overall figure height
        barmode='overlay',  # Overlay histograms for grouped categories
        title=f'Univariate Plot for {X} grouped by {hue}',  # Title of the plot
    )

    # Set x-axis label for the bottom row (shared x-axis)
    if x_label:
        fig.update_xaxes(title_text=x_label, row=2, col=1)
    else:
        fig.update_xaxes(title_text=X, row=2, col=1)

    # Set y-axis labels
    fig.update_yaxes(title_text="Density", row=1, col=1)  # For the boxplot (Density or distribution)
    fig.update_yaxes(title_text="Count", row=2, col=1)    # For the histogram (Count or frequency)

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)



def univariate_plotting_notworking(df, X, hue, variable_types, x_label=None):
    # Create subplots with two rows, shared x-axis, and different row heights
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.2, 0.8])

    # Box plot on the top row
    if variable_types[X] == 'discrete': 
        # Discrete/Non-numeric box plot
        box_trace = go.Box(x=df[X], marker=dict(opacity=0.5), boxmean='sd', name="Boxplot", orientation='h')
        hist_trace = go.Histogram(x=df[X], name="Histogram", marker=dict(opacity=0.7), barmode='overlay')
    
    elif variable_types[X] == 'non-numeric':
        box_trace = go.Box(x=df[X], marker=dict(opacity=0.5), boxmean='sd', name="Boxplot", orientation='h')
        hist_trace = go.Histogram(x=df[X], name="Countplot", marker=dict(opacity=0.7), barmode='overlay')
    
    else:
        # Continuous variable box plot
        box_trace = go.Box(x=df[X], marker=dict(opacity=0.5), boxmean='sd', name="Boxplot", orientation='h')
        hist_trace = go.Histogram(x=df[X], name="Density", marker=dict(opacity=0.7), histnorm='probability density')


    fig.add_trace(box_trace, row=1, col=1)
    fig.add_trace(hist_trace, row=2, col=1)

    # Update layout and axes labels
    fig.update_layout(
        height=800,  # Set the overall figure height
        showlegend=True,  # Disable legend
        title=f'Univariate Plot for {X}',  # Title of the plot
    )

    # Set x-axis label for the bottom row (shared x-axis)
    if x_label:
        fig.update_xaxes(title_text=x_label, row=2, col=1)
    else:
        fig.update_xaxes(title_text=X, row=2, col=1)

    # Set y-axis labels
    fig.update_yaxes(title_text="Density", row=1, col=1)  # For the boxplot (Density or distribution)
    fig.update_yaxes(title_text="Count", row=2, col=1)    # For the histogram (Count or frequency)

    # Display the interactive plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def univariate_plotting2(df, X, hue, variable_types, x_label=None) -> None:
    
    # Create the figure based on the type of variable
    if variable_types[X] == 'discrete':
        fig_box = px.box(df, x=X, color=hue, points="all")
        fig_count = px.histogram(df, x=X, color=hue, barmode='relative')
        
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Box Plot', 'Count Plot'))
        for trace in fig_box.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig_count.data:
            fig.add_trace(trace, row=2, col=1)

    elif variable_types[X] == 'non-numeric':
        fig_count = px.histogram(df, x=X, color=hue, barmode='overlay')
        fig = fig_count

    else:  # Assuming this is for numeric types
        fig_box = px.box(df, x=X, color=hue)
        fig_hist = px.histogram(df, x=X, color=hue, histnorm='probability density', 
                                marginal='rug', barmode='overlay')

        fig = make_subplots(rows=2, cols=1, subplot_titles=('Box Plot', 'Histogram'))
        for trace in fig_box.data:
            fig.add_trace(trace, row=1, col=1)
        for trace in fig_hist.data:
            fig.add_trace(trace, row=2, col=1)

    # Update plot size
    fig.update_layout(height=1200, width=1200)

    # Update color scheme based on hue
    hue_categories = df[hue].unique()
    colors = px.colors.qualitative.Plotly[:len(hue_categories)]  # Get distinct colors

    for trace in fig.data:
        if trace.name in hue_categories:
            color_index = list(hue_categories).index(trace.name) % len(colors)
            trace.marker.color = colors[color_index]

    if x_label:
        fig.update_layout(xaxis_title=x_label)

    fig.update_yaxes(title_text='Count', row=2, col=1)
    fig.update_yaxes(title_text='Density', row=1, col=1)

    # Display the plot
    st.plotly_chart(fig)


def display2():
    sub_tab1, sub_tab2 = st.tabs(["Local Interpretation", "Global Interpretation"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
        
    if st.session_state.ETL_output:
        # Extract Interpretation data
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        shap_values = interpretation_data['shap_values']
        #st.write(shap_values)
        features = interpretation_data['features']
        shap_data = features.drop(SURVEY_ID_VAR, axis=1)

        with sub_tab1:
            st.write(predic_data.columns)
            st.write()
            survey_id = predic_data[predic_data.anomaly_prediction == 1][SURVEY_ID_VAR]
            selected_survey = st.selectbox("Select a survey ID", survey_id)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Anomaly prediction")
                st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_prediction'])
            with col2:
                st.subheader("Anomaly Score")
                st.write(predic_data.loc[predic_data[SURVEY_ID_VAR] == selected_survey, 'anomaly_score'])

            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
            with sub_sub2_tab1:
                pt.id_survey_shap_bar_plot(SURVEY_ID_VAR, 
                                           selected_survey, 
                                           features, 
                                           shap_values)
            with sub_sub2_tab2:
                pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, 
                                             selected_survey=selected_survey, 
                                             data=features,
                                             shap_values=shap_values)
        with sub_tab2:
            nb_features = len(shap_data.columns)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
            st.pyplot(fig)


def display3():
    sub_tab1, sub_tab2 = st.tabs(["Local Interpretation", "Global Interpretation"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
        
    if st.session_state.ETL_output:
        # Extract Interpretation data
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        shap_values = interpretation_data['shap_values']
        
        # Drop the first column from features
        features = interpretation_data['features'].iloc[:, 1:]  # Drop the first column

        # Create a mapping of variable names to labels
        variable_mapping = st.session_state.variable_names  # Assuming this is a dict

        with sub_tab1:
            st.write(predic_data.columns)
            survey_id = predic_data[predic_data.anomaly_prediction == 1][SURVEY_ID_VAR]
            selected_survey = st.selectbox("Select a survey ID", survey_id)

            # Get the index of the selected survey
            selected_index = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey].index[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Anomaly prediction")
                st.write(predic_data.loc[selected_index, 'anomaly_prediction'])
            with col2:
                st.subheader("Anomaly Score")
                st.write(predic_data.loc[selected_index, 'anomaly_score'])

            # Extract SHAP values for the selected survey
            survey_shap_values = shap_values[selected_index].values  # Accessing the numerical values directly

            # Create a Series for SHAP values
            shap_series = pd.Series(survey_shap_values, index=features.columns)

            # Filter for features with negative SHAP values and get the top 5
            negative_shap_df = shap_series[shap_series < 0].nsmallest(5)

            st.subheader("Top 5 Features with Lowest Negative SHAP Values")
            for feature, value in negative_shap_df.items():
                # Map the variable name to its label
                feature_label = variable_mapping.get(feature, feature)  # Default to feature name if not found
                st.write(f"{feature_label}: {value}")

                # Assuming 'variable_meaning' is a dictionary where keys are feature names
                feature_comment = st.session_state.variable_meaning.get(feature, "No comment available.")
                st.text_area(f"Comment for {feature_label}", value=feature_comment, height=100)

            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
            with sub_sub2_tab1:
                pt.id_survey_shap_bar_plot(SURVEY_ID_VAR, 
                                           selected_survey, 
                                           features, 
                                           shap_values)
            with sub_sub2_tab2:
                pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, 
                                             selected_survey=selected_survey, 
                                             data=features,
                                             shap_values=shap_values)

        with sub_tab2:
            nb_features = len(features.columns)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
            st.pyplot(fig)



def display():
    sub_tab1, sub_tab2 = st.tabs(["Local Interpretation", "Global Interpretation"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
        
    if st.session_state.ETL_output:
        # Extract Interpretation data
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        
        shap_values = interpretation_data['shap_values']
        features = interpretation_data['features']
        
        # Get the survey ID
        survey_id = predic_data[SURVEY_ID_VAR]
        selected_survey = st.selectbox("Select a survey ID", survey_id)

        # Find the anomaly prediction and score
        selected_row = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey]

        if selected_row.empty:
            st.error("No data found for the selected survey ID.")
            return

        anomaly_prediction = selected_row['anomaly_prediction'].values[0]
        anomaly_score = selected_row['anomaly_score'].values[0]

        with sub_tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Anomaly prediction")
                st.write(anomaly_prediction)
            with col2:
                st.subheader("Anomaly Score")
                st.write(anomaly_score)

            # Extract the SHAP values for the selected survey
            survey_index = predic_data.index[predic_data[SURVEY_ID_VAR] == selected_survey][0]
            survey_shap_values = shap_values[survey_index]

            # Check if the shap_values is an Explanation object
            if isinstance(survey_shap_values, shap.Explanation):
                shap_values_array = survey_shap_values.values
            else:
                shap_values_array = survey_shap_values

            # Debug: Check the shapes of the SHAP values and features
            num_features = len(features.columns)
            num_shap_values = shap_values_array.shape[0]

            if num_features != num_shap_values:
                st.error(f"Number of features ({num_features}) does not match number of SHAP values ({num_shap_values}).")
                return

            # Create a DataFrame with feature names and their corresponding SHAP values
            shap_values_df = pd.DataFrame(shap_values_array.reshape(1, -1), columns=features.columns)

            # Get the lowest 5 feature importance values
            lowest_features = shap_values_df.T.sort_values(by=0).head(5)
            lowest_features.columns = ['SHAP Value']

            # Display information about the lowest features
            for feature, value in lowest_features.iterrows():
                variable_name = feature
                explanation = st.session_state.variable_meaning.get(variable_name, "No explanation available.")
                st.text(f"**{variable_name}**: {explanation} | Value: {value[0]:.4f}")

            # Create sub tabs for feature importance and force plot
            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
            with sub_sub2_tab1:
                pt.id_survey_shap_bar_plot(SURVEY_ID_VAR, selected_survey, features, shap_values)
            with sub_sub2_tab2:
                pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, selected_survey=selected_survey, data=features, shap_values=shap_values)

        with sub_tab2:
            nb_features = len(features.columns)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
            st.pyplot(fig)


import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt

def display():
    sub_tab1, sub_tab2 = st.tabs(["Local Interpretation", "Global Interpretation"])

    if "variable_meaning" not in st.session_state:
        st.error("Failed to load variable mapping.")
        return
        
    if st.session_state.ETL_output:
        # Extract Interpretation data
        interpretation_data = st.session_state.ETL_output['SHAP_interpretation']
        predic_data = st.session_state.ETL_output['features_prediction_score']
        shap_values = interpretation_data['shap_values']
        
        # Drop the first column from features
        features = interpretation_data['features'].iloc[:, 1:]  # Drop the first column

        # Create a mapping of variable names to labels
        variable_mapping = st.session_state.variable_names  # Assuming this is a dict

        with sub_tab1:
            st.write(predic_data.columns)
            survey_id = predic_data[predic_data.anomaly_prediction == 1][SURVEY_ID_VAR]
            selected_survey = st.selectbox("Select a survey ID", survey_id)

            # Get the index of the selected survey
            selected_index = predic_data[predic_data[SURVEY_ID_VAR] == selected_survey].index[0]

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Anomaly prediction")
                st.write(predic_data.loc[selected_index, 'anomaly_prediction'])
            with col2:
                st.subheader("Anomaly Score")
                st.write(predic_data.loc[selected_index, 'anomaly_score'])

            # Extract SHAP values for the selected survey
            survey_shap_values = shap_values[selected_index].values  # Accessing the numerical values directly

            # Create a Series for SHAP values
            shap_series = pd.Series(survey_shap_values, index=features.columns)

            # Filter for features with negative SHAP values and get the top 5
            negative_shap_df = shap_series[shap_series < 0].nsmallest(5)

            st.subheader("Top 5 Features with Lowest Negative SHAP Values")
            for feature, value in negative_shap_df.items():
                # Map the variable name to its label
                feature_label = variable_mapping.get(feature, feature)  # Default to feature name if not found
                st.write(f"{feature_label}: {value}")

                # Assuming 'variable_meaning' is a dictionary where keys are feature names
                feature_comment = st.session_state.variable_meaning.get(feature, "No comment available.")
                st.text_area(f"Comment for {feature_label}", value=feature_comment, height=100)

            sub_sub2_tab1, sub_sub2_tab2 = st.tabs(["Feature Importance", "Force Plot"])
            with sub_sub2_tab1:
                pt.id_survey_shap_bar_plot(SURVEY_ID_VAR, 
                                           selected_survey, 
                                           features, 
                                           shap_values)
            with sub_sub2_tab2:
                pt.id_survey_shap_force_plot(survey_id_var=SURVEY_ID_VAR, 
                                             selected_survey=selected_survey, 
                                             data=features,
                                             shap_values=shap_values)

        with sub_tab2:
            nb_features = len(features.columns)
            fig, ax = plt.subplots()
            shap.plots.bar(shap_values, max_display=nb_features, show=False, ax=ax)
            st.pyplot(fig)
