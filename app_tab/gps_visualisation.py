from app_utils import plot_tools as pt
import streamlit as st
import pandas as pd
import pydeck as pdk
import geopandas as gpd
from geopy.distance import geodesic

LAT_COLUMN = "latitude"
LON_COLUMN = "longitude"
ENUMERATOR_ID_VAR = "enum_id"
POINT_ID_VAR = "point_id"

def display():
    data = st.session_state.ETL_output['features_prediction_score']
    upload_checkbox = st.checkbox("Check the box if the assessment collected GPS-points", value=False)

    if upload_checkbox:
        uploaded_file1 = st.file_uploader("Upload the sampling frame", type=["xlsx"])
        uploaded_file2 = st.file_uploader("Upload the GPS recorded points", type=["xlsx"])

        if uploaded_file1 and uploaded_file2:
            df1 = pd.read_excel(uploaded_file1)
            df2 = pd.read_excel(uploaded_file2)

            # Merge dataframes on enum_id and index by point_id
            merged_df = pd.merge(df1, df2, on=[ENUMERATOR_ID_VAR, POINT_ID_VAR], suffixes=('_frame', '_recorded'))

            # Compute the distance between each pair of points
            def compute_distance(row):
                point1 = (row[LAT_COLUMN + '_frame'], row[LON_COLUMN + '_frame'])
                point2 = (row[LAT_COLUMN + '_recorded'], row[LON_COLUMN + '_recorded'])
                return geodesic(point1, point2).meters

            merged_df['distance_meters'] = merged_df.apply(compute_distance, axis=1)
            #st.dataframe(merged_df.iloc[:, :-2])

            unique_enums = data[ENUMERATOR_ID_VAR].unique()

            if 'enum_color_map' not in st.session_state:
                st.session_state.enum_color_map = {enum: color for enum, color in zip(unique_enums, pt.generate_palette_colors(len(unique_enums)))}
            
            #selected_enums = st.multiselect("Select Enumerator(s):", unique_enums, key="enum_gps_filter", 
            #                                placeholder="Enumerator ID(s)")

            #if selected_enums:
            #    df_enum = data[data[ENUMERATOR_ID_VAR].isin(selected_enums)]
            #else:
            #    df_enum = pd.DataFrame()  
            df_enum = pd.DataFrame()  
            st.dataframe(merged_df, hide_index=True)

            #colors = [st.session_state.enum_color_map.get(enum) for enum in unique_enums] if selected_enums else []

            # Display points on the map
            icon_data_1 = {
                "url": "https://img.icons8.com/ios-filled/50/ff0000/marker.png",  # URL for the pin image
                "width": 96,
                "height": 96,
                "anchorY": 96,
            }
            
            icon_data_2 = {
                "url": "https://img.icons8.com/ios-filled/50/0000ff/marker.png",  
                "width": 96,
                "height": 96,
                "anchorY": 96,
            }

            merged_df['icon_data_1'] = None
            merged_df['icon_data_2'] = None
            merged_df['icon_data_1'] = merged_df['icon_data_1'].apply(lambda x: icon_data_1)
            merged_df['icon_data_2'] = merged_df['icon_data_2'].apply(lambda x: icon_data_2)

            icon_layer_1 = pdk.Layer(
                "IconLayer",
                data=merged_df,
                get_icon='icon_data_1',
                get_size=4,
                size_scale=15,
                get_position=[LON_COLUMN + '_frame', LAT_COLUMN + '_frame'],
            )

            icon_layer_2 = pdk.Layer(
                "IconLayer",
                data=merged_df,
                get_icon='icon_data_2',
                get_size=4,
                size_scale=15,
                get_position=[LON_COLUMN + '_recorded', LAT_COLUMN + '_recorded'],
            )

            # Create a circle layer for a 500m radius around each pin
            circle_layer_1 = pdk.Layer(
                "ScatterplotLayer",
                data=merged_df,
                get_position=[LON_COLUMN + '_frame', LAT_COLUMN + '_frame'],
                get_radius=500,
                get_fill_color=[255, 0, 0, 50],
            )

            circle_layer_2 = pdk.Layer(
                "ScatterplotLayer",
                data=merged_df,
                get_position=[LON_COLUMN + '_recorded', LAT_COLUMN + '_recorded'],
                #get_radius=500,
                get_fill_color=[0, 0, 255, 50],
            )

            # Display the distance on the map
            line_layer = pdk.Layer(
                "LineLayer",
                data=merged_df,
                get_source_position=[LON_COLUMN + '_frame', LAT_COLUMN + '_frame'],
                get_target_position=[LON_COLUMN + '_recorded', LAT_COLUMN + '_recorded'],
                get_color=[0, 0, 0],
                get_width=3,
            )

            text_layer = pdk.Layer(
                "TextLayer",
                data=merged_df,
                get_position=[LON_COLUMN + '_frame', LAT_COLUMN + '_frame'],
                get_text='distance_meters',
                get_size=16,
                get_color=[0, 0, 0],
                get_angle=0,
            )

            st.pydeck_chart(pdk.Deck(
                layers=[icon_layer_1, icon_layer_2, circle_layer_1, circle_layer_2, line_layer, text_layer],
                initial_view_state=pdk.ViewState(
                    latitude=merged_df[LAT_COLUMN + '_frame'].mean(),
                    longitude=merged_df[LON_COLUMN + '_frame'].mean(),
                    zoom=10,
                    pitch=0
                ),
                map_style="mapbox://styles/mapbox/light-v9",
            ))

            if not df_enum.empty:
                #st.write(f"GPS data for selected enumerator IDs: {selected_enums}")
                df_enum_display = df_enum.copy()
                df_enum_display.rename(index=st.session_state.variable_mapping, columns=st.session_state.variable_mapping, inplace=True)
                df_enum_display = df_enum_display.set_index(df_enum_display.columns[0])
                st.dataframe(merged_df.iloc[:, :-2], hide_index=True)
            else:
                st.write("No GPS data available for the selected enumerators.")
