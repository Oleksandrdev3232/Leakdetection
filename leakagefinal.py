import pandas as pd
import xgboost as xgb
import numpy as np
import os
import plotly.graph_objects as go

# Path to the directory
segments_folder = r"C:\Users\user\Desktop\GP assignment\evaporation_events_file"

# Load segment for the model
def loading_segments_model(segments_folder):
    X = []
    y = []
    # Go through desired segment files
    for file_name in os.listdir(segments_folder):
        if file_name.endswith(".csv"):
            segment_path = os.path.join(segments_folder, file_name)
            segment_df = pd.read_csv(segment_path)

            # Extracting data for feature engineering
            water_temperature = segment_df['water_temperature']
            air_temperature = segment_df['air_temperature']
            delta_water_temperature = water_temperature.diff().fillna(0.0)  # Find the delta change for water_temperature
            delta_air_temperature = air_temperature.diff().fillna(0.0)  # Find the delta change for air_temperature
            water_height_change = segment_df['filtered_water_height'].diff().fillna(0.0)  # Use change in water height

            # Find the change rate of water_height using the delta of water_temperature
            delta_water_temperature_nonzero = delta_water_temperature.replace(0.0, np.nan)
            # Avoid dividing by 0
            water_height_temp_rate = water_height_change / delta_water_temperature_nonzero
            # Find the change rate of water_height using the delta of air_temperature
            delta_air_temperature_nonzero = delta_air_temperature.replace(0.0, np.nan)
            # Avoid dividing by 0
            water_height_air_temp_rate = water_height_change / delta_air_temperature_nonzero
            # Convert timestamp to Unix to be able to operate with it
            timestamp = pd.to_datetime(segment_df['timestamp']).astype(np.int64) // 10**9  # Convert to seconds
            delta_timestamp = timestamp.diff().fillna(0.0)  # Find the delta of timestamp

            # Define features and append them
            features = np.column_stack(
                (water_temperature, delta_water_temperature, air_temperature, delta_air_temperature,
                 water_height_temp_rate, water_height_air_temp_rate, timestamp, delta_timestamp)
            )
            X.extend(features)
            y.extend(water_height_change)

    return np.array(X), np.array(y)

# Load segmented data and calculate features
X, y = loading_segments_model(segments_folder)

# state the hyper parameters
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=40,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=3,
    min_child_weight=1,
    n_estimators=500,
    subsample=1.0
)
xgb_model.fit(X, y)
# Full dataset
full_dataset_df = pd.read_csv(r"C:\Users\user\Desktop\GP assignment\leakagefileF.csv")

# Predict water height change
predicted_water_height_change_evaporation = xgb_model.predict(X)

full_dataset_df['valve_effect'] = -0.049583 * (full_dataset_df['valve_value'] < full_dataset_df['filtered_water_height'])
full_dataset_df['tap_effect'] = 0.045 * (full_dataset_df['tap_on'] == 1)
full_dataset_df['precipitation_effect'] = full_dataset_df['precipitation']

# extract the timeframe
def extract_timeframe_data(starting_timestamp, ending_timestamp):
    timeframe_df = full_dataset_df[(full_dataset_df['timestamp'] >= starting_timestamp) & (full_dataset_df['timestamp'] <= ending_timestamp)]
    return timeframe_df

def visualize_all_effects(timeframe_df, predicted_water_height_change_evaporation, full_dataset_df):
    fig = go.Figure()

    total_valve_effect = timeframe_df['valve_effect'].sum()
    total_tap_effect = timeframe_df['tap_effect'].sum()
    total_precipitation_effect = timeframe_df['precipitation_effect'].sum()

    # Find indices for the timeframe in the full dataset
    start_idx = full_dataset_df[full_dataset_df['timestamp'] == timeframe_df['timestamp'].iloc[0]].index[0]
    end_idx = full_dataset_df[full_dataset_df['timestamp'] == timeframe_df['timestamp'].iloc[-1]].index[0]

    predicted_height_change = predicted_water_height_change_evaporation[start_idx:end_idx+1]
    total_evaporation_effect = -abs(predicted_height_change.sum())
    # leakage detection function
    leakage_indices = []
    consecutive_decrease_count = 0
    start_index = None
    # set the max total evaporation effect
    evaporation_threshold = -10
    for idx in range(1, len(timeframe_df)):
        current_row = timeframe_df.iloc[idx]
        previous_row = timeframe_df.iloc[idx - 1]

        # make sure that the water level has consecutive decrease of 15 data points
        if current_row['filtered_water_height'] > previous_row['filtered_water_height']:
            consecutive_decrease_count += 1
            if start_index is None:
                start_index = idx
        else:
            consecutive_decrease_count = 0
            start_index = None
        if consecutive_decrease_count >= 15:
            #check the valve condition
            if current_row['valve_value'] > current_row['filtered_water_height']:
                total_evaporation_affect = predicted_height_change[start_index:idx+1].sum()
                # check for evaporation total effect
                if total_evaporation_affect > evaporation_threshold:
                    leakage_indices.extend(range(start_index, idx+1))
            consecutive_decrease_count = 0
            start_index = None

    fig.add_trace(go.Scatter(x=timeframe_df['timestamp'], y=timeframe_df['filtered_water_height'], mode='lines', name='Water height'))
    # Create annotations for total effects
    annotations = [
        dict(xref='paper', yref='paper', x=0.90, y=0.92, xanchor='left', yanchor='bottom',
             text=f"Total Valve Effect: {total_valve_effect:.2f} mm",
             font=dict(family='Arial', size=12, color='orange'), showarrow=False),
        dict(xref='paper', yref='paper', x=0.90, y=0.88, xanchor='left', yanchor='bottom',
             text=f"Total tap effect: {total_tap_effect:.2f} mm",
             font=dict(family='Arial', size=12, color='purple'), showarrow=False),
        dict(xref='paper', yref='paper', x=0.90, y=0.84, xanchor='left', yanchor='bottom',
             text=f"Total precipitation effect: {total_precipitation_effect:.2f} mm",
             font=dict(family='Arial', size=12, color='green'), showarrow=False),
        dict(xref='paper', yref='paper', x=0.90, y=0.80, xanchor='left', yanchor='bottom',
             text=f"Total evaporation effect: {total_evaporation_effect:.2f} mm",
             font=dict(family='Arial', size=12, color='red'), showarrow=False)]
    # Add annotations
    for annotation in annotations:
        fig.add_annotation(annotation)

    # Add markers for the desired processes
    fig.add_trace(go.Scatter(x=timeframe_df['timestamp'].loc[timeframe_df['valve_effect'] != 0],
                             y=timeframe_df['filtered_water_height'].loc[timeframe_df['valve_effect'] != 0],
                             mode='markers', name='Valve effect', marker=dict(color='orange')))

    fig.add_trace(go.Scatter(x=timeframe_df['timestamp'].loc[timeframe_df['tap_effect'] != 0],
                             y=timeframe_df['filtered_water_height'].loc[timeframe_df['tap_effect'] != 0],
                             mode='markers', name='Tap effect', marker=dict(color='purple')))

    fig.add_trace(go.Scatter(x=timeframe_df['timestamp'].loc[timeframe_df['precipitation_effect'] != 0],
                             y=timeframe_df['filtered_water_height'].loc[timeframe_df['precipitation_effect'] != 0],
                             mode='markers', name='Precipitation effect', marker=dict(color='green')))

    fig.add_trace(go.Scatter(x=timeframe_df['timestamp'],
                             y=timeframe_df['filtered_water_height'] + predicted_height_change,
                             mode='markers', name='Evaporation effect', marker=dict(color='red')))

    fig.add_trace(go.Scatter(x=timeframe_df.iloc[leakage_indices]['timestamp'],
                             y=timeframe_df.iloc[leakage_indices]['filtered_water_height'],
                             mode='markers', name='Leakage detected', marker=dict(color='black', size=8)))

    fig.update_layout(title='Combined effects on water height with leakage detection',
                      xaxis_title='Timestamp',
                      yaxis_title='Water Height (mm)',
                      legend=dict(x=0, y=1, traceorder="normal", font=dict(family="sans-serif", size=12, color="black")))
    fig.show()

# Choose a timeframe (start and end timestamps)
starting_timestamp = '2023-10-29 00:00:00'
ending_timestamp = '2023-10-30 00:00:00'
# Extract data for the desired timeframe
timeframe_df = extract_timeframe_data(starting_timestamp, ending_timestamp)

# Visualize the effects during the selected timeframe
visualize_all_effects(timeframe_df, predicted_water_height_change_evaporation, full_dataset_df)