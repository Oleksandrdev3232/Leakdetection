import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import medfilt
def apply_filter(data, window_size):
    data['filtered_water_height'] = medfilt(data['water_height'], kernel_size=window_size)
    return data
csv_file_path = r"C:\Users\user\Desktop\GP assignment\rain_info.csv"
data = pd.read_csv(csv_file_path, parse_dates=['timestamp'])
window_size = 3

filtered_data = apply_filter(data.copy(), window_size)
timestamp_interval = 2000
timestamps = filtered_data['timestamp'].iloc[::timestamp_interval]

plt.figure(figsize=(10, 6))
plt.plot(data['timestamp'], data['water_height'], label='Original Water Height', alpha=0.7)
plt.plot(filtered_data['timestamp'], filtered_data['filtered_water_height'], color='red')
plt.xticks(timestamps, rotation=45)

plt.title('Original/Median filtered water height')
plt.xlabel('Timestamp')
plt.ylabel('Water height')
plt.legend()
plt.tight_layout()
filtered_data.to_csv(r"C:\Users\user\Desktop\GP assignment\taptesting.csv", index=False)

plt.show()
