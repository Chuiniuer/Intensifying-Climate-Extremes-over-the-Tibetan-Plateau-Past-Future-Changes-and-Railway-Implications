import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"
output_dir_rx1 = r"F:\phdl1\climate extremes\RX1day"
output_dir_rx5 = r"F:\phdl1\climate extremes\RX5day"

# Ensure output directories exist
os.makedirs(output_dir_rx1, exist_ok=True)
os.makedirs(output_dir_rx5, exist_ok=True)

# Target computation years
start_year, end_year = 1961, 2014

# Compute RX1day and RX5day for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing RX1day & RX5day"):
    input_file = os.path.join(data_dir, f"pre_{year}.tif")

    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Single-band output

        # Read all daily precipitation data
        pre_data = src.read().astype(np.float32)  # Shape (num_days, height, width)
        nan_mask = np.isnan(pre_data[0])  # Record invalid value areas

        # Compute RX1day (maximum daily precipitation)
        rx1day = np.nanmax(pre_data, axis=0)
        rx1day[nan_mask] = np.nan  # Handle invalid values

        # Compute RX5day (maximum 5-day sliding window cumulative precipitation)
        rx5day = np.full_like(rx1day, np.nan, dtype=np.float32)  # Initialise RX5day
        for d in range(num_days - 4):  # Only slide until the 5th-to-last day
            window_sum = np.nansum(pre_data[d:d+5], axis=0)  # Compute 5-day cumulative precipitation
            rx5day = np.nanmax(np.stack([rx5day, window_sum]), axis=0)  # Progressively take maximum
        rx5day[nan_mask] = np.nan  # Handle invalid values

    # Save RX1day
    output_file_rx1 = os.path.join(output_dir_rx1, f"RX1day_{year}.tif")
    with rasterio.open(output_file_rx1, "w", **meta) as dst:
        dst.write(rx1day, 1)
        dst.set_band_description(1, f"RX1day_{year}")

    # Save RX5day
    output_file_rx5 = os.path.join(output_dir_rx5, f"RX5day_{year}.tif")
    with rasterio.open(output_file_rx5, "w", **meta) as dst:
        dst.write(rx5day, 1)
        dst.set_band_description(1, f"RX5day_{year}")

print("RX1day & RX5day calculation completed. Results saved.")
