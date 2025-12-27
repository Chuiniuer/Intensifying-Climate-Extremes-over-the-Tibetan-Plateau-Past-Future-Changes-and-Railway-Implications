import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmin"
tnin90_file = r"F:\phdl1\climate extremes\TN90p\threshold\TNin90.tif"
output_dir = r"F:\phdl1\climate extremes\TN90p\yearly"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read TNin90 (90th percentile for baseline period)
with rasterio.open(tnin90_file) as src:
    tnin90 = src.read()  # Read all 366 days of 90th percentile values
    meta = src.meta.copy()  # Copy metadata
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

# Compute TN90p for each yea
for year in tqdm(range(start_year, end_year + 1), desc="Computing TN90p"):
    input_file = os.path.join(data_dir, f"tmin_{year}.tif")

    # Read minimum temperature data for the year
    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        tn90p = np.zeros((src.height, src.width), dtype=np.float32)  # Initialise TN90p count array
        valid_pixel_count = np.zeros((src.height, src.width), dtype=np.float32)  # Count valid days
        nan_mask = np.zeros((src.height, src.width), dtype=bool)  # Record which pixels should be NaN

        for band in range(1, num_days + 1):  # 1-based index
            date_str = src.descriptions[band - 1]  # Read date
            _, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read data for the day
            invalid_mask = np.isnan(data)  # Invalid value mask
            nan_mask |= invalid_mask  # Record whether pixel is invalid

            # calculate TN90p
            tn90p += (data > tnin90[day_of_year]).astype(np.float32)  # Count occurrences above TNin90
            valid_pixel_count += (~invalid_mask).astype(np.float32)  # Count valid days

        # Calculate final percentage
        valid_mask = valid_pixel_count > 0  # Only compute for pixels with data
        tn90p[valid_mask] /= valid_pixel_count[valid_mask]  # Calculate TN90p percentage
        tn90p[~valid_mask] = np.nan  # Keep invalid areas as NaN

        # Debug output
        print(f"Year {year}: Valid pixels count min={valid_pixel_count.min()}, max={valid_pixel_count.max()}")

    # Output TN90p result
    output_file = os.path.join(output_dir, f"TN90p_{year}.tif")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(tn90p, 1)  # 写入单波段
        dst.set_band_description(1, f"TN90p_{year}")

print("TN90p calculation completed. Results saved to:", output_dir)
