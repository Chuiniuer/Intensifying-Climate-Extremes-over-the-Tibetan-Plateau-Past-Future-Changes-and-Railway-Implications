import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"
txin90_file = r"F:\phdl1\climate extremes\TX90p\threshold\TXin90.tif"
output_dir = r"F:\phdl1\climate extremes\TX90p\yearly"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read TXin90 (90th percentile for baseline period)
with rasterio.open(txin90_file) as src:
    txin90 = src.read()  # Read all 366 days of 90th percentile values
    meta = src.meta.copy()  # Copy metadata
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

# Compute TX90p for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing TX90p"):
    input_file = os.path.join(data_dir, f"tmax_{year}.tif")

    # Read maximum temperature data for the year
    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        tx90p = np.zeros((src.height, src.width), dtype=np.float32)  # Initialise TX90p count array
        valid_pixel_count = np.zeros((src.height, src.width), dtype=np.float32)  # Count valid days
        nan_mask = np.zeros((src.height, src.width), dtype=bool)  # Record which pixels should be NaN

        for band in range(1, num_days + 1):  # 1-based index
            date_str = src.descriptions[band - 1]  # Read date
            _, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read data for the day
            invalid_mask = np.isnan(data)  # Invalid value mask
            nan_mask |= invalid_mask  # Record whether pixel is invalid

            # Calculate TX90p (above TXin90)
            tx90p += (data > txin90[day_of_year]).astype(np.float32)  # Count warm-day occurrences
            valid_pixel_count += (~invalid_mask).astype(np.float32)  # Count valid days

        # Calculate final percentage
        valid_mask = valid_pixel_count > 0  # Only compute for pixels with data
        tx90p[valid_mask] /= valid_pixel_count[valid_mask]  # Calculate TX90p percentage
        tx90p[~valid_mask] = np.nan  # Keep invalid areas as NaN

        # Debug output
        print(f"Year {year}: Valid pixels count min={valid_pixel_count.min()}, max={valid_pixel_count.max()}")

    # Output TX90p result
    output_file = os.path.join(output_dir, f"TX90p_{year}.tif")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(tx90p, 1)  # wite single band
        dst.set_band_description(1, f"TX90p_{year}")

print("TX90p calculation completed. Results saved to:", output_dir)
