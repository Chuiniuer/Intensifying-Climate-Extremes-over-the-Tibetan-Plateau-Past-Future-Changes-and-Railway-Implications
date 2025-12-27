import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths (tmax)
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"
txin10_file = r"F:\phdl1\climate extremes\TX10p\threshold\TXin10.tif"
output_dir = r"F:\phdl1\climate extremes\TX10p\yearly"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read TXin10 (10th percentile for baseline period)
with rasterio.open(txin10_file) as src:
    txin10 = src.read()  # Read all 366 days of 10th percentile values
    meta = src.meta.copy()  # Copy metadata
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

# Compute TX10p for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing TX10p"):
    input_file = os.path.join(data_dir, f"tmax_{year}.tif")

    # Read maximum temperature data for the year
    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        tx10p = np.zeros((src.height, src.width), dtype=np.float32)  # Initialise TX10p count array
        valid_pixel_count = np.zeros((src.height, src.width), dtype=np.float32)  # Count valid days
        nan_mask = np.zeros((src.height, src.width), dtype=bool)  # Record which pixels should be NaN

        for band in range(1, num_days + 1):  # 1-based index
            date_str = src.descriptions[band - 1]  # Read date
            _, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read data for the day
            invalid_mask = np.isnan(data)  # Invalid value mask
            nan_mask |= invalid_mask  # Record whether pixel is invalid

            # Calculate TX10p
            tx10p += (data < txin10[day_of_year]).astype(np.float32)  # Count occurrences below TXin10
            valid_pixel_count += (~invalid_mask).astype(np.float32)  # Count valid days

        # Calculate final percentage
        valid_mask = valid_pixel_count > 0  # Only compute for pixels with data
        tx10p[valid_mask] /= valid_pixel_count[valid_mask]  # Calculate TX10p percentage
        tx10p[~valid_mask] = np.nan  # Keep invalid areas as NaN

        # Debug output
        print(f"Year {year}: Valid pixels count min={valid_pixel_count.min()}, max={valid_pixel_count.max()}")

    # Output TX10p result
    output_file = os.path.join(output_dir, f"TX10p_{year}.tif")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(tx10p, 1)  # Write single band
        dst.set_band_description(1, f"TX10p_{year}")

print("TX10p calculation completed. Results saved to:", output_dir)
