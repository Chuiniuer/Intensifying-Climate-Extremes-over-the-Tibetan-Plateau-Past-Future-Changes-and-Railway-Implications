import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"
output_file = r"F:\phdl1\climate extremes\TX90p\threshold\TXin90.tif"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Parse all GeoTIFF file paths
tif_files = [os.path.join(data_dir, f"tmax_{year}.tif") for year in range(start_year, end_year + 1)]

# Read the first file to obtain georeferencing information
with rasterio.open(tif_files[0]) as src:
    meta = src.meta.copy()  # Copy metadata
    height, width = src.height, src.width  # Raster dimensions
    transform = src.transform  # Affine transformation
    crs = src.crs  # Coordinate reference system

# 366-day array (for storing TXin90)
txin90 = np.full((366, height, width), np.nan, dtype=np.float32)

# Read all data and organise by calendar day
all_data = {d: [] for d in range(366)}

for year, file in tqdm(zip(range(start_year, end_year + 1), tif_files), desc="Reading Data", total=len(tif_files)):
    with rasterio.open(file) as src:
        num_days = src.count  # Get the number of days in that year (365 or 366)

        for band in range(1, num_days + 1):  # 1-based index in rasterio
            date_str = src.descriptions[band - 1]  # Read date, e.g., "1961-01-01"
            year, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read band data
            all_data[day_of_year].append(data)

# Calculate 90th percentile (using a 5-day moving window)
for day in tqdm(range(366), desc="Computing TXin90"):
    window_data = []

    # Take data for a 5-day moving window
    for offset in range(-2, 3):
        day_idx = (day + offset) % 366  # Ensure cyclic calculation
        window_data.extend(all_data[day_idx])

    if window_data:  # Avoid calculation on empty data
        window_stack = np.stack(window_data, axis=0)
        txin90[day] = np.percentile(window_stack, 90, axis=0)

# Update metadata to accommodate 366 days
meta.update({"count": 366, "dtype": "float32", "compress": "lzw"})

# Save TXin90 result as GeoTIFF
with rasterio.open(output_file, "w", **meta) as dst:
    for day in range(366):
        dst.write(txin90[day], day + 1)  # 1-based index
        dst.set_band_description(day + 1, f"Day-{day + 1}")

print("TXin90 calculation completed. Output saved to:", output_file)
