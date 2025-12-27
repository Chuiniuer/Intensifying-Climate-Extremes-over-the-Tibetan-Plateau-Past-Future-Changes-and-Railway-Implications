import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input precipitation data directory
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"
# Output PRwn95 directory
output_dir = r"F:\phdl1\climate extremes\PRwn95"
os.makedirs(output_dir, exist_ok=True)

# Baseline period 1961-2014
start_year, end_year = 1961, 2014

# Initialise wet day precipitation data list
wet_days_data = []

# Read all baseline period data
for year in tqdm(range(start_year, end_year + 1), desc="Collecting wet day precipitation"):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        precip_data = src.read()  # Shape: (days, height, width))
        wet_mask = precip_data >= 1  # Select wet days (precipitation ≥ 1 mm)
        wet_precip = np.where(wet_mask, precip_data, np.nan)  # Keep only wet day precipitation

        wet_days_data.append(wet_precip)

# Concatenate all wet day data
wet_days_data = np.concatenate(wet_days_data, axis=0)  # Shape: (days_total, height, width)

# Calculate 95th percentile of wet day precipitation
prwn95 = np.nanpercentile(wet_days_data, 95, axis=0)

# Read any year's data as template
template_year = 1961
template_file = os.path.join(pre_dir, f"pre_{template_year}.tif")
with rasterio.open(template_file) as src:
    meta = src.meta.copy()
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Single-band output

# Save PRwn95 result
output_file = os.path.join(output_dir, "PRwn95_1961-2014.tif")
with rasterio.open(output_file, "w", **meta) as dst:
    dst.write(prwn95, 1)
    dst.set_band_description(1, "PRwn95 (1961-2014)")

print("PRwn95 calculation completed. Result saved to:", output_file)
