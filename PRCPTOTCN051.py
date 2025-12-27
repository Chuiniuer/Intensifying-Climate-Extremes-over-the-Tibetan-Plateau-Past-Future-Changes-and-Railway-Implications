import os
import numpy as np
import rasterio
from tqdm import tqdm

# 📂 Directory settings
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"  # Precipitation data directory
output_dir = r"F:\phdl1\climate extremes\PRCPTOT"  # Result output directory
os.makedirs(output_dir, exist_ok=True)

# Compute PRCPTOT for years 1961-2014
all_years = range(1961, 2014 + 1)

print("Computing PRCPTOT (annual total precipitation on wet days)...")
for year in tqdm(all_years):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")
    output_file = os.path.join(output_dir, f"PRCPTOT_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw", "nodata": np.nan})  # Set output nodata to NaN

        data = src.read().astype(np.float32)  # Read all days (days, height, width)

        # Identify invalid value areas
        invalid_mask = np.isnan(data)

        # Wet days (precipitation ≥ 1 mm)
        wet_mask = data >= 1

        # Compute PRCPTOT (total precipitation on all wet days)
        prcptot = np.nansum(np.where(wet_mask, data, 0), axis=0).astype(np.float32)  # (height, width)

        # Maintain invalid areas as NaN
        prcptot[invalid_mask[0]] = np.nan

        # Save PRCPTOT result
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(prcptot, 1)
            dst.set_band_description(1, f"PRCPTOT_{year}")

print("✅ PRCPTOT computation (1961-2014) completed. Results saved to:", output_dir)
