import os
import numpy as np
import rasterio
from tqdm import tqdm

# 📂 Directory settings
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"  # Precipitation data directory
output_dir = r"F:\phdl1\climate extremes\R95p"  # Result output directory
prwn95_file = r"F:\phdl1\climate extremes\PRwn95\PRwn95.tif"  # Precomputed PRwn95 file
os.makedirs(output_dir, exist_ok=True)

# Load PRwn95
with rasterio.open(prwn95_file) as src:
    prwn95 = src.read(1).astype(np.float32)  # (height, width)
    prwn95_meta = src.meta.copy()
    nodata_value = src.nodata  # Read original NoData value

# Compute R95p for years 1961-2014
all_years = range(1961, 2014 + 1)  # Only compute baseline period 1961-2014

print("Computing R95p (annual total precipitation above PRwn95)...")
for year in tqdm(all_years):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")
    output_file = os.path.join(output_dir, f"R95p_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw", "nodata": np.nan})  # Set output nodata to NaN

        data = src.read().astype(np.float32)  # 读取所有天 (days, height, width)

        # Identify invalid value areas
        invalid_mask = np.isnan(data)  # Record invalid value areas

        # Wet days (precipitation ≥ 1 mm)
        wet_mask = data >= 1

        # Calculate precipitation exceeding PRwn95
        extreme_precip = np.where((wet_mask) & (data > prwn95), data - prwn95, 0)

        # Set invalid value areas to NaN
        extreme_precip[invalid_mask] = np.nan

        # Compute R95p (total precipitation exceeding PRwn95)
        r95p = np.nansum(extreme_precip, axis=0).astype(np.float32)  # (height, width)

        # Set invalid value areas to NaN (final check)
        r95p[np.isnan(prwn95)] = np.nan

        # Save R95p result
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(r95p, 1)
            dst.set_band_description(1, f"R95p_{year}")

print("✅ R95p computation (1961-2014) completed. Results saved to:", output_dir)
