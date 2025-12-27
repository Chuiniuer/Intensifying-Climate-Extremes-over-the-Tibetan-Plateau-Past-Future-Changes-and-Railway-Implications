import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input precipitation data directory
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"
# Output SDII directory
output_dir = r"F:\phdl1\climate extremes\SDII"
os.makedirs(output_dir, exist_ok=True)

# Processing year range
start_year, end_year = 1961, 2014  # Adjust as appropriate

# Process each year's data
for year in tqdm(range(start_year, end_year + 1), desc="Computing SDII"):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")
    output_file = os.path.join(output_dir, f"SDII_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

        total_precip = np.zeros((src.height, src.width), dtype=np.float32)
        wet_days = np.zeros((src.height, src.width), dtype=np.int32)

        # Process day by day
        for band in range(1, src.count + 1):
            data = src.read(band)
            mask = data >= 1  # Calculate wet days (precipitation ≥ 1 mm)
            total_precip += np.where(mask, data, 0)  # Accumulate precipitation only on wet days
            wet_days += mask  # Count wet days

        # Calculate SDII, avoiding division by zero
        sdii = np.where(wet_days > 0, total_precip / wet_days, np.nan)

        # Save SDII result
        with rasterio.open(output_file, "w", **meta) as dst:
            dst.write(sdii, 1)
            dst.set_band_description(1, f"SDII_{year}")

print("SDII calculation completed. Results saved to:", output_dir)
