import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input precipitation data directory
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"
# Output directories
output_dir_r1mm = r"F:\phdl1\climate extremes\R1mm"
output_dir_r10mm = r"F:\phdl1\climate extremes\R10mm"
os.makedirs(output_dir_r1mm, exist_ok=True)
os.makedirs(output_dir_r10mm, exist_ok=True)

# Processing year range (only 1961-2014)
start_year, end_year = 1961, 2014

# Process each year's data
for year in tqdm(range(start_year, end_year + 1), desc="Computing R1mm & R10mm"):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")
    output_file_r1mm = os.path.join(output_dir_r1mm, f"R1mm_{year}.tif")
    output_file_r10mm = os.path.join(output_dir_r10mm, f"R10mm_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

        r1mm_days = np.zeros((src.height, src.width), dtype=np.float32)  # Initialised to 0
        r10mm_days = np.zeros((src.height, src.width), dtype=np.float32)  # Initialised to 0*

        # Process day by day
        for band in range(1, src.count + 1):
            data = src.read(band)

            # Only calculate non-NaN areas
            valid_mask = ~np.isnan(data)
            r1mm_days[valid_mask] += (data[valid_mask] >= 1).astype(np.float32)   # Count wet days
            r10mm_days[valid_mask] += (data[valid_mask] >= 10).astype(np.float32)  # Count heavy rain days

        # Handle NaN values
        r1mm_days[~valid_mask] = np.nan
        r10mm_days[~valid_mask] = np.nan

        # Save R1mm result
        with rasterio.open(output_file_r1mm, "w", **meta) as dst:
            dst.write(r1mm_days, 1)
            dst.set_band_description(1, f"R1mm_{year}")

        # Save R10mm result
        with rasterio.open(output_file_r10mm, "w", **meta) as dst:
            dst.write(r10mm_days, 1)
            dst.set_band_description(1, f"R10mm_{year}")

print("R1mm & R10mm calculation completed for 1961-2014. Results saved to respective folders.")
