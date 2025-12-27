import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input temperature data paths
tmean_dir = r"F:\phdl1\QTP_CN05.1_converted\tmean"

# Output directories
freeze_index_dir = r"F:\phdl1\climate extremes\Freeze_Index"
thaw_index_dir = r"F:\phdl1\climate extremes\Thaw_Index"

# Create output directories
os.makedirs(freeze_index_dir, exist_ok=True)
os.makedirs(thaw_index_dir, exist_ok=True)

# Target baseline period
start_year, end_year = 1961, 2014

# Compute Freezing Index (FI) and Thawing Index (TI)
for year in tqdm(range(start_year, end_year + 1), desc="Computing Freeze and Thaw Index"):
    input_file = os.path.join(tmean_dir, f"tm_{year}.tif")

    # Read data
    with rasterio.open(input_file) as src:
        num_days = src.count
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

        freeze_index = np.zeros((src.height, src.width), dtype=np.float32)
        thaw_index = np.zeros((src.height, src.width), dtype=np.float32)
        nan_mask = np.zeros((src.height, src.width), dtype=bool)

        for band in range(1, num_days + 1):
            data = src.read(band)
            nan_mask |= np.isnan(data)  # Record invalid value areas

            freeze_index += np.abs(data) * (data < 0)  # Freezing Index (accumulation of absolute negative temperatures)
            thaw_index += data * (data > 0)  # Thawing Index (accumulation of positive temperatures)

        # Handle invalid values
        freeze_index[nan_mask] = np.nan
        thaw_index[nan_mask] = np.nan

    # Output Freezing Index
    freeze_output_file = os.path.join(freeze_index_dir, f"Freeze_Index_{year}.tif")
    with rasterio.open(freeze_output_file, "w", **meta) as dst:
        dst.write(freeze_index, 1)
        dst.set_band_description(1, f"Freeze_Index_{year}")

    # Output Thawing Index
    thaw_output_file = os.path.join(thaw_index_dir, f"Thaw_Index_{year}.tif")
    with rasterio.open(thaw_output_file, "w", **meta) as dst:
        dst.write(thaw_index, 1)
        dst.set_band_description(1, f"Thaw_Index_{year}")

print("Freezing Index and Thawing Index calculation completed. Results saved.")
