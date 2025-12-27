import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input data paths
tmax_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"
tmin_dir = r"F:\phdl1\QTP_CN05.1_converted\tmin"

# Output directories
output_base_dir = r"F:\phdl1\climate extremes"
output_dirs = {
    "TXx": os.path.join(output_base_dir, "TXx"),
    "TXn": os.path.join(output_base_dir, "TXn"),
    "TNx": os.path.join(output_base_dir, "TNx"),
    "TNn": os.path.join(output_base_dir, "TNn"),
}

# Ensure each output directory exists
for folder in output_dirs.values():
    os.makedirs(folder, exist_ok=True)

# Target computation years
start_year, end_year = 1961, 2014

# Compute TXx, TXn, TNx, TNn
for year in tqdm(range(start_year, end_year + 1), desc="Computing TXx, TXn, TNx, TNn"):
    tmax_file = os.path.join(tmax_dir, f"tmax_{year}.tif")
    tmin_file = os.path.join(tmin_dir, f"tmin_{year}.tif")

    if not os.path.exists(tmax_file) or not os.path.exists(tmin_file):
        print(f"Skipping {year}, missing data files.")
        continue

    # Read Tmax data
    with rasterio.open(tmax_file) as src:
        tmax_data = src.read()  # Shape (num_days, height, width)
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

    # Read Tmin data
    with rasterio.open(tmin_file) as src:
        tmin_data = src.read()

    # Compute TXx, TXn, TNx, TNn
    txx = np.nanmax(tmax_data, axis=0)  # Maximum Tmax per pixel
    txn = np.nanmin(tmax_data, axis=0)  # Minimum Tmax per pixel
    tnx = np.nanmax(tmin_data, axis=0)  # Maximum Tmin per pixel
    tnn = np.nanmin(tmin_data, axis=0)  # Minimum Tmin per pixel

    # Handle invalid values
    txx[np.isnan(txx)] = np.nan
    txn[np.isnan(txn)] = np.nan
    tnx[np.isnan(tnx)] = np.nan
    tnn[np.isnan(tnn)] = np.nan

    # Save TXx
    txx_file = os.path.join(output_dirs["TXx"], f"TXx_{year}.tif")
    with rasterio.open(txx_file, "w", **meta) as dst:
        dst.write(txx, 1)
        dst.set_band_description(1, f"TXx_{year}")

    # Save TXn
    txn_file = os.path.join(output_dirs["TXn"], f"TXn_{year}.tif")
    with rasterio.open(txn_file, "w", **meta) as dst:
        dst.write(txn, 1)
        dst.set_band_description(1, f"TXn_{year}")

    # Save TNx
    tnx_file = os.path.join(output_dirs["TNx"], f"TNx_{year}.tif")
    with rasterio.open(tnx_file, "w", **meta) as dst:
        dst.write(tnx, 1)
        dst.set_band_description(1, f"TNx_{year}")

    # Save TNn
    tnn_file = os.path.join(output_dirs["TNn"], f"TNn_{year}.tif")
    with rasterio.open(tnn_file, "w", **meta) as dst:
        dst.write(tnn, 1)
        dst.set_band_description(1, f"TNn_{year}")

print("TXx, TXn, TNx, TNn calculation completed. Results saved to respective folders.")
