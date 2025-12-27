import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input data paths
tmin_dir = r"F:\phdl1\QTP_CN05.1_converted\tmin"  # TN
tmax_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"  # TX
tmean_dir = r"F:\phdl1\QTP_CN05.1_converted\tmean"  # TM

# Output directories
output_dirs = {
    "FD": r"F:\phdl1\climate extremes\FD",
    "ID": r"F:\phdl1\climate extremes\ID",
    "DTR": r"F:\phdl1\climate extremes\DTR",
    "TFR": r"F:\phdl1\climate extremes\TFR"
}
for d in output_dirs.values():
    os.makedirs(d, exist_ok=True)

# Target baseline period
start_year, end_year = 1961, 2014
TFR_max = 1000  # Set maximum allowable (thaw-freeze rate) TFR value

# Compute indices for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing FD, ID, DTR, TFR"):
    tmin_file = os.path.join(tmin_dir, f"tmin_{year}.tif")
    tmax_file = os.path.join(tmax_dir, f"tmax_{year}.tif")
    tmean_file = os.path.join(tmean_dir, f"tm_{year}.tif")

    # Read TN (minimum temperature)
    with rasterio.open(tmin_file) as src:
        tn_data = src.read()
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output
        height, width = src.height, src.width

    # Read TX (maximum temperature)
    with rasterio.open(tmax_file) as src:
        tx_data = src.read()

    # Read TM (mean temperature, for TFR)
    with rasterio.open(tmean_file) as src:
        tm_data = src.read()

    # Initialise result arrays
    fd = np.zeros((height, width), dtype=np.float32)
    id = np.zeros((height, width), dtype=np.float32)
    dtr = np.zeros((height, width), dtype=np.float32)
    thaw_index = np.zeros((height, width), dtype=np.float32)  # Thawing index
    freeze_index = np.zeros((height, width), dtype=np.float32)  # Freezing index

    # Iterate over all pixels
    for i in range(height):
        for j in range(width):
            tn_series = tn_data[:, i, j]  # TN time series
            tx_series = tx_data[:, i, j]  # TX time series
            tm_series = tm_data[:, i, j]  # TM time series

            # Handle invalid values
            if np.any(np.isnan(tn_series)) or np.any(np.isnan(tx_series)) or np.any(np.isnan(tm_series)):
                fd[i, j] = np.nan
                id[i, j] = np.nan
                dtr[i, j] = np.nan
                thaw_index[i, j] = np.nan
                freeze_index[i, j] = np.nan
                continue

            # Calculate FD (Frost Days)
            fd[i, j] = np.sum(tn_series < 0)

            # Calculate ID (Ice Days)
            id[i, j] = np.sum(tx_series < 0)

            # Calculate DTR (Diurnal Temperature Range)
            dtr[i, j] = np.mean(tx_series - tn_series)

            # Calculate thawing index (cumulative absolute temperature > 0°C)
            thaw_index[i, j] = np.sum(tm_series[tm_series > 0])

            # Calculate freezing index (cumulative absolute temperature < 0°C)
            print(tm_series[tm_series < 0].shape)
            freeze_index[i, j] = np.abs(np.sum(tm_series[tm_series < 0]))

    # Calculate TFR (thawing index / freezing index)
    tfr = np.divide(thaw_index, freeze_index, out=np.full_like(thaw_index, np.nan), where=freeze_index != 0)

    # Outlier handling
    tfr[freeze_index < 1] = np.nan  # Avoid extremely high values
    tfr[tfr > TFR_max] = np.nan  # Set maximum threshold to avoid extreme outliers

    # Output results
    output_files = {
        "FD": os.path.join(output_dirs["FD"], f"FD_{year}.tif"),
        "ID": os.path.join(output_dirs["ID"], f"ID_{year}.tif"),
        "DTR": os.path.join(output_dirs["DTR"], f"DTR_{year}.tif"),
        "TFR": os.path.join(output_dirs["TFR"], f"TFR_{year}.tif"),
    }

    for key, data in zip(output_files.keys(), [fd, id, dtr, tfr]):
        with rasterio.open(output_files[key], "w", **meta) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, f"{key}_{year}")

print("FD, ID, DTR, TFR calculation completed.")
