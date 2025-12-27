import os
import numpy as np
import rasterio
from tqdm import tqdm

# Input precipitation data directory
pre_dir = r"F:\phdl1\QTP_CN05.1_converted\pre"
# Output directories
output_dir_cdd = r"F:\phdl1\climate extremes\CDD"
output_dir_cwd = r"F:\phdl1\climate extremes\CWD"
os.makedirs(output_dir_cdd, exist_ok=True)
os.makedirs(output_dir_cwd, exist_ok=True)

# Processing year range (only 1961-2014)
start_year, end_year = 1961, 2014

# Process each year's data
for year in tqdm(range(start_year, end_year + 1), desc="Computing CDD & CWD"):
    input_file = os.path.join(pre_dir, f"pre_{year}.tif")
    output_file_cdd = os.path.join(output_dir_cdd, f"CDD_{year}.tif")
    output_file_cwd = os.path.join(output_dir_cwd, f"CWD_{year}.tif")

    if not os.path.exists(input_file):
        print(f"Warning: {input_file} not found, skipping...")
        continue

    with rasterio.open(input_file) as src:
        meta = src.meta.copy()
        meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

        cdd_max = np.zeros((src.height, src.width), dtype=np.float32)  # Maximum consecutive dry days
        cwd_max = np.zeros((src.height, src.width), dtype=np.float32)  # Maximum consecutive wet days

        # Read all daily precipitation data
        precip_data = src.read()  # Shape: (days, height, width)

        # Iterate over each pixel
        for i in range(src.height):
            for j in range(src.width):
                pixel_precip = precip_data[:, i, j]

                if np.all(np.isnan(pixel_precip)):  # Skip invalid data
                    cdd_max[i, j] = np.nan
                    cwd_max[i, j] = np.nan
                    continue

                # Calculate CDD (longest consecutive days with <1 mm precipitation)
                cdd_count = 0
                max_cdd = 0
                cwd_count = 0
                max_cwd = 0

                for day in range(len(pixel_precip)):
                    if pixel_precip[day] < 1:
                        cdd_count += 1
                        max_cdd = max(max_cdd, cdd_count)
                        cwd_count = 0  # Break wet day streak
                    else:
                        cwd_count += 1
                        max_cwd = max(max_cwd, cwd_count)
                        cdd_count = 0  # Break dry day streak

                cdd_max[i, j] = max_cdd
                cwd_max[i, j] = max_cwd

        # Save CDD result
        with rasterio.open(output_file_cdd, "w", **meta) as dst:
            dst.write(cdd_max, 1)
            dst.set_band_description(1, f"CDD_{year}")

        # Save CWD result
        with rasterio.open(output_file_cwd, "w", **meta) as dst:
            dst.write(cwd_max, 1)
            dst.set_band_description(1, f"CWD_{year}")

print("CDD & CWD calculation completed for 1961-2014. Results saved to respective folders.")
