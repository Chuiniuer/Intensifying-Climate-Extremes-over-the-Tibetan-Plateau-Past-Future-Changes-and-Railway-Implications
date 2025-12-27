import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmin"
tnin10_file = r"F:\phdl1\climate extremes\TN10p\threshold\TNin10.tif"
output_dir = r"F:\phdl1\climate extremes\CSDI\yearly"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read TNin10 (10th percentile for baseline period)
with rasterio.open(tnin10_file) as src:
    tnin10 = src.read()  # Read all 366 days of 10th percentile values
    meta = src.meta.copy()  # Copy metadata
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

# Record cold wave days from the end of the previous year
prev_year_tail = np.zeros((meta["height"], meta["width"]), dtype=np.int32)

# Compute CSDI for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing CSDI"):
    input_file = os.path.join(data_dir, f"tmin_{year}.tif")
    next_year_file = os.path.join(data_dir, f"tmin_{year + 1}.tif") if year < end_year else None

    # Read minimum temperature data for the year
    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        csdi = np.zeros((src.height, src.width), dtype=np.float32)  # Initialise CSDI count array
        nan_mask = np.zeros((src.height, src.width), dtype=bool)  # Record invalid value areas
        cold_wave_mask = np.zeros((num_days, src.height, src.width), dtype=bool)  # Current year only

        # Read data for the current year
        for band in range(1, num_days + 1):  # 1-based index
            date_str = src.descriptions[band - 1]  # Read date
            _, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read data for the day
            invalid_mask = np.isnan(data)  # Invalid value mask
            nan_mask |= invalid_mask  # Record whether pixel is invalid

            # Mark days below TNin10
            cold_wave_mask[band - 1] = (data < tnin10[day_of_year]) & (~invalid_mask)

        # Calculate CSDI for the current year
        year_end_tail = np.zeros((src.height, src.width), dtype=np.int32)  # Record cold wave days at year end
        for i in range(src.height):
            for j in range(src.width):
                if nan_mask[i, j]:  # Skip invalid values
                    continue

                cw_series = cold_wave_mask[:, i, j]
                count = 0  # Cold wave count for current year
                csdi_value = 0
                first_non_cold_day = False  # Whether the first non-cold day has been encountered

                for d in range(num_days):
                    if cw_series[d]:  # Day is a cold day
                        count += 1
                    else:  # Encountered a non-cold day
                        if prev_year_tail[i, j] > 0 and not first_non_cold_day:
                            csdi_value += count  # Unconditionally add previous year's cold wave days
                            prev_year_tail[i, j] = 0  # Reset after adding
                            first_non_cold_day = True  # Mark that first non-cold day has been encountered
                            count = 0
                            continue
                        if count >= 6:
                            csdi_value += count  # Add to CSDI
                        count = 0  # Reset count

                # Temporarily store cold wave days at year end
                year_end_tail[i, j] = count

                # CSDI result for this year
                csdi[i, j] = csdi_value

        # Read Tmin data for the beginning of next year
        if next_year_file and os.path.exists(next_year_file):
            with rasterio.open(next_year_file) as next_src:
                next_num_days = next_src.count
                next_cold_wave = np.zeros((min(6, next_num_days), src.height, src.width), dtype=bool)

                for band in range(1, min(7, next_num_days + 1)):  # Take up to 6 days
                    date_str = next_src.descriptions[band - 1]  # Read date
                    _, month, day = map(int, date_str.split("-"))
                    day_of_year = (datetime(year + 1, month, day) - datetime(year + 1, 1, 1)).days  # 0-based

                    data = next_src.read(band)  # Read data for the day
                    invalid_mask = np.isnan(data)
                    nan_mask |= invalid_mask  # Continue recording invalid values

                    # Mark days below TNin10
                    next_cold_wave[band - 1] = (data < tnin10[day_of_year]) & (~invalid_mask)

                # If the beginning of the year is still a cold wave, consider merging with year_end_tail
                prev_year_tail.fill(0)  # Reset first
                for i in range(src.height):
                    for j in range(src.width):
                        if nan_mask[i, j]:  # Skip invalid values
                            continue

                        # Calculate consecutive cold wave days at the beginning of next year
                        next_start_count = 0
                        for d in range(min(6, next_num_days)):
                            if next_cold_wave[d, i, j]:
                                next_start_count += 1
                            else:
                                break

                        # Determine if the sum of year_end_tail and next_start_count is >= 6
                        if year_end_tail[i, j] + next_start_count >= 6:
                            csdi[i, j] += year_end_tail[i, j]  # Add to current year's CSDI first
                            prev_year_tail[i, j] = next_start_count  # Pass to next year

        # Handle invalid values
        csdi[nan_mask] = np.nan

        # Debug output
        print(f"Year {year}: CSDI min={np.nanmin(csdi)}, max={np.nanmax(csdi)}")

    # Output CSDI result
    output_file = os.path.join(output_dir, f"CSDI_{year}.tif")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(csdi, 1)  # Write single band
        dst.set_band_description(1, f"CSDI_{year}")

print("CSDI calculation completed. Results saved to:", output_dir)
