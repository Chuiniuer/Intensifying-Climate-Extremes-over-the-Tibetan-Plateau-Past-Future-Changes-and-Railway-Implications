import os
import numpy as np
import rasterio
from tqdm import tqdm
from datetime import datetime

# Input data paths
data_dir = r"F:\phdl1\QTP_CN05.1_converted\tmax"
txin90_file = r"F:\phdl1\climate extremes\TX90p\threshold\TXin90.tif"
output_dir = r"F:\phdl1\climate extremes\WSDI\yearly"

# Target baseline period
start_year, end_year = 1961, 2014

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Read TXin90 (90th percentile for baseline period)
with rasterio.open(txin90_file) as src:
    txin90 = src.read()  # Read all 366 days of 90th percentile values
    meta = src.meta.copy()  # Copy metadata
    meta.update({"count": 1, "dtype": "float32", "compress": "lzw"})  # Adapt for single-band output

# Record heat wave days from the end of the previous year
prev_year_tail = np.zeros((meta["height"], meta["width"]), dtype=np.int32)

# Compute WSDI for each year
for year in tqdm(range(start_year, end_year + 1), desc="Computing WSDI"):
    input_file = os.path.join(data_dir, f"tmax_{year}.tif")
    next_year_file = os.path.join(data_dir, f"tmax_{year + 1}.tif") if year < end_year else None

    # Read maximum temperature data for the year
    with rasterio.open(input_file) as src:
        num_days = src.count  # Number of days in the year (365 or 366)
        wsdi = np.zeros((src.height, src.width), dtype=np.float32)  # Initialise WSDI count array
        nan_mask = np.zeros((src.height, src.width), dtype=bool)  # Record invalid value areas
        heat_wave_mask = np.zeros((num_days, src.height, src.width), dtype=bool)  # Current year only

        # Read data for the current year
        for band in range(1, num_days + 1):  # 1-based index
            date_str = src.descriptions[band - 1]  # Read date
            _, month, day = map(int, date_str.split("-"))
            day_of_year = (datetime(year, month, day) - datetime(year, 1, 1)).days  # 0-based

            data = src.read(band)  # Read data for the day
            invalid_mask = np.isnan(data)  # Invalid value mask
            nan_mask |= invalid_mask  # Record whether pixel is invalid

            # Mark days exceeding TXin90
            heat_wave_mask[band - 1] = (data > txin90[day_of_year]) & (~invalid_mask)

        # Calculate WSDI for the current year
        year_end_tail = np.zeros((src.height, src.width), dtype=np.int32)  # Record heat wave days at year end
        for i in range(src.height):
            for j in range(src.width):
                if nan_mask[i, j]:  # Skip invalid values
                    continue

                hw_series = heat_wave_mask[:, i, j]
                count = 0  # Heat wave count for current year
                wsdi_value = 0
                first_non_warm_day = False  # Whether the first non-warm day has been encountered

                for d in range(num_days):
                    if hw_series[d]:  # Day is a warm day
                        count += 1
                    else:  # Encountered a non-warm day
                        if prev_year_tail[i, j] > 0 and not first_non_warm_day:
                            wsdi_value += count  # Unconditionally add previous year's heat wave days
                            prev_year_tail[i, j] = 0  # Reset after adding
                            first_non_warm_day = True  # Mark that first non-warm day has been encountered
                            count = 0
                            continue
                        if count >= 6:
                            wsdi_value += count  # Add to WSDI
                        count = 0  # Reset count

                # Temporarily store heat wave days at year end
                year_end_tail[i, j] = count

                # WSDI result for this year
                wsdi[i, j] = wsdi_value

        # Read Tmax data for the beginning of next year
        if next_year_file and os.path.exists(next_year_file):
            with rasterio.open(next_year_file) as next_src:
                next_num_days = next_src.count
                next_heat_wave = np.zeros((min(6, next_num_days), src.height, src.width), dtype=bool)

                for band in range(1, min(7, next_num_days + 1)):  # Take up to 6 days
                    date_str = next_src.descriptions[band - 1]  # Read date
                    _, month, day = map(int, date_str.split("-"))
                    day_of_year = (datetime(year + 1, month, day) - datetime(year + 1, 1, 1)).days  # 0-based

                    data = next_src.read(band)  # Read data for the day
                    invalid_mask = np.isnan(data)
                    nan_mask |= invalid_mask  # Continue recording invalid values

                    # Mark days exceeding TXin90
                    next_heat_wave[band - 1] = (data > txin90[day_of_year]) & (~invalid_mask)

                # If the beginning of the year is still a heat wave, consider merging with year_end_tail
                prev_year_tail.fill(0)  # Reset first
                for i in range(src.height):
                    for j in range(src.width):
                        if nan_mask[i, j]:  # Skip invalid values
                            continue

                        # Calculate consecutive heat wave days at the beginning of next year
                        next_start_count = 0
                        for d in range(min(6, next_num_days)):
                            if next_heat_wave[d, i, j]:
                                next_start_count += 1
                            else:
                                break

                        # Determine if the sum of year_end_tail and next_start_count is >= 6
                        if year_end_tail[i, j] + next_start_count >= 6:
                            wsdi[i, j] += year_end_tail[i, j]  # Add to current year's WSDI first
                            prev_year_tail[i, j] = next_start_count  # Pass to next year
        # Handle invalid values
        wsdi[nan_mask] = np.nan

        # Debug output
        print(f"Year {year}: WSDI min={np.nanmin(wsdi)}, max={np.nanmax(wsdi)}")

    # Output WSDI result
    output_file = os.path.join(output_dir, f"WSDI_{year}.tif")
    with rasterio.open(output_file, "w", **meta) as dst:
        dst.write(wsdi, 1)  # Write single band
        dst.set_band_description(1, f"WSDI_{year}")

print("WSDI calculation completed. Results saved to:", output_dir)
