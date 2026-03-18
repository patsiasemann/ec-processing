import pandas as pd
import numpy as np
import importlib, calc_footprint_FFP_climatology as footprint
importlib.reload(footprint)

from src.flux_footprints import (
    subset_or_resample,
    filter_ffp,
    footprint_area,
)

# ---------------------------------------------------------------------
# DEFINE PARAMETERS

# Measurement heights (m) 
z1 = 1.1
z2 = 2.1
z3 = 3.1

# Wind direction ranges for filtering (degrees)
katabatic = (80, 160)
anabatic = (240, 300)
crosswind = (170, 230)


# ---------------------------------------------------------------------
# SET FILTERS AND RESTRICTIONS

# Set the measurement height for footprint computation
level = z1
height = "1m"

# Time window and/or averaging time for analysis
time_window = None # tuple of pd.Timestamp, e.g. ("2023-06-23 00:00", "2023-07-23 23:59")
resample_minutes = None # float, e.g. 30 for 30-minute averages; None to keep original resolution

# Set the wind direction range for filtering
wind_dir_min, wind_dir_max = katabatic

# Wind speed range for filtering (m/s)
wind_speed_range = None # tuple of floats, e.g. (0.5, 10.0)

# Jet filters (True to apply, False to skip)
uw_jet_filter = False
uT_jet_filter = False
jet_filter_range = katabatic

# ---------------------------------------------------------------------

# Import the roughness data with semicolon delimiter
file_path = "Data/SILVEX2_Silvia2_roughness_ffp_data.CSV"
df = pd.read_csv(file_path, sep=';', skiprows=[1], low_memory=False)
df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%d.%m.%Y %H:%M")

df = df.sort_values("datetime").set_index("datetime")
for col in [f"wind_speed_{height}", f"L_{height}", f"u*_{height}", f"var_v_{height}", f"wind_dir_{height}", f"cov_uw_{height}", f"cov_uT_{height}"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df[f"sigma_v_{height}"] = np.sqrt(df[f"var_v_{height}"])

# Average and/or subset the data before filtering
df_toprocess = subset_or_resample(
    df,
    resample_minutes=resample_minutes,
    time_window=time_window,
)
if df_toprocess.empty:
    raise ValueError("No data left after time selection; adjust restrictions.")

# Apply wind direction, wind speed and flux filters
df_filtered = filter_ffp(
    df_toprocess,
    height=height,
    wind_dir_range=(wind_dir_min, wind_dir_max),
    uw_jet_filter=uw_jet_filter,
    uT_jet_filter=uT_jet_filter,
    wind_speed_range=wind_speed_range,
)

# Directional jet filter at 1-minute resolution
df_compute = df_filtered.dropna(subset=[f"wind_speed_{height}", f"L_{height}", f"u*_{height}", f"var_v_{height}", f"sigma_v_{height}", f"wind_dir_{height}"])
if df_compute.empty:
    raise ValueError("No data left after filtering; adjust filter settings.")


def compute_interval_footprint_areas(df: pd.DataFrame, height: str = height, zm: float = z1) -> pd.DataFrame:
    """Compute 30-min footprint areas and retain contours for 10–80% levels.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the necessary columns for FFP computation at 1-minute resolution.
    height : str, optional
        Height suffix to use for column names (e.g., height), by default height.
    zm : float, optional
        Measurement height above displacement height [m], by default z1.
    """
    col_ws = f"wind_speed_{height}"
    col_L = f"L_{height}"
    col_ustar = f"u*_{height}"
    col_sigmav = f"sigma_v_{height}"
    col_wd = f"wind_dir_{height}"

    area_cols = [f"area_{p}" for p in range(10, 90, 10)]
    contour_cols = [f"contour_{p}" for p in range(10, 90, 10)]
    records = []
    contour_records = []
    count_records = []
    min_samples = 5  # skip intervals with too few 1-minute points

    # Standard 30-min bins, left-labeled (e.g., 11:30 represents [11:30, 12:00))
    for ts, frame in df.resample("30min", label="left", closed="left"):
        subset = frame[[col_ws, col_L, col_ustar, col_sigmav, col_wd]].dropna()
        valid_count = len(subset)
        count_records.append([ts, valid_count])
        if valid_count < min_samples:
            continue

        try:
            FFP_bin = footprint.FFP_climatology(
                zm=zm,
                z0=None,
                umean=subset[col_ws].tolist(),
                h=[100.0] * len(subset),
                ol=subset[col_L].tolist(),
                sigmav=subset[col_sigmav].tolist(),
                ustar=subset[col_ustar].tolist(),
                wind_dir=subset[col_wd].tolist(),
                verbosity=0,
                fig=False,
                show_heatmap=False,
                domain=[-200, 200, -200, 200],
                dx=0.5,
                dy=0.5,
            )
        except Exception as exc:
            print(f"Skipping interval starting {ts} due to FFP error: {exc}")
            continue

        areas = []
        contours = []
        for i in range(8):
            xr = FFP_bin["xr"][i]
            yr = FFP_bin["yr"][i]
            if xr is None or yr is None:
                areas.append(np.nan)
                contours.append(None)
                continue
            if len(xr) < 3 or len(yr) < 3:
                areas.append(np.nan)
                contours.append(None)
                continue
            if len(xr) != len(yr):
                print(f"Skipping contour {i} for interval starting {ts}: xr/yr length mismatch")
                areas.append(np.nan)
                contours.append(None)
                continue
            areas.append(footprint_area(xr, yr))
            contours.append((xr, yr))

        records.append([ts] + areas)
        contour_records.append([ts] + contours)
        print(f"Processed interval starting {ts}: valid samples={valid_count}")

    result_areas = pd.DataFrame(records, columns=["datetime"] + area_cols)
    result_contours = pd.DataFrame(contour_records, columns=["datetime"] + contour_cols)
    result_counts = pd.DataFrame(count_records, columns=["datetime", "valid_samples"])
    return result_areas, result_contours, result_counts

# Compute per-interval footprint areas (10–80%) for the current 1-minute filtered dataset
interval_footprint_areas, interval_footprint_contours, interval_valid_counts = compute_interval_footprint_areas(df_compute, height=height, zm=level)
print(interval_footprint_areas.head())

# Save outputs to CSVs for later
interval_footprint_areas.to_csv("Data/interval_footprint_areas.csv", index=False)
interval_footprint_contours.to_csv("Data/interval_footprint_contours.csv", index=False)
interval_valid_counts.to_csv("Data/interval_valid_counts.csv", index=False)