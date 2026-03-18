import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple


def subset_or_resample(
    df: pd.DataFrame,
    resample_minutes: Optional[int] = None,
    time_window: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Optionally restrict to a time window and/or resample before applying filters.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing datetime index.
    resample_minutes : int, optional
        Number of minutes for resampling (default is None).
    time_window : tuple of pd.Timestamp, optional
        (start, end) timestamps to restrict the data (default is None).
    """
    data = df.copy()

    if time_window is not None:
        start, end = time_window
        start_ts = pd.to_datetime(start)
        end_ts = pd.to_datetime(end)
        data = data.loc[(data.index >= start_ts) & (data.index <= end_ts)]
        print("Data successfully restricted from ", start, " to ", end)

    if resample_minutes is not None:
        data = (
            data.resample(f"{resample_minutes}min")
            .mean(numeric_only=True)
        )
        print("Data successfully resampled to ", resample_minutes, " min averages")

    return data


def filter_ffp(
    df: pd.DataFrame,
    height: str = "1m",
    wind_dir_range: Optional[Tuple[float, float]] = None,
    jet_filter_range: Optional[Tuple[float, float]] = None,
    uw_jet_filter: Optional[bool] = False,
    uT_jet_filter: Optional[bool] = False,
    wind_speed_range: Optional[Tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    Apply optional filters before passing data to FFP_climatology.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing wind direction, wind speed, and covariance columns.
    height : str, optional
        Height suffix for column names (default is "1m").
    wind_dir_range : tuple of float, optional
        (min_deg, max_deg); supports wrap-around if min>max.
    uw_jet_filter : bool, optional
        If True, apply uw jet filter.
    uT_jet_filter : bool, optional
        If True, apply uT jet filter.
    wind_speed_range : tuple of float, optional
        (min_ws, max_ws).
    """
    col_ws = f"wind_speed_{height}"
    col_wd = f"wind_dir_{height}"
    col_cov_uw = f"cov_uw_{height}"
    col_cov_uT = f"cov_uT_{height}"

    mask = pd.Series(True, index=df.index)

    # Pre-compute wind direction as numeric for reuse
    wd = pd.to_numeric(df[col_wd], errors="coerce")

    if wind_dir_range is not None:
        wd_min, wd_max = wind_dir_range
        if wd_min <= wd_max:
            mask &= (wd >= wd_min) & (wd <= wd_max)
        else:
            # wrap-around (e.g., 315–45)
            mask &= ((wd >= wd_min) | (wd <= wd_max))

    # Jet filters: apply only within the specified directional window, otherwise allow values through
    if jet_filter_range is None:
        jet_mask = pd.Series(True, index=df.index)
    else:
        jet_min, jet_max = jet_filter_range
        if jet_min <= jet_max:
            jet_mask = wd.between(jet_min, jet_max, inclusive="both")
        else:
            jet_mask = (wd >= jet_min) | (wd <= jet_max)

    if uw_jet_filter is not False:
        mask &= (~jet_mask) | (df[col_cov_uw] <= 0)

    if uT_jet_filter is not False:
        mask &= (~jet_mask) | (df[col_cov_uT] >= 0)

    if wind_speed_range is not None:
        ws_min, ws_max = wind_speed_range
        ws = pd.to_numeric(df[col_ws], errors="coerce")
        mask &= (ws >= ws_min) & (ws <= ws_max)

    print("Data successfully filtered!")
    return df.loc[mask]



def footprint_area(xr, yr):
    """
    Compute area enclosed by a footprint contour.

    Parameters
    ----------
    xr : array-like
        x-coordinates of the contour (meters).
    yr : array-like
        y-coordinates of the contour (meters).

    Returns
    -------
    float
        Area in square meters.
    """
    x = np.asarray(xr, dtype=float)
    y = np.asarray(yr, dtype=float)
    if x.size != y.size:
        raise ValueError("xr and yr must have the same length")
    if x.size < 3:
        raise ValueError("At least 3 points are required to form a polygon")

    # Ensure the polygon is closed
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.append(x, x[0])
        y = np.append(y, y[0])

    # Classic shoelace (Gauss) area formula
    area = 0.5 * np.abs(np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:]))
    return area
