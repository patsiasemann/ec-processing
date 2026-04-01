"""Build roughness datasets for multiple sites.

This script reads EddyPro full_output CSVs (comma-delimited) for SILVEX
sites and HEFEXIII NetCDF flux data, then writes per-site roughness
files matching the SILVEX2_Silvia2 format. Footprint fields that are not
available in a source are left as NaN. Output files are written with a
semicolon delimiter.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import xarray as xr

KAPPA = 0.4


# Column names we want per height in the final CSV.
BASE_FIELDS = [
	"wind_speed",
	"wind_dir",
	"u*",
	"TKE",
	"L",
	"(z-d)/L",
	"T*",
	"var_v",
	"cov_uw",
	"cov_uT",
	"cov_wT",
	"footprintmodel",
	"x_peak",
	"x_offset",
	"x_10%",
	"x_30%",
	"x_50%",
	"x_70%",
	"x_90%",
]


# Mapping from EddyPro full_output column names to the standard names above.
EDDYPRO_MAP: Dict[str, str] = {
	"wind_speed": "wind_speed",
	"wind_dir": "wind_dir",
	"u*": "u*",
	"TKE": "TKE",
	"L": "L",
	"(z-d)/L": "(z-d)/L",
	"T*": "T*",
	"var_v": "v_var",
	"cov_uw": "cov_uw",
	"cov_uT": "cov_uT",
	"cov_wT": "cov_wT",
	"footprintmodel": "model",
	"x_peak": "x_peak",
	"x_offset": "x_offset",
	"x_10%": "x_10%",
	"x_30%": "x_30%",
	"x_50%": "x_50%",
	"x_70%": "x_70%",
	"x_90%": "x_90%",
}


# Mapping from HEFEXIII NetCDF variables to the standard names.
HEFEX_MAP: Dict[str, str] = {
	"wind_speed": "meanU",
	"wind_dir": "sdir",
	"u*": "ustar",
	"TKE": "tke",
	"cov_uw": "uw",
	"cov_uT": "uT",
	"cov_wT": "wT",
	"var_v": "vv",
	# L, (z-d)/L, T*, footprint fields are unavailable and will be NaN.
}


@dataclass
class HeightConfig:
	label: str
	measurement_height: float


@dataclass
class SiteConfig:
	name: str
	kind: str  # "eddypro" or "hefex_nc"
	base_dir: Path
	heights: List[HeightConfig]


def to_output_date(date_series: pd.Series, time_series: pd.Series) -> pd.DataFrame:
	"""Return a DataFrame with formatted date/time columns."""
	dt = pd.to_datetime(date_series.astype(str) + " " + time_series.astype(str), errors="coerce")
	return pd.DataFrame(
		{
			"date": dt.dt.strftime("%d.%m.%Y"),
			"time": dt.dt.strftime("%H:%M"),
		}
	)


def compute_z0(wind_speed: pd.Series, ustar: pd.Series, z: float) -> pd.Series:
	with np.errstate(divide="ignore", invalid="ignore"):
		mask = ustar > 0
		out = pd.Series(np.nan, index=wind_speed.index)
		out.loc[mask] = z * np.exp(-wind_speed.loc[mask] * KAPPA / ustar.loc[mask])
	return out


def load_eddypro_height(csv_path: Path, height: HeightConfig) -> pd.DataFrame:
	df_raw = pd.read_csv(csv_path, sep=",", na_values=["NaN"])
	date_time = to_output_date(df_raw["date"], df_raw["time"])

	data = date_time.copy()
	for out_name, src_name in EDDYPRO_MAP.items():
		col = df_raw[src_name] if src_name in df_raw.columns else np.nan
		data[f"{out_name}_{height.label}"] = col

	data[f"z0_{height.label}"] = compute_z0(
		data[f"wind_speed_{height.label}"], data[f"u*_{height.label}"], height.measurement_height
	)
	return data


def load_hefex_height(ds: xr.Dataset, height: HeightConfig) -> pd.DataFrame:
	ds_sel = ds.sel(heights=height.measurement_height)
	df_raw = ds_sel.to_dataframe().reset_index()
	dt = pd.to_datetime(df_raw["time"], errors="coerce")
	data = pd.DataFrame({"date": dt.dt.strftime("%d.%m.%Y"), "time": dt.dt.strftime("%H:%M")})
	for out_name in BASE_FIELDS:
		src = HEFEX_MAP.get(out_name)
		series = df_raw[src] if src in df_raw.columns else np.nan
		data[f"{out_name}_{height.label}"] = series

	data[f"z0_{height.label}"] = compute_z0(
		data[f"wind_speed_{height.label}"], data[f"u*_{height.label}"], height.measurement_height
	)
	return data


def merge_heights(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
	def _merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
		return pd.merge(left, right, on=["date", "time"], how="outer")

	frames = list(frames)
	if not frames:
		return pd.DataFrame()
	return reduce(_merge, frames)


def column_order(heights: List[HeightConfig]) -> List[str]:
	order: List[str] = ["date", "time"]
	for h in heights:
		order.extend([f"{field}_{h.label}" for field in BASE_FIELDS])
	order.extend([f"z0_{h.label}" for h in heights])
	return order


def find_csv(base_dir: Path) -> Optional[Path]:
	matches = sorted(base_dir.glob("*full_output*_adv.csv"))
	return matches[0] if matches else None


def process_site(site: SiteConfig, output_dir: Path) -> None:
	frames: List[pd.DataFrame] = []

	if site.kind == "eddypro":
		for height in site.heights:
			csv_path = find_csv(site.base_dir / height.label)
			if not csv_path:
				continue
			frames.append(load_eddypro_height(csv_path, height))
	elif site.kind == "hefex_nc":
		nc_path = site.base_dir / "fluxes_20Hz.nc"
		ds = xr.open_dataset(nc_path)
		for height in site.heights:
			frames.append(load_hefex_height(ds, height))
	else:
		raise ValueError(f"Unknown site kind: {site.kind}")

	combined = merge_heights(frames)
	if combined.empty:
		return

	combined = combined.sort_values(["date", "time"])
	combined = combined[column_order(site.heights)]

	output_dir.mkdir(parents=True, exist_ok=True)
	out_path = output_dir / f"{site.name}_roughness_ffp_data.CSV"
	combined.to_csv(out_path, sep=";", index=False)


def main() -> None:
	root = Path(__file__).resolve().parent
	data_dir = root / "data"

	sites: List[SiteConfig] = [
		SiteConfig(
			name="SILVEX2_Silvia1",
			kind="eddypro",
			base_dir=data_dir / "SILVEX2_Silvia1",
			heights=[
				HeightConfig("1m", 1.1),
				HeightConfig("2m", 2.1),
				HeightConfig("3m", 3.1),
			],
		),
		SiteConfig(
			name="SILVEX1_Silvia1",
			kind="eddypro",
			base_dir=data_dir / "SILVEX1_Silvia1",
			heights=[HeightConfig("1m", 1.1), HeightConfig("2m", 2.1)],
		),
		SiteConfig(
			name="SILVEX1_Silvia2",
			kind="eddypro",
			base_dir=data_dir / "SILVEX1_Silvia2",
			heights=[HeightConfig("1m", 1.1), HeightConfig("2m", 2.1)],
		),
		SiteConfig(
			name="SILVEX1_Silvia3",
			kind="eddypro",
			base_dir=data_dir / "SILVEX1_Silvia3",
			heights=[HeightConfig("1m", 1.1), HeightConfig("2m", 2.1)],
		),
		SiteConfig(
			name="HEFEXIII",
			kind="hefex_nc",
			base_dir=data_dir / "HEFEXIII",
			heights=[
				HeightConfig("0.5m", 0.5),
				HeightConfig("3m", 3.0),
				HeightConfig("5m", 5.0),
				HeightConfig("9m", 9.0),
			],
		),
	]

	for site in sites:
		process_site(site, data_dir)


if __name__ == "__main__":
	main()
