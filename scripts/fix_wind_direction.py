"""Correct EddyPro wind direction using time-varying sensor orientation.

The EddyPro processing assumed a constant orientation (default: 150 deg).
This script applies a time-dependent orientation timeline and linearly
interpolates orientation between provided timestamps, then updates wind
direction accordingly.

Correction model:
    wind_dir_corrected = wind_dir_raw + (orientation_true(t) - orientation_assumed)

Usage examples:
	python fix_wind_direction.py
	python fix_wind_direction.py --in-place
	python fix_wind_direction.py --orientation-file orientation_timeline.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/SILVEX1_Silvia3")
DEFAULT_FILE_GLOB = "**/eddypro_*_full_output_*_adv.csv"
DEFAULT_ASSUMED_ORIENTATION = 150.0

EMBEDDED_ASSUMED_ORIENTATION_BY_SITE: dict[str, dict[str, float]] = {
	"silvex1_silvia1": {"1m": 175.0, "2m": 175.0},
	"silvex1_silvia3": {"1m": 150.0, "2m": 150.0},
}

# Embedded orientation timelines from field notes.
# Used by default so no external CSV is required.
EMBEDDED_ORIENTATION_POINTS_BY_SITE: dict[str, dict[str, list[tuple[str, float]]]] = {
	"silvex1_silvia1": {
		"1m": [
			("2024-08-21 12:56", 181.0),
			("2024-08-22 17:20", 175.0),
			("2024-08-23 09:00", 174.0),
			("2024-08-25 09:17", 175.0),
			("2024-08-27 14:07", 147.0),
			("2024-08-27 14:10", 161.0),
			("2024-08-29 12:42", 165.0),
			("2024-08-31 12:37", 182.0),
			("2024-08-31 12:46", 183.0),
			("2024-09-06 09:58", 150.0),
		],
		"2m": [
			("2024-08-21 12:56", 182.0),
			("2024-08-22 17:20", 178.0),
			("2024-08-23 09:00", 176.0),
			("2024-08-25 09:17", 174.0),
			("2024-08-27 14:07", 148.0),
			("2024-08-27 14:10", 162.0),
			("2024-08-29 12:42", 166.0),
			("2024-08-31 12:37", 183.0),
			("2024-08-31 12:46", 185.0),
			("2024-09-06 09:58", 152.0),
		],
	},
	"silvex1_silvia3": {
		"1m": [
			("2024-08-21 16:21", 149.0),
			("2024-08-22 14:45", 146.0),
			("2024-08-25 09:45", 156.0),
			("2024-08-25 10:07", 165.0),
			("2024-08-29 12:25", 173.0),
			("2024-08-31 11:30", 191.0),
			("2024-09-06 10:00", 208.0),
		],
		"2m": [
			("2024-08-21 16:21", 151.0),
			("2024-08-22 14:45", 148.0),
			("2024-08-25 09:45", 155.0),
			("2024-08-25 10:07", 163.0),
			("2024-08-29 12:25", 172.0),
			("2024-08-31 11:30", 190.0),
			("2024-09-06 10:00", 205.0),
		],
	},
}


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Apply time-varying orientation correction to EddyPro wind direction.",
	)
	parser.add_argument(
		"--orientation-file",
		type=Path,
		help=(
			"Optional CSV with timestamp and orientation columns. "
			"If omitted, uses embedded timelines for known sites and heights."
		),
	)
	parser.add_argument(
		"--data-dir",
		type=Path,
		default=DEFAULT_DATA_DIR,
		help=f"Directory containing EddyPro files (default: {DEFAULT_DATA_DIR}).",
	)
	parser.add_argument(
		"--file-glob",
		default=DEFAULT_FILE_GLOB,
		help=f"Glob for input files under data-dir (default: {DEFAULT_FILE_GLOB}).",
	)
	parser.add_argument(
		"--assumed-orientation",
		type=float,
		default=None,
		help=(
			"Orientation that EddyPro assumed as constant. "
			"If omitted, uses embedded site/height-specific defaults where available, "
			f"otherwise falls back to {DEFAULT_ASSUMED_ORIENTATION}."
		),
	)
	parser.add_argument(
		"--in-place",
		action="store_true",
		help="Overwrite input files. If omitted, writes *_windcorr.csv outputs.",
	)
	parser.add_argument(
		"--output-suffix",
		default="_windcorr",
		help="Suffix for output files when not using --in-place.",
	)
	return parser.parse_args()


def normalize_degrees(values: pd.Series | np.ndarray) -> np.ndarray:
	arr = np.asarray(values, dtype=float)
	return np.mod(arr, 360.0)


def normalize_timeline(timeline: pd.DataFrame) -> pd.DataFrame:
	timeline = timeline.copy()
	timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], errors="coerce")
	timeline["orientation_deg"] = pd.to_numeric(timeline["orientation_deg"], errors="coerce")
	timeline = timeline.dropna(subset=["timestamp", "orientation_deg"]).sort_values("timestamp")

	if timeline.empty:
		raise ValueError("No valid timestamp/orientation rows after parsing orientation data.")

	timeline["orientation_deg"] = normalize_degrees(timeline["orientation_deg"])
	timeline = timeline.groupby("timestamp", as_index=False)["orientation_deg"].mean()

	if len(timeline) < 2:
		raise ValueError("At least two orientation points are required for interpolation.")

	return timeline


def get_embedded_timelines_for_site(site_key: str) -> dict[str, pd.DataFrame]:
	if site_key not in EMBEDDED_ORIENTATION_POINTS_BY_SITE:
		known = sorted(EMBEDDED_ORIENTATION_POINTS_BY_SITE)
		raise ValueError(f"No embedded orientation timeline for site '{site_key}'. Known sites: {known}")

	timelines: dict[str, pd.DataFrame] = {}
	for height, points in EMBEDDED_ORIENTATION_POINTS_BY_SITE[site_key].items():
		df = pd.DataFrame(points, columns=["timestamp", "orientation_deg"])
		timelines[height] = normalize_timeline(df)
	return timelines


def load_orientation_timeline(path: Path) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Orientation file not found: {path}")

	timeline = pd.read_csv(path)
	if timeline.empty:
		raise ValueError(f"Orientation file is empty: {path}")

	lower_to_actual = {c.lower().strip(): c for c in timeline.columns}

	# Accept a few common naming variants.
	timestamp_col = None
	for candidate in ("timestamp", "datetime", "time", "date_time"):
		if candidate in lower_to_actual:
			timestamp_col = lower_to_actual[candidate]
			break

	orientation_col = None
	for candidate in (
		"orientation_deg",
		"orientation",
		"sensor_orientation",
		"azimuth",
	):
		if candidate in lower_to_actual:
			orientation_col = lower_to_actual[candidate]
			break

	if timestamp_col is None or orientation_col is None:
		raise ValueError(
			"Orientation file must contain timestamp and orientation columns. "
			"Accepted timestamp names: timestamp, datetime, time, date_time. "
			"Accepted orientation names: orientation_deg, orientation, sensor_orientation, azimuth."
		)

	timeline = timeline[[timestamp_col, orientation_col]].copy()
	timeline.columns = ["timestamp", "orientation_deg"]
	return normalize_timeline(timeline)



def infer_site_and_sensor_height(path: Path) -> tuple[str, str]:
	site_key = None
	for part in path.parts:
		part_l = part.lower().strip()
		if part_l in EMBEDDED_ORIENTATION_POINTS_BY_SITE:
			site_key = part_l
			break

	if site_key is None:
		raise ValueError(
			f"Could not infer site from path '{path}'. "
			f"Expected one of: {sorted(EMBEDDED_ORIENTATION_POINTS_BY_SITE)}"
		)

	for part in path.parts:
		part_l = part.lower().strip()
		if part_l in EMBEDDED_ORIENTATION_POINTS_BY_SITE[site_key]:
			return site_key, part_l

	raise ValueError(
		f"Could not infer sensor height from path '{path}'. "
		"Expected path component like '1m' or '2m'."
	)


def get_assumed_orientation(site_key: str, height: str, override: float | None) -> float:
	if override is not None:
		return float(override)

	site_defaults = EMBEDDED_ASSUMED_ORIENTATION_BY_SITE.get(site_key, {})
	if height in site_defaults:
		return float(site_defaults[height])

	return float(DEFAULT_ASSUMED_ORIENTATION)


def read_eddypro_with_header(path: Path) -> tuple[pd.DataFrame, list[str], list[str], str]:
	with path.open("r", encoding="utf-8", newline="") as f:
		reader = csv.reader(f)
		rows = [next(reader) for _ in range(3)]

	file_info_row, header_row, units_row = rows

	# EddyPro adv CSV format: metadata row, header row, units row, then data rows.
	df = pd.read_csv(path, header=1, skiprows=[2])
	return df, file_info_row, header_row, units_row


def interpolate_orientation(
	target_timestamps: pd.Series,
	timeline: pd.DataFrame,
) -> np.ndarray:
	"""Interpolate circular orientation angles onto target timestamps."""

	t_target = target_timestamps.astype("int64").to_numpy(dtype=float)
	t_src = timeline["timestamp"].astype("int64").to_numpy(dtype=float)
	angles_deg = timeline["orientation_deg"].to_numpy(dtype=float)

	# Unwrap avoids incorrect interpolation across 0/360 crossings.
	angles_rad_unwrapped = np.unwrap(np.deg2rad(angles_deg))
	interp_rad = np.interp(
		t_target,
		t_src,
		angles_rad_unwrapped,
		left=angles_rad_unwrapped[0],
		right=angles_rad_unwrapped[-1],
	)

	return normalize_degrees(np.rad2deg(interp_rad))


def correct_wind_direction_for_file(
	path: Path,
	timeline: pd.DataFrame,
	assumed_orientation: float,
	in_place: bool,
	output_suffix: str,
) -> Path:
	df, file_info_row, _, units_row = read_eddypro_with_header(path)

	required_cols = ["date", "time", "wind_dir"]
	missing = [c for c in required_cols if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in {path.name}: {missing}")

	timestamp = pd.to_datetime(
		df["date"].astype(str).str.strip() + " " + df["time"].astype(str).str.strip(),
		errors="coerce",
	)
	if timestamp.isna().all():
		raise ValueError(f"Could not parse any timestamps in {path.name}")

	valid_mask = ~timestamp.isna()
	if not valid_mask.any():
		raise ValueError(f"No valid timestamps available in {path.name}")

	orientation_t = np.full(len(df), np.nan, dtype=float)
	orientation_t[valid_mask.to_numpy()] = interpolate_orientation(timestamp[valid_mask], timeline)

	wind_dir_raw = pd.to_numeric(df["wind_dir"], errors="coerce").to_numpy(dtype=float)
	delta = orientation_t - float(assumed_orientation)
	wind_dir_corrected = normalize_degrees(wind_dir_raw + delta)

	df["wind_dir"] = np.where(np.isfinite(wind_dir_raw), wind_dir_corrected, np.nan)

	if in_place:
		out_path = path
	else:
		out_path = path.with_name(f"{path.stem}{output_suffix}{path.suffix}")

	with out_path.open("w", encoding="utf-8", newline="") as out:
		writer = csv.writer(out)
		writer.writerow(file_info_row)
		writer.writerow(df.columns.tolist())
		writer.writerow(units_row)

		df.to_csv(out, index=False, header=False, na_rep="NaN", lineterminator="\n")

	return out_path


def main() -> int:
	args = parse_args()

	data_dir: Path = args.data_dir

	if not data_dir.exists():
		print(f"Data directory not found: {data_dir}", file=sys.stderr)
		return 1

	if args.orientation_file:
		single_timeline = load_orientation_timeline(args.orientation_file)
		timelines_by_height = {h: single_timeline for h in ("1m", "2m")}
		print(f"Using orientation points from {args.orientation_file} for all heights")
	else:
		timelines_by_height = None
		print("Using embedded orientation points by inferred site and sensor height")

	files = sorted(p for p in data_dir.glob(args.file_glob) if p.is_file())

	if not files:
		print(
			f"No files found in {data_dir} matching '{args.file_glob}'.",
			file=sys.stderr,
		)
		return 1

	print(f"Processing {len(files)} file(s) under {data_dir}")

	processed = 0
	failures = 0
	for path in files:
		try:
			site_key, height = infer_site_and_sensor_height(path)
			assumed_orientation = get_assumed_orientation(site_key, height, args.assumed_orientation)

			if timelines_by_height is None:
				site_timelines = get_embedded_timelines_for_site(site_key)
				timeline = site_timelines[height]
			else:
				timeline = timelines_by_height[height]

			out_path = correct_wind_direction_for_file(
				path=path,
				timeline=timeline,
				assumed_orientation=assumed_orientation,
				in_place=args.in_place,
				output_suffix=args.output_suffix,
			)
			processed += 1
			print(f"OK: {path} -> {out_path} (assumed orientation {assumed_orientation:.1f} deg)")
		except Exception as exc:  # pragma: no cover - best-effort batch processing
			failures += 1
			print(f"FAILED: {path}: {exc}", file=sys.stderr)

	print(f"Completed. Processed: {processed}, failed: {failures}")
	return 0 if failures == 0 else 2


if __name__ == "__main__":
	sys.exit(main())
