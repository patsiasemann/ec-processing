"""Normalize TIMESTAMP values in .dat files to always include milliseconds.

Usage:
	python fix_timestamps.py [input_dir]

If no directory is provided, the default is the PEDDY input folder mentioned
in the request. Only lines that have a TIMESTAMP field are modified, and only
those timestamps that have no fractional seconds get ".0" appended. Existing
fractions are left unchanged.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_INPUT_DIR = Path(r"F:\Data\SILVEX-I\EC Data\Silvia3_Mitte\PEDDY\input")


def normalize_timestamp(value: str) -> str:
	"""Ensure a TIMESTAMP string contains fractional seconds.

	If the value has no decimal point, ".0" is appended. Existing fractional
	parts are left untouched.
	"""

	trimmed = value.strip()
	if not trimmed:
		return value

	if "." not in trimmed:
		# Preserve original surrounding whitespace, only augment the core value.
		prefix = value[: value.index(trimmed)] if value != trimmed else ""
		suffix = value[value.index(trimmed) + len(trimmed) :] if value != trimmed else ""
		return f"{prefix}{trimmed}.0{suffix}"

	return value


def iter_dat_files(base_dir: Path) -> Iterable[Path]:
	"""Yield all .dat files under the given directory (non-recursive)."""

	return sorted(p for p in base_dir.glob("*.dat") if p.is_file())


def detect_dialect(sample: str) -> csv.Dialect:
	"""Detect CSV dialect from a sample string, defaulting to comma."""

	try:
		return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", " "])
	except csv.Error:
		# Fall back to a simple comma-separated format.
		class SimpleDialect(csv.excel):
			delimiter = ","

		return SimpleDialect()


def process_file(path: Path) -> bool:
	"""Process a single .dat file in a streaming fashion.

	Returns True if the file was rewritten, False if no TIMESTAMP column exists.
	"""

	with path.open("r", encoding="utf-8", newline="") as src:
		sample = src.read(4096)
		src.seek(0)
		dialect = detect_dialect(sample)

		reader = csv.reader(src, dialect)
		try:
			header = next(reader)
		except StopIteration:
			return False

		try:
			ts_idx = header.index("TIMESTAMP")
		except ValueError:
			return False

		temp_path = path.with_suffix(path.suffix + ".tmp")
		with temp_path.open("w", encoding="utf-8", newline="") as dst:
			writer = csv.writer(dst, dialect)
			writer.writerow(header)

			for row in reader:
				if ts_idx < len(row):
					row[ts_idx] = normalize_timestamp(row[ts_idx])
				writer.writerow(row)

	temp_path.replace(path)
	return True


def main() -> int:
	input_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT_DIR
	if not input_dir.exists():
		print(f"Input directory not found: {input_dir}", file=sys.stderr)
		return 1

	dat_files = list(iter_dat_files(input_dir))
	if not dat_files:
		print(f"No .dat files found in {input_dir}")
		return 0

	processed = 0
	skipped = 0
	for file_path in dat_files:
		try:
			changed = process_file(file_path)
			if changed:
				processed += 1
			else:
				skipped += 1
		except Exception as exc:  # pragma: no cover - best-effort logging
			print(f"Failed on {file_path}: {exc}", file=sys.stderr)

	print(f"Processed {processed} file(s); skipped {skipped} (no TIMESTAMP).")
	return 0


if __name__ == "__main__":
	sys.exit(main())
