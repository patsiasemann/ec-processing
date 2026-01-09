from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


@dataclass(frozen=True)
class SplitSpec:
    """Defines how a sensor level should be extracted from a source file."""

    label: str
    suffix: str
    column_map: Dict[str, str]
    required: set[str] = field(default_factory=set)

    @property
    def source_columns(self) -> List[str]:
        return list(self.column_map.keys())


SPLIT_SPECS: List[SplitSpec] = [
    SplitSpec(
        label="1m",
        suffix="_1m",
        column_map={
            "TIMESTAMP": "TIMESTAMP",
            "Ux_CSAT3": "Ux",
            "Uy_CSAT3": "Uy",
            "Uz_CSAT3": "Uz",
            "Ts_CSAT3": "Ts",
            "Diag_CSAT3": "diag_sonic",
        },
        required={"TIMESTAMP", "Ux_CSAT3", "Uy_CSAT3", "Uz_CSAT3"},
    ),
    SplitSpec(
        label="2m",
        suffix="_2m",
        column_map={
            "TIMESTAMP": "TIMESTAMP",
            "Ux_lower_CSAT3B": "Ux",
            "Uy_lower_CSAT3B": "Uy",
            "Uz_lower_CSAT3B": "Uz",
            "Ts_lower_CSAT3B": "Ts",
            "Diag_lower_CSAT3B": "diag_sonic",
        },
        required={"TIMESTAMP", "Ux_lower_CSAT3B", "Uy_lower_CSAT3B", "Uz_lower_CSAT3B"},
    ),
    SplitSpec(
        label="3m",
        suffix="_3m",
        column_map={
            "TIMESTAMP": "TIMESTAMP",
            "Ux_upper_CSAT3B": "Ux",
            "Uy_upper_CSAT3B": "Uy",
            "Uz_upper_CSAT3B": "Uz",
            "Ts_upper_CSAT3B": "Ts",
            "Diag_upper_CSAT3B": "diag_sonic",
        },
        required={"TIMESTAMP", "Ux_upper_CSAT3B", "Uy_upper_CSAT3B", "Uz_upper_CSAT3B"},
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split multi-level EC files into sensor-specific .dat files.")
    parser.add_argument(
        "--input-dir",
        default="D:\\SILVEX II 2025\\EC data\\Silvia 1 (unten)\\converted\\",
        help="Directory containing the converted .dat files.",
    )
    parser.add_argument(
        "--output-dir",
        default="D:\\SILVEX II 2025\\EC data\\Silvia 1 (unten)\\PEDDY\\input\\",
        help="Directory to write the split .dat files to.",
    )
    parser.add_argument(
        "--pattern",
        default="SILVEXII_Silvia1_sonics_*.dat",
        help="Glob pattern used to discover source files inside --input-dir.",
    )
    parser.add_argument("--delimiter", default=",", help="Field delimiter used in the source files (default: ,).")
    parser.add_argument("--encoding", default="utf-8", help="File encoding used for reading (default: utf-8).")
    return parser.parse_args()


def available_columns(columns: Iterable[str], df_columns: Iterable[str]) -> List[str]:
    df_cols = set(df_columns)
    return [column for column in columns if column in df_cols]


def process_file(
    file_path: Path,
    specs: List[SplitSpec],
    out_dir: Path,
    delimiter: str,
    encoding: str,
) -> None:
    print(f"\n---------- Processing file: {file_path.name} ----------")
    df = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=1,  # actual column names live on the second line of TOA5 exports
        skiprows=[2, 3],  # drop units and processing rows
        low_memory=False,
    )

    for spec in specs:
        missing_required = spec.required - set(df.columns)
        if missing_required:
            print(f"Skipping {spec.label}: missing required columns {sorted(missing_required)}")
            continue

        columns_present = available_columns(spec.source_columns, df.columns)
        if not columns_present:
            print(f"Skipping {spec.label}: none of the configured columns were found.")
            continue

        missing_optional = [col for col in spec.source_columns if col not in columns_present and col not in spec.required]
        if missing_optional:
            print(f"{spec.label}: dropping optional columns {missing_optional} (not in file)")

        rename_map = {col: spec.column_map[col] for col in columns_present}
        df_subset = df[columns_present].rename(columns=rename_map)

        output_path = out_dir / f"{file_path.stem}{spec.suffix}.dat"
        df_subset.to_csv(output_path, index=False, na_rep="NaN")
        print(f"Saved {spec.label} subset to {output_path}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob(args.pattern))
    if not files:
        print("No files matched the provided pattern. Nothing to do.")
        return

    for file_path in files:
        process_file(file_path, SPLIT_SPECS, output_dir, delimiter=args.delimiter, encoding=args.encoding)


if __name__ == "__main__":
    main()
    