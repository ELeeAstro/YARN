#!/usr/bin/env python3
"""
tag_obs_response.py
====================

Overview:
    Annotate an observational dataset with custom response flags that
    downstream averaging routines can use to pick non-uniform kernels.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


DEFAULT_MODE = "boxcar"


def _parse_modes(modes: List[str], nrows: int) -> List[str]:
    if not modes:
        return [DEFAULT_MODE] * nrows
    if len(modes) == 1:
        mode = modes[0]
        if mode == "auto":
            return [DEFAULT_MODE] * nrows
        return [mode] * nrows
    if len(modes) != nrows:
        raise ValueError(
            f"Provided {len(modes)} response modes for {nrows} rows; lengths must match."
        )
    return modes


def _detect_delimiter(path: Path) -> Tuple[str, bool]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if "," in stripped:
                return ",", True
            return None, False  # use whitespace split
    return None, False


def _read_delimited(path: Path, delimiter: str | None) -> List[List[str]]:
    with path.open("r", encoding="utf-8") as handle:
        if delimiter is None:
            return [
                [cell for cell in line.split() if cell]
                for line in handle
                if line.strip()
            ]
        reader = csv.reader(handle, delimiter=delimiter)
        return [row for row in reader if row]


def _write_delimited(path: Path, rows: List[List[str]], delimiter: str | None) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        if delimiter is None:
            for row in rows:
                handle.write(" ".join(row) + "\n")
        else:
            writer = csv.writer(handle, delimiter=delimiter)
            writer.writerows(rows)


def annotate_observations(
    input_path: Path, output_path: Path, modes: List[str], delimiter: str | None
) -> None:
    lines = _read_delimited(input_path, delimiter)

    if not lines:
        raise ValueError(f"No data found in {input_path}")

    header = lines[0]
    has_header = any(cell.lower().startswith("lam") for cell in header)
    data_rows = lines[1:] if has_header else lines

    mode_flags = _parse_modes(modes, len(data_rows))

    annotated = []
    if has_header:
        annotated.append(header + ["response_mode"])
    for row, mode in zip(data_rows, mode_flags):
        annotated.append(row + [mode])

    _write_delimited(output_path, annotated, delimiter)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach response_mode flags to observational data rows."
    )
    parser.add_argument("input", type=Path, help="CSV file with columns lam,dlam,depth...")
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination CSV (default: <input> with '_tagged' suffix).",
    )
    parser.add_argument(
        "--mode",
        nargs="*",
        default=[],
        help=(
            "Response modes to assign. Provide one per row, or a single mode to broadcast. "
            "Use 'auto' to keep the default (boxcar) for all rows."
        ),
    )
    args = parser.parse_args()

    output_path = (
        args.output
        if args.output is not None
        else args.input.with_name(args.input.stem + "_tagged" + args.input.suffix)
    )

    delimiter, detected = _detect_delimiter(args.input)
    annotate_observations(args.input, output_path, args.mode, delimiter)
    print(f"[tag_obs_response] wrote {output_path}")


if __name__ == "__main__":
    main()
