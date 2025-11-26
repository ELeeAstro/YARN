#!/usr/bin/env python3
"""
convert_cia_to_npz.py
=====================

Overview:
    Convert a legacy HITRAN-style CIA text table into a compressed NumPy
    archive (.npz) containing the original temperature grid, wavenumber grid,
    and opacity values without forcing the data onto a shared grid.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np


def _infer_species_and_suffix(path: Path) -> Tuple[str, str]:
    stem = path.stem
    if "_" in stem:
        parts = stem.split("_", 1)
        return parts[0], parts[1]
    return stem, "orig"


def _read_cia_block(stream) -> Tuple[np.ndarray, float]:
    header = stream.readline()
    if not header:
        return np.array([]), np.nan
    parts = header.split()
    if len(parts) < 5:
        raise ValueError(f"Malformed CIA header line: {header.strip()}")
    nrec = int(parts[3])
    temperature = float(parts[4])
    wn = np.empty(nrec, dtype=float)
    sigma = np.empty(nrec, dtype=float)
    for idx in range(nrec):
        line = stream.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading CIA block.")
        val = line.split()
        if len(val) < 2:
            raise ValueError(f"Malformed CIA data line: {line.strip()}")
        wn[idx] = float(val[0])
        sigma[idx] = float(val[1])
    return wn, temperature, sigma


def _load_cia_text(path: Path) -> Tuple[str, str, np.ndarray, np.ndarray, np.ndarray]:
    species, suffix = _infer_species_and_suffix(path)
    temperatures = []
    sigma_rows = []
    wn_master = None

    with path.open("r", encoding="utf-8") as handle:
        while True:
            pos = handle.tell()
            header = handle.readline()
            if not header:
                break
            if not header.strip():
                continue
            handle.seek(pos)
            wn, temperature, sigma = _read_cia_block(handle)
            if wn.size == 0:
                break
            if wn_master is None:
                wn_master = wn
            elif wn_master.shape != wn.shape or not np.allclose(wn_master, wn):
                raise ValueError(
                    "Encountered a temperature slice with a different "
                    "wavenumber grid; rebinning is required."
                )
            temperatures.append(temperature)
            sigma_rows.append(sigma)

    if wn_master is None or not sigma_rows:
        raise ValueError(f"No CIA records found in {path}")

    sigma_array = np.vstack(sigma_rows)
    temp_array = np.asarray(temperatures, dtype=float)
    return species, suffix, temp_array, wn_master, sigma_array


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a CIA text table into a compressed NPZ archive."
    )
    parser.add_argument("input", type=Path, help="Path to the .cia text file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (.npz). Defaults to <species>_orig.npz in the same directory.",
    )
    args = parser.parse_args()

    species, suffix, temps, wavenumbers, sigma = _load_cia_text(args.input)
    output = (
        args.output
        if args.output is not None
        else args.input.with_name(f"{species}_{suffix}.npz")
    )

    sigma = np.maximum(sigma,1e-199)

    sigma = np.log10(sigma)

    np.savez_compressed(
        output,
        mol=species,
        T=temps,
        wn=wavenumbers,
        sig=sigma,
    )
    print(f"[convert_cia_to_npz] saved {output}")


if __name__ == "__main__":
    main()
