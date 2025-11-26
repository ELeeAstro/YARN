"""
plot_csv_only.py
================

Simple script to read and plot CSV observation files only.
Extracts first 4 columns: x (wavelength), xerr (bandwidth), y (transit depth), yerr (error)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
from pathlib import Path


def read_csv_file(filepath):
    """
    Read CSV file and extract first 4 data columns.

    Returns:
        dict with keys: 'x', 'xerr', 'y', 'yerr'
    """
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        x, xerr, y, yerr = [], [], [], []

        for row in reader:
            try:
                x.append(float(row['CENTRALWAVELNG']))
                xerr.append(float(row['BANDWIDTH']))
                y.append(float(row['PL_TRANDEP']))
                yerr.append(abs(float(row['PL_TRANDEPERR1'])))
            except (ValueError, KeyError):
                continue

    return {
        'x': np.array(x),
        'xerr': np.array(xerr),
        'y': np.array(y),
        'yerr': np.array(yerr)
    }


def load_all_csv(directory='./'):
    """Load all CSV files from directory."""
    data_dict = {}
    csv_files = list(Path(directory).glob('*.csv'))

    for filepath in csv_files:
        data = read_csv_file(filepath)
        if len(data['x']) > 0:
            data_dict[filepath.name] = data
            print(f"Loaded {filepath.name}: {len(data['x'])} points")

    return data_dict

def main():
    """Main function."""
    print("="*60)
    print("Loading CSV files...")
    print("="*60)

    # Load all CSV data
    data_dict = load_all_csv('./')

    if len(data_dict) == 0:
        print("No CSV files found!")
        return None

    print(f"\nTotal CSV files loaded: {len(data_dict)}")

    # Create plots
    # print("\n" + "="*60)
    # print("Creating plots...")
    # print("="*60)

    # plot_individual(data_dict)
    # plot_combined(data_dict)

    # print("\n" + "="*60)
    # print("Done!")
    # print("="*60)

    return data_dict


if __name__ == "__main__":
    # Run and return data
    all_data = main()

    # Extract data individually for each file
    print("\n" + "="*60)
    print("Extracting individual datasets...")
    print("="*60)

    # HST_1
    if 'HST_1.csv' in all_data:
        HST_1_wl = all_data['HST_1.csv']['x']
        HST_1_dwl = all_data['HST_1.csv']['xerr']/2.0
        HST_1_y = all_data['HST_1.csv']['y']/100.0
        HST_1_yerr = all_data['HST_1.csv']['yerr']/100.0
        print(f"HST_1: {len(HST_1_wl)} points")

    # HST_2
    if 'HST_2.csv' in all_data:
        HST_2_wl = all_data['HST_2.csv']['x']
        HST_2_dwl = all_data['HST_2.csv']['xerr']/2.0
        HST_2_y = all_data['HST_2.csv']['y']/100.0
        HST_2_yerr = all_data['HST_2.csv']['yerr']/100.0
        print(f"HST_2: {len(HST_2_wl)} points")

    # HST_3
    if 'HST_3.csv' in all_data:
        HST_3_wl = all_data['HST_3.csv']['x']
        HST_3_dwl = all_data['HST_3.csv']['xerr']/2.0
        HST_3_y = all_data['HST_3.csv']['y']/100.0
        HST_3_yerr = all_data['HST_3.csv']['yerr']/100.0
        print(f"HST_3: {len(HST_3_wl)} points")

    # HST_4
    if 'HST_4.csv' in all_data:
        HST_4_wl = all_data['HST_4.csv']['x']
        HST_4_dwl = all_data['HST_4.csv']['xerr']/2.0
        HST_4_y = all_data['HST_4.csv']['y']/100.0
        HST_4_yerr = all_data['HST_4.csv']['yerr']/100.0
        print(f"HST_4: {len(HST_4_wl)} points")

    # SOSS_1
    if 'SOSS_1.csv' in all_data:
        SOSS_1_wl = all_data['SOSS_1.csv']['x']
        SOSS_1_dwl = all_data['SOSS_1.csv']['xerr']/2.0
        SOSS_1_y = all_data['SOSS_1.csv']['y']/100.0 - 434e-6
        SOSS_1_yerr = all_data['SOSS_1.csv']['yerr']/100.0
        print(f"SOSS_1: {len(SOSS_1_wl)} points")

    # SOSS_2
    if 'SOSS_2.csv' in all_data:
        SOSS_2_wl = all_data['SOSS_2.csv']['x']
        SOSS_2_dwl = all_data['SOSS_2.csv']['xerr']/2.0
        SOSS_2_y = all_data['SOSS_2.csv']['y']/100.0 - 434e-6
        SOSS_2_yerr = all_data['SOSS_2.csv']['yerr']/100.0
        print(f"SOSS_2: {len(SOSS_2_wl)} points")

    # MIRI_1
    if 'MIRI_1.csv' in all_data:
        MIRI_1_wl = all_data['MIRI_1.csv']['x']
        MIRI_1_dwl = all_data['MIRI_1.csv']['xerr']/2.0
        MIRI_1_y = all_data['MIRI_1.csv']['y']/100.0 - 169e-6
        MIRI_1_yerr = all_data['MIRI_1.csv']['yerr']/100.0
        print(f"MIRI_1: {len(MIRI_1_wl)} points")

    S_1_wl = 3.6
    S_1_dwl = 0.38
    S_1_y = 1.5177/100.0
    S_1_yerr = 0.0123/100.0

    S_2_wl = 4.5
    S_2_dwl = 0.56
    S_2_y = 1.5679/100.0
    S_2_yerr = 0.0149/100.0

    print("\n" + "="*60)
    print("All data extracted and ready to use!")
    print("="*60)
    print("\nAvailable variables:")
    print("  HST_1_wl, HST_1_dwl, HST_1_y, HST_1_yerr")
    print("  HST_2_wl, HST_2_dwl, HST_2_y, HST_2_yerr")
    print("  HST_3_wl, HST_3_dwl, HST_3_y, HST_3_yerr")
    print("  HST_4_wl, HST_4_dwl, HST_4_y, HST_4_yerr")
    print("  SOSS_1_wl, SOSS_1_dwl, SOSS_1_y, SOSS_1_yerr")
    print("  SOSS_2_wl, SOSS_2_dwl, SOSS_2_y, SOSS_2_yerr")
    print("  MIRI_1_wl, MIRI_1_dwl, MIRI_1_y, MIRI_1_yerr")

    # Create combined plot
    print("\n" + "="*60)
    print("Creating combined plot...")
    print("="*60)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot each dataset individually
    if 'HST_1.csv' in all_data:
        ax.errorbar(HST_1_wl, HST_1_y, xerr=HST_1_dwl, yerr=HST_1_yerr,
                    fmt='o', markersize=5, capsize=2, label='HST_1', alpha=0.8)

    if 'HST_2.csv' in all_data:
        ax.errorbar(HST_2_wl, HST_2_y, xerr=HST_2_dwl, yerr=HST_2_yerr,
                    fmt='s', markersize=5, capsize=2, label='HST_2', alpha=0.8)

    if 'HST_3.csv' in all_data:
        ax.errorbar(HST_3_wl, HST_3_y, xerr=HST_3_dwl, yerr=HST_3_yerr,
                    fmt='^', markersize=5, capsize=2, label='HST_3', alpha=0.8)

    if 'HST_4.csv' in all_data:
        ax.errorbar(HST_4_wl, HST_4_y, xerr=HST_4_dwl, yerr=HST_4_yerr,
                    fmt='v', markersize=5, capsize=2, label='HST_4', alpha=0.8)

    if 'SOSS_1.csv' in all_data:
        ax.errorbar(SOSS_1_wl, SOSS_1_y, xerr=SOSS_1_dwl, yerr=SOSS_1_yerr,
                    fmt='D', markersize=5, capsize=2, label='SOSS_1', alpha=0.8)

    if 'SOSS_2.csv' in all_data:
        ax.errorbar(SOSS_2_wl, SOSS_2_y, xerr=SOSS_2_dwl, yerr=SOSS_2_yerr,
                    fmt='p', markersize=5, capsize=2, label='SOSS_2', alpha=0.8)

    ax.errorbar(S_1_wl, S_1_y, xerr=S_1_dwl, yerr=S_1_yerr, fmt='s', markersize=8, capsize=2, label='S_1', alpha=0.8)
    
    ax.errorbar(S_2_wl, S_2_y, xerr=S_2_dwl, yerr=S_2_yerr, fmt='s', markersize=8, capsize=2, label='S_2', alpha=0.8)    

    if 'MIRI_1.csv' in all_data:
        ax.errorbar(MIRI_1_wl, MIRI_1_y, xerr=MIRI_1_dwl, yerr=MIRI_1_yerr,
                    fmt='*', markersize=8, capsize=2, label='MIRI_1', alpha=0.8)



    ax.set_xlabel('Wavelength (μm)', fontsize=13)
    ax.set_ylabel('Transit Depth', fontsize=13)
    ax.set_title('Combined CSV Observations', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.set_xscale('log')

    plt.tight_layout()
    plt.show()

    # Save all data to a single .txt file
    print("\n" + "="*60)
    print("Saving combined data to file...")
    print("="*60)

    # Collect all data into a single list
    all_combined = []

    # Collect HST_1 data
    if 'HST_1.csv' in all_data:
        for i in range(len(HST_1_wl)):
            all_combined.append([HST_1_wl[i], HST_1_dwl[i], HST_1_y[i], HST_1_yerr[i]])

    # Collect HST_2 data
    if 'HST_2.csv' in all_data:
        for i in range(len(HST_2_wl)):
            all_combined.append([HST_2_wl[i], HST_2_dwl[i], HST_2_y[i], HST_2_yerr[i]])

    # Collect HST_3 data
    if 'HST_3.csv' in all_data:
        for i in range(len(HST_3_wl)):
            all_combined.append([HST_3_wl[i], HST_3_dwl[i], HST_3_y[i], HST_3_yerr[i]])

    # Collect HST_4 data
    if 'HST_4.csv' in all_data:
        for i in range(len(HST_4_wl)):
            all_combined.append([HST_4_wl[i], HST_4_dwl[i], HST_4_y[i], HST_4_yerr[i]])

    # Collect SOSS_1 data
    if 'SOSS_1.csv' in all_data:
        for i in range(len(SOSS_1_wl)):
            all_combined.append([SOSS_1_wl[i], SOSS_1_dwl[i], SOSS_1_y[i], SOSS_1_yerr[i]])

    # Collect SOSS_2 data
    if 'SOSS_2.csv' in all_data:
        for i in range(len(SOSS_2_wl)):
            all_combined.append([SOSS_2_wl[i], SOSS_2_dwl[i], SOSS_2_y[i], SOSS_2_yerr[i]])

    # Collect S_1 data (custom single point)
    all_combined.append([S_1_wl, S_1_dwl, S_1_y, S_1_yerr])

    # Collect S_2 data (custom single point)
    all_combined.append([S_2_wl, S_2_dwl, S_2_y, S_2_yerr])

    # Collect MIRI_1 data
    if 'MIRI_1.csv' in all_data:
        for i in range(len(MIRI_1_wl)):
            all_combined.append([MIRI_1_wl[i], MIRI_1_dwl[i], MIRI_1_y[i], MIRI_1_yerr[i]])

    # Sort by wavelength (first column)
    all_combined.sort(key=lambda x: x[0])

    # Write to file
    output_filename = 'combined_observations.txt'
    with open(output_filename, 'w') as f:
        f.write("# Combined observational data\n")
        f.write("# Format: wavelength(um)  delta_wavelength(um)  transit_depth  uncertainty  response_mode\n")
        f.write("# Data sorted by wavelength (lowest to highest)\n")
        f.write("# " + "="*80 + "\n")

        for row in all_combined:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.8f} {row[3]:.8f} boxcar\n")

    print(f"Saved: {output_filename} (sorted by wavelength)")

    # Count and display total points
    total_points = 0
    if 'HST_1.csv' in all_data: total_points += len(HST_1_wl)
    if 'HST_2.csv' in all_data: total_points += len(HST_2_wl)
    if 'HST_3.csv' in all_data: total_points += len(HST_3_wl)
    if 'HST_4.csv' in all_data: total_points += len(HST_4_wl)
    if 'SOSS_1.csv' in all_data: total_points += len(SOSS_1_wl)
    if 'SOSS_2.csv' in all_data: total_points += len(SOSS_2_wl)
    total_points += 2  # S_1 and S_2 custom points
    if 'MIRI_1.csv' in all_data: total_points += len(MIRI_1_wl)

    print(f"Total data points written: {total_points}")
    print(f"  - Including S_1 and S_2 custom points")
    print("="*60)
