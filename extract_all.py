"""
Walk data/{area}/{date}/{measurement_type}/ and convert every EDX/EDT pair
into a row-per-cell DataFrame, then save one Parquet file per scenario:

    output/{area}_{date}_{measurement_type}.parquet

Each file contains all timesteps stacked, with columns:
    area, date, measurement_type, time, hour, i, j, k, x_m, y_m, z_m, <variables...>

`hour` (int8) is derived from the EDX filename timestamp so filtering by hour
is instant without any datetime parsing at load time.

Reuses the parsing functions from extract_edt.py.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ── import helpers from existing extractor ────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from extract_edt import (
    read_text_file,
    read_tag,
    read_integer_tag,
    read_number_list,
    expand_spacing,
    cell_centres,
    find_matching_edt,
    parse_timestamp,
    NODATA_VALUE,
)

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR   = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "output"

# Vertical grid level to extract (0 = ground level)
Z_LEVEL = 0


# ── Core parser ───────────────────────────────────────────────────────────────

def parse_timestep(edx_path: Path, z_level: int = Z_LEVEL) -> pd.DataFrame:
    """Parse one EDX/EDT pair and return a tidy DataFrame (no file I/O)."""
    edx_text = read_text_file(edx_path)

    nx = read_integer_tag(edx_text, "nr_xdata")
    ny = read_integer_tag(edx_text, "nr_ydata")
    nz = read_integer_tag(edx_text, "nr_zdata")
    n_vars       = read_integer_tag(edx_text, "nr_variables")
    data_per_var = read_integer_tag(edx_text, "Data_per_variable")

    variable_names = [
        n.strip()
        for n in read_tag(edx_text, "name_variables").split(",")
        if n.strip()
    ]

    spacing_x = expand_spacing(read_number_list(read_tag(edx_text, "spacing_x")), nx)
    spacing_y = expand_spacing(read_number_list(read_tag(edx_text, "spacing_y")), ny)
    spacing_z = expand_spacing(read_number_list(read_tag(edx_text, "spacing_z")), nz)

    x_coords = cell_centres(spacing_x)
    y_coords = cell_centres(spacing_y)
    z_coords = cell_centres(spacing_z)

    # Pass edx_path so parse_timestamp can read the hour from the filename
    timestamp = parse_timestamp(edx_text, edx_path=edx_path)
    hour = int(timestamp.hour) if not pd.isnull(timestamp) else -1

    edt_path   = find_matching_edt(edx_path)
    raw_values = np.fromfile(edt_path, dtype="<f4")

    expected = n_vars * nx * ny * nz * data_per_var
    if raw_values.size != expected:
        raise ValueError(
            f"Size mismatch in {edt_path.name}: "
            f"expected {expected:,}, got {raw_values.size:,}"
        )

    data     = raw_values.reshape(n_vars, nz, ny, nx, data_per_var)
    selected = data[:, z_level, :, :, :]  # shape: (n_vars, ny, nx, data_per_var)

    values = (
        selected.transpose(1, 2, 3, 0)          # (ny, nx, data_per_var, n_vars)
                .reshape(ny * nx * data_per_var, n_vars)
                .astype(float)
    )
    values[np.isclose(values, NODATA_VALUE)] = np.nan

    jj, ii, component = np.meshgrid(
        np.arange(ny), np.arange(nx), np.arange(data_per_var), indexing="ij"
    )

    df = pd.DataFrame({
        "time": timestamp,
        "hour": np.int8(hour),
        "i":    ii.ravel().astype(np.int16),
        "j":    jj.ravel().astype(np.int16),
        "k":    np.int8(z_level),
        "x_m":  x_coords[ii.ravel()].astype(np.float32),
        "y_m":  y_coords[jj.ravel()].astype(np.float32),
        "z_m":  np.float32(z_coords[z_level]),
    })

    if data_per_var > 1:
        df["component"] = component.ravel().astype(np.int8)

    for col_idx, var_name in enumerate(variable_names):
        df[var_name] = values[:, col_idx].astype(np.float32)

    return df


# ── Scenario processor ────────────────────────────────────────────────────────

def process_scenario(area: str, date: str, mtype: str) -> None:
    src_dir = BASE_DIR / area / date / mtype
    if not src_dir.exists():
        print(f"  [SKIP] folder not found: {src_dir}")
        return

    edx_files = sorted(src_dir.glob("*.EDX")) + sorted(src_dir.glob("*.edx"))
    if not edx_files:
        print(f"  [SKIP] no EDX files in {src_dir}")
        return

    print(f"  {len(edx_files)} timestep(s) found")

    frames = []
    failed = 0
    for edx_path in edx_files:
        try:
            df = parse_timestep(edx_path)
            frames.append(df)
            print(f"    OK  {edx_path.name}", end="\r")
        except Exception as e:
            failed += 1
            print(f"    ERR {edx_path.name}: {e}")

    print()
    if not frames:
        print("  [SKIP] all timesteps failed")
        return

    combined = pd.concat(frames, ignore_index=True)

    # Prepend metadata columns
    combined.insert(0, "measurement_type", mtype)
    combined.insert(0, "date",             date)
    combined.insert(0, "area",             area)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{area}_{date}_{mtype}.parquet"
    combined.to_parquet(out_path, index=False, compression="snappy")

    size_mb = out_path.stat().st_size / 1_048_576
    print(f"  Saved → {out_path.name}  ({len(combined):,} rows, {size_mb:.1f} MB, {failed} failed)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    scenarios = []
    if BASE_DIR.exists():
        for area_dir in sorted(BASE_DIR.iterdir()):
            if not area_dir.is_dir():
                continue
            for date_dir in sorted(area_dir.iterdir()):
                if not date_dir.is_dir():
                    continue
                for type_dir in sorted(date_dir.iterdir()):
                    if not type_dir.is_dir():
                        continue
                    scenarios.append((area_dir.name, date_dir.name, type_dir.name))

    if not scenarios:
        print(f"No scenario folders found under {BASE_DIR}")
        return

    print(f"Found {len(scenarios)} scenario(s) under {BASE_DIR}\n")

    for area, date, mtype in scenarios:
        print(f"[{area} / {date} / {mtype}]")
        process_scenario(area, date, mtype)
        print()

    print("All done.")
    print(f"Parquet files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
