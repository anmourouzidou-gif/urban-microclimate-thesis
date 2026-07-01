from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

# Folder containing the ENVI-met Atmosphere EDX/EDT files
INPUT_FOLDER = Path(__file__).parent

# Folder where the CSV files will be written
OUTPUT_FOLDER = Path(__file__).parent / "output"

# Vertical grid level.
# Leonardo showed "XY-Cut at k=0", so start with 0.
Z_LEVEL = 0

# ENVI-met missing-data value
NODATA_VALUE = -999.0


# ============================================================
# FUNCTIONS
# ============================================================

def read_text_file(path: Path) -> str:
    """Read an EDX file using common Windows text encodings."""
    raw = path.read_bytes()

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue

    return raw.decode("utf-8", errors="replace")


def read_tag(text: str, tag_name: str, required: bool = True):
    """Extract a value from an XML-style EDX tag."""
    pattern = (
        rf"<\s*{re.escape(tag_name)}\s*>"
        rf"(.*?)"
        rf"<\s*/\s*{re.escape(tag_name)}\s*>"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        return match.group(1).strip()

    if required:
        raise ValueError(f"Could not find EDX tag: {tag_name}")

    return None


def read_integer_tag(text: str, tag_name: str) -> int:
    value = read_tag(text, tag_name)

    match = re.search(r"-?\d+", value)

    if not match:
        raise ValueError(
            f"Could not interpret {tag_name} as an integer: {value}"
        )

    return int(match.group())


def read_number_list(value: str) -> np.ndarray:
    """Extract all numeric values from an EDX field."""
    numbers = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][-+]?\d+)?",
        value,
    )

    return np.array(
        [float(number) for number in numbers],
        dtype=float,
    )


def expand_spacing(values: np.ndarray, number_of_cells: int) -> np.ndarray:
    """
    ENVI-met may store one constant spacing value or one value per cell.
    """
    if len(values) == number_of_cells:
        return values

    if len(values) == 1:
        return np.repeat(values[0], number_of_cells)

    raise ValueError(
        f"Expected 1 or {number_of_cells} spacing values, "
        f"but found {len(values)}."
    )


def cell_centres(spacing: np.ndarray) -> np.ndarray:
    """Calculate the physical centre coordinate of every grid cell."""
    return np.cumsum(spacing) - spacing / 2


def find_matching_edt(edx_path: Path) -> Path:
    """Find the EDT file corresponding to an EDX file."""
    possible_paths = [
        edx_path.with_suffix(".EDT"),
        edx_path.with_suffix(".edt"),
        edx_path.with_suffix(".Edt"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"No matching EDT file was found for:\n{edx_path}"
    )


def parse_timestamp(edx_text: str, edx_path=None):
    """Parse simulation timestamp.

    Tries the EDX filename first (most reliable — format *_YYYY-MM-DD_HH.MM.SS.EDX),
    then falls back to the <simulation_date> / <simulation_time> tags inside the file.
    ENVI-met uses dots as separators in the time tag (e.g. '11.00.01'), which is
    handled by replacing them with colons before parsing.
    """
    import re

    # ── 1. try filename ───────────────────────────────────────────────────────
    if edx_path is not None:
        m = re.search(
            r"(\d{4}-\d{2}-\d{2})_(\d{2})\.(\d{2})\.(\d{2})",
            Path(edx_path).name,
        )
        if m:
            date_str, hh, mm, ss = m.groups()
            return pd.Timestamp(f"{date_str} {hh}:{mm}:{ss}")

    # ── 2. fall back to EDX tags ──────────────────────────────────────────────
    date_value = read_tag(edx_text, "simulation_date", required=False)
    time_value = read_tag(edx_text, "simulation_time", required=False)

    if not date_value:
        return pd.NaT

    if not time_value:
        time_value = "00:00:00"

    # ENVI-met stores time as "HH.MM.SS" — convert dots to colons
    time_value = time_value.replace(".", ":")

    return pd.to_datetime(f"{date_value} {time_value}", dayfirst=True, errors="coerce")


def export_one_timestep(edx_path: Path):
    """Decode one matching EDX/EDT pair and write one CSV."""

    print(f"Reading: {edx_path.name}")

    edx_text = read_text_file(edx_path)

    # Grid dimensions
    nx = read_integer_tag(edx_text, "nr_xdata")
    ny = read_integer_tag(edx_text, "nr_ydata")
    nz = read_integer_tag(edx_text, "nr_zdata")

    # Number of variables
    number_of_variables = read_integer_tag(
        edx_text,
        "nr_variables",
    )

    # Usually 1 for ordinary atmosphere rasters
    data_per_variable = read_integer_tag(
        edx_text,
        "Data_per_variable",
    )

    # Variable names
    variable_names_text = read_tag(
        edx_text,
        "name_variables",
    )

    variable_names = [
        name.strip()
        for name in variable_names_text.split(",")
        if name.strip()
    ]

    if len(variable_names) != number_of_variables:
        raise ValueError(
            f"EDX reports {number_of_variables} variables, "
            f"but {len(variable_names)} names were found."
        )

    # Grid spacing
    spacing_x = expand_spacing(
        read_number_list(read_tag(edx_text, "spacing_x")),
        nx,
    )

    spacing_y = expand_spacing(
        read_number_list(read_tag(edx_text, "spacing_y")),
        ny,
    )

    spacing_z = expand_spacing(
        read_number_list(read_tag(edx_text, "spacing_z")),
        nz,
    )

    if Z_LEVEL < 0 or Z_LEVEL >= nz:
        raise ValueError(
            f"Requested Z_LEVEL={Z_LEVEL}, but available levels "
            f"are 0 to {nz - 1}."
        )

    x_coordinates = cell_centres(spacing_x)
    y_coordinates = cell_centres(spacing_y)
    z_coordinates = cell_centres(spacing_z)

    timestamp = parse_timestamp(edx_text, edx_path=edx_path)

    edt_path = find_matching_edt(edx_path)

    # ENVI-met EDT values are normally 32-bit floating point
    raw_values = np.fromfile(
        edt_path,
        dtype="<f4",
    )

    cells_per_variable = nx * ny * nz * data_per_variable

    expected_values = (
        number_of_variables * cells_per_variable
    )

    if raw_values.size != expected_values:
        raise ValueError(
            f"Unexpected EDT file size for {edt_path.name}\n"
            f"Expected {expected_values:,} float values, "
            f"but found {raw_values.size:,}.\n"
            f"This file may use a different ENVI-met output layout."
        )

    # ENVI-met storage order:
    # variable, z, y, x, component
    data = raw_values.reshape(
        number_of_variables,
        nz,
        ny,
        nx,
        data_per_variable,
    )

    # Extract only the selected horizontal level
    selected = data[:, Z_LEVEL, :, :, :]

    # Rearrange rows as grid cells and columns as variables
    values = selected.transpose(
        1,
        2,
        3,
        0,
    ).reshape(
        ny * nx * data_per_variable,
        number_of_variables,
    )

    values = values.astype(float)

    # Replace ENVI-met NoData values with empty values
    values[np.isclose(values, NODATA_VALUE)] = np.nan

    # Build grid indices
    jj, ii, component = np.meshgrid(
        np.arange(ny),
        np.arange(nx),
        np.arange(data_per_variable),
        indexing="ij",
    )

    ii = ii.ravel()
    jj = jj.ravel()
    component = component.ravel()

    output = pd.DataFrame(
        {
            "time": timestamp,
            "i": ii,
            "j": jj,
            "k": Z_LEVEL,
            "x_m": x_coordinates[ii],
            "y_m": y_coordinates[jj],
            "z_m": z_coordinates[Z_LEVEL],
        }
    )

    if data_per_variable > 1:
        output["component"] = component

    for column_number, variable_name in enumerate(variable_names):
        output[variable_name] = values[:, column_number]

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_FOLDER / f"{edx_path.stem}.csv"

    output.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        f"Saved: {output_path.name} "
        f"({len(output):,} rows, {len(output.columns):,} columns)"
    )


def main():
    if not INPUT_FOLDER.exists():
        raise FileNotFoundError(
            f"The input folder does not exist:\n{INPUT_FOLDER}"
        )

    edx_files = sorted(
        file
        for file in INPUT_FOLDER.iterdir()
        if file.is_file()
        and file.suffix.lower() == ".edx"
    )

    if not edx_files:
        raise FileNotFoundError(
            f"No EDX files were found in:\n{INPUT_FOLDER}"
        )

    print(f"Input folder: {INPUT_FOLDER}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    print(f"EDX files found: {len(edx_files)}")
    print(f"Selected vertical level: k={Z_LEVEL}")
    print()

    successful = 0
    failed = 0

    for edx_path in edx_files:
        try:
            export_one_timestep(edx_path)
            successful += 1

        except Exception as error:
            failed += 1
            print()
            print(f"ERROR while reading {edx_path.name}")
            print(error)
            print()

    print()
    print("Export finished.")
    print(f"Successful files: {successful}")
    print(f"Failed files: {failed}")
    print(f"Results folder: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()