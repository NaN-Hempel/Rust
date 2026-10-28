## Quick script to generate a dataframe of the solar system with planets and their moons, including their positions and velocities. 
# This script uses the `pandas` library to create a structured representation of the solar system data.

import pandas as pd
from pathlib import Path

def project_dir(start: Path | None = None) -> Path:
    """Walk up from this file until a directory containing Cargo.toml or .git."""
    here = (start or Path(__file__)).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "Cargo.toml").exists() or (candidate / ".git").exists():
            return candidate
    return here.parent   # nothing found; fall back to the file's directory


BASE = project_dir()
DATA_DIR = BASE / "src" / "data" / "solar.csv"
valsDIR = BASE / "src" / "data" / "init_vals/"
HEADERS = [
    "Name", "Type", "Parent", "Class",
    "Mean radius (km)", "Mass (Earth = 1)", "Mean density (g/cm3)",
    "Surface gravity (m/s2)", "Escape velocity (km/s)",
    "Semi-major axis (AU)", "Semi-major axis (10^3 km)", "Semi-major axis (primary radii)",
    "Orbital velocity (km/s)", "Orbital eccentricity",
    "Orbital inclination (deg)", "Rotation period (hours)",
    "Axial tilt (deg)", "Mean surface temp (C)", "Colour", "Rings",
]



def generate_solar_system_dataframe(path: str | Path | None = None) -> pd.DataFrame:
    """Load the solar system CSV into a DataFrame.

    Default location is src/data/solar_system_objects.csv, found relative to
    the project root (the directory holding Cargo.toml) rather than to this
    file, so the path holds wherever the script is invoked from.

    `path` overrides it.
    """
    csv_file_path = Path(path) if path else DATA_DIR / ""

    return pd.read_csv(csv_file_path)

def generate_csv_file(path, df):
    vals = df[HEADERS]
    vals = vals.fillna(0)
    vals.to_csv(f"{path}/init_values.csv", index=False)
    return vals

def main():
    
    dataframe = generate_solar_system_dataframe()
    test(dataframe)
    dataframe = generate_csv_file(valsDIR, dataframe)
    test(dataframe)
    

def test(df):
    # Generate the solar system dataframe
    # Display the dataframe
    print("\n")
    print(df, "\n\n")


if __name__ == "__main__":
    main()