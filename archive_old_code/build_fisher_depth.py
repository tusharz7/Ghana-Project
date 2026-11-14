import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
#  LOAD DEPTH GRID
# ─────────────────────────────────────────────

def loadDepthGrid(cross_ref_folder: Path):
    """
    Load the ocean depth grid from binary files.

    Files:
      XC.data  — longitude for each grid cell  (385 x 1376, float32 big-endian)
      Depth.data — seafloor depth in metres     (385 x 1376, float32 big-endian)

    Latitude is reconstructed from the grid spacing (0.015625 deg/cell)
    anchored at the known Ghana coastline row.
    """
    xc    = np.fromfile(cross_ref_folder / "XC.data",    dtype=">f4").reshape(385, 1376)
    depth = np.fromfile(cross_ref_folder / "Depth.data", dtype=">f4").reshape(385, 1376)

    # Longitude: uniform per row, taken from first row of XC
    lons = xc[0]

    # Latitude: reconstruct from grid spacing
    # Grid step = 0.015625 deg; land (depth=0) starts at row ~239 which ~ 5.0N (Ghana coast)
    lat_step  = 0.015625
    lat_start = 5.0 - 239 * lat_step   # ~1.266 N
    lats      = lat_start + np.arange(385) * lat_step

    print(f"Depth grid loaded: {depth.shape} cells")
    print(f"  Lat range : {lats.min():.3f}N  to  {lats.max():.3f}N")
    print(f"  Lon range : {lons.min():.3f}E  to  {lons.max():.3f}E")
    print(f"  Depth range: {depth.min():.0f}m  to  {depth.max():.0f}m  (0 = land)")

    return lats, lons, depth


def lookupDepth(lat: float, lon: float,
                lats: np.ndarray, lons: np.ndarray,
                depth_grid: np.ndarray) -> float:
    """
    Find the seafloor depth (metres) at a given lat/lon by finding
    the nearest grid cell.
    Returns NaN if the point is outside the grid bounds.
    """
    if (lat < lats.min() or lat > lats.max() or
            lon < lons.min() or lon > lons.max()):
        return np.nan

    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    return float(depth_grid[i_lat, i_lon])


# ─────────────────────────────────────────────
#  LOAD FISHER DATA
# ─────────────────────────────────────────────

def loadFisherFile(file_path: Path, file_id: int) -> Optional[pd.DataFrame]:
    """Load a single fisher Excel file and return cleaned DataFrame."""
    try:
        df = pd.read_excel(file_path, header=4)
        df.columns = ["events", "timestamp", "address",
                      "lat_lng", "speed", "heading",
                      "altitude", "via", "drivers"]

        df = df[df["timestamp"] != "Date"].copy()
        df = df.dropna(subset=["lat_lng", "speed"])

        lat_lon  = df["lat_lng"].str.split(",", expand=True)
        df["lat"] = pd.to_numeric(lat_lon[0].str.strip(), errors="coerce")
        df["lon"] = pd.to_numeric(lat_lon[1].str.strip(), errors="coerce")
        df = df.dropna(subset=["lat", "lon"])

        if len(df) < 100: # need at least 100 rows
            return None

        df["speed"]     = pd.to_numeric(df["speed"],   errors="coerce").fillna(0)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["file_id"]   = file_id
        df["file_name"] = file_path.stem

        return df[["file_id", "file_name", "timestamp", "lat", "lon", "speed", "heading"]]

    except Exception as e:
        return None


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    fisher_folder    = Path("/Users/tushy/Documents/Research/Ghana Project/Exported Files")
    cross_ref_folder = Path("/Users/tushy/Documents/Research/Ghana Project/cross_reference")
    output_folder    = Path("/Users/tushy/Documents/Research/Ghana Project/fisherDepth")
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1. Load depth grid
    lats, lons, depth_grid = loadDepthGrid(cross_ref_folder)

    # 2. Process each fisher file individually
    file_paths = sorted(fisher_folder.glob("*.xlsx"))
    print(f"\nFound {len(file_paths)} Excel files to process.\n")

    for file_id, file_path in enumerate(file_paths, start=1):
        df = loadFisherFile(file_path, file_id)
        if df is None:
            continue

        # 3. Look up depth for every GPS timepoint
        df["depth_m"] = df.apply(
            lambda row: lookupDepth(row["lat"], row["lon"], lats, lons, depth_grid),
            axis=1
        )

        # 4. Flag points that landed on land (depth = 0) or outside grid (NaN)
        df["on_land"]       = df["depth_m"] == 0.0
        df["outside_grid"]  = df["depth_m"].isna()

        # 5. Save to fisherDepth folder — one CSV per fisher
        out_path = output_folder / f"{file_path.stem}_depth.csv"
        df.to_csv(out_path, index=False)

        n_valid   = (~df["on_land"] & ~df["outside_grid"]).sum()
        n_land    = df["on_land"].sum()
        n_outside = df["outside_grid"].sum()
        print(f"  {file_path.name}")
        print(f"    Depth range: {df['depth_m'].min():.0f}m – {df['depth_m'].max():.0f}m")

    print(f"Done.")


if __name__ == "__main__":
    main()
