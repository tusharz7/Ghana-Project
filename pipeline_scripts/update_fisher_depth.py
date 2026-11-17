"""Redo depth_m and on_land in every fisherDepth/*_depth.csv with the corrected depth grid.

Positions, times and all other columns are left as they are. The grid latitudes now come from
data/YC.data (the old build rebuilt them and was ~29 km too far north).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "archive_old_code"))
from build_fisher_depth import loadDepthGrid  # noqa: E402

FISHER_DEPTH_FOLDER = PROJECT_ROOT / "fisherDepth"


def nearest_depth(lat, lon, lats, lons, grid):
    """Depth (m) of the nearest grid cell; NaN outside the grid (same rule as lookupDepth)."""
    i = np.abs(lats[None, :] - lat[:, None]).argmin(axis=1)
    j = np.abs(lons[None, :] - lon[:, None]).argmin(axis=1)
    depth = grid[i, j].astype(float)
    outside = (lat < lats.min()) | (lat > lats.max()) | (lon < lons.min()) | (lon > lons.max())
    depth[outside] = np.nan
    return depth


def main() -> None:
    lats, lons, grid = loadDepthGrid(PROJECT_ROOT / "data")
    total = changed = land_before = land_after = 0
    for path in sorted(FISHER_DEPTH_FOLDER.glob("*_depth.csv")):
        df = pd.read_csv(path, dtype={"heading": "object"}, keep_default_na=False, na_values=[""])
        old = df["depth_m"].to_numpy()
        new = nearest_depth(df["lat"].to_numpy(float), df["lon"].to_numpy(float), lats, lons, grid)
        land_before += int(df["on_land"].astype(str).eq("True").sum())
        df["depth_m"] = new
        df["on_land"] = new == 0.0
        land_after += int(df["on_land"].sum())
        total += len(df)
        changed += int((~np.isclose(old, new, equal_nan=True)).sum())
        df.to_csv(path, index=False)
    print(f"Pings: {total:,}; depth changed for {changed:,}; on land {land_before:,} -> {land_after:,}")


if __name__ == "__main__":
    main()
