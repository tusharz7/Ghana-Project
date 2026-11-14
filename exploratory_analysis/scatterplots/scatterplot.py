from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
import contourpy
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


# ─────────────────────────────────────────────
#  EXPERIMENT PARAMETERS
# ─────────────────────────────────────────────

# Change these values to compare different rolling-window plots.
ROLLING_WINDOW_MINUTES = 30
MIN_POINTS_IN_WINDOW = 3
COLOR_METRIC = "dist_to_shore"

# Plot appearance and units.
DISTANCE_UNITS = "km"
SPEED_UNITS = "knots"
SCATTER_ALPHA = 0.4
SCATTER_POINT_SIZE = 10
COLORMAP = "viridis"
X_AXIS_MIN_KNOTS = 0
X_AXIS_MAX_KNOTS = 10 # just for scaling purposes
DENSITY_HEXBIN_GRID_SIZE = 50
DENSITY_COLORMAP = "magma"

# Distance constants, Tema port location, and project folders.
EARTH_RADIUS_KM = 6371.0088 # used for haversine distance calculations
KM_PER_NAUTICAL_MILE = 1.852
TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600 # look at only Tema (within 1 km)
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600
PROJECT_ROOT = Path(__file__).resolve().parents[2]
FISHER_DEPTH_FOLDER = PROJECT_ROOT / "fisherDepth"
FISHER_METADATA_FOLDER = PROJECT_ROOT / "exploratory_analysis" / "fisherMetadata"
SCATTERPLOT_OUTPUT_FOLDER = PROJECT_ROOT / "exploratory_analysis" / "scatterplots"
DEPTH_GRID_FOLDER = PROJECT_ROOT / "data"
REQUIRED_COLUMNS = {
    "fisher_id",
    "timestamp",
    "lat",
    "lon",
    "dist_to_shore",
    "dist_to_port",
}
VALID_COLOR_METRICS = {"dist_to_shore", "dist_to_port"}
GEOD = Geod(ellps="WGS84")
COASTLINE_PROJECTION = Transformer.from_crs(
    "EPSG:4326",
    "+proj=aeqd +lat_0=5 +lon_0=0 +datum=WGS84 +units=m +no_defs",
    always_xy=True,
)


# using haversine distance calculation to find distance to shore
# (takes into account the curvature of the Earth)
def _haversine_km(
    lat1: np.ndarray | float,
    lon1: np.ndarray | float,
    lat2: np.ndarray | float,
    lon2: np.ndarray | float,
) -> np.ndarray:
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    # Haversine gives great-circle distance without treating degrees as meters
    a = (
        np.sin(delta_lat / 2.0) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# defining functions for coastline extraction and distance calculations
def _build_coastline_index(
    depth_grid_folder: Path,
) -> tuple[STRtree, list[LineString]]:
    """Extract the land/water boundary and build an index of line segments."""
    xc = np.fromfile(depth_grid_folder / "XC.data", dtype=">f4").reshape(385, 1376)
    yc = np.fromfile(depth_grid_folder / "YC.data", dtype=">f4").reshape(385, 1376)
    depth = np.fromfile(
        depth_grid_folder / "Depth.data",
        dtype=">f4",
    ).reshape(385, 1376)

    # convert depth to land (0) and ocean (1)
    ocean_mask = (depth > 0).astype("float32")
    # finds the boundary where 0's turn into 1's (finding 0.5 as coastline)
    contour_lines = contourpy.contour_generator(
        x=xc[0, :],
        y=yc[:, 0],
        z=ocean_mask,
        name="serial",
    ).lines(0.5)

    # convert degrees to meters then divide coastline so we get small line segments 
    # (get a bunch of points then draw lines connecting them to get the coastline)
    segments: list[LineString] = []
    for contour_line in contour_lines:
        if len(contour_line) < 2:
            continue
        x, y = COASTLINE_PROJECTION.transform(
            contour_line[:, 0],
            contour_line[:, 1],
        )
        coordinates = list(zip(x, y))
        segments.extend(
            LineString([coordinates[index], coordinates[index + 1]])
            for index in range(len(coordinates) - 1)
            if coordinates[index] != coordinates[index + 1]
        )
    # build the STRtree index for fast nearest-neighbor queries
    return STRtree(segments), segments


def _distance_to_shore_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    coastline_index: STRtree,
    coastline_segments: list[LineString],
) -> np.ndarray:
    """Return projected point-to-nearest-coastline distances in kilometers."""
    x, y = COASTLINE_PROJECTION.transform(longitudes, latitudes)
    distances = np.empty(len(latitudes), dtype=float)

    # STRtree avoids comparing every GPS point with every coastline segment.
    for index, (point_x, point_y) in enumerate(zip(x, y)):
        point = Point(float(point_x), float(point_y))
        nearest = coastline_index.nearest(point)
        segment = coastline_segments[int(nearest)]
        distances[index] = point.distance(segment) / 1000.0
    return distances


# Return distance from Tema Fishing Harbour in km
def _distance_to_tema_port_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    _, _, distances_m = GEOD.inv(
        np.full(len(latitudes), TEMA_PORT_LON),
        np.full(len(latitudes), TEMA_PORT_LAT),
        longitudes,
        latitudes,
    )
    return np.asarray(distances_m, dtype=float) / 1000.0


# ─────────────────────────────────────────────
#  FISHER DATA LOADING
# ─────────────────────────────────────────────

def load_fisher_depth_for_plot(
    csv_path: Path,
    coastline_index: STRtree,
    coastline_segments: list[LineString],
) -> pd.DataFrame:
    """Load one point-level depth CSV and add the plotting input columns."""
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    point_columns = {"file_name", "timestamp", "lat", "lon"}
    missing = point_columns.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} is missing columns: {sorted(missing)}")

    latitudes = pd.to_numeric(df["lat"], errors="raise").to_numpy(dtype=float)
    longitudes = pd.to_numeric(df["lon"], errors="raise").to_numpy(dtype=float)

    # Convert fisherDepth columns into the common plotting input structure.
    result = pd.DataFrame(
        {
            "fisher_id": df["file_name"].astype(str),
            "timestamp": df["timestamp"],
            "lat": latitudes,
            "lon": longitudes,
            "dist_to_shore": _distance_to_shore_km(
                latitudes,
                longitudes,
                coastline_index,
                coastline_segments,
            ),
            "dist_to_port": _distance_to_tema_port_km(
                latitudes,
                longitudes,
            ),
        }
    )
    return result.sort_values(["fisher_id", "timestamp"], kind="stable")


# ─────────────────────────────────────────────
#  ROLLING SPEED AND STRAIGHTNESS
# ─────────────────────────────────────────────

# GPS points in last 30 minutes -> Distances and time between each pair ->
# Instantaneous speeds in knots -> Median speed -> Typical speed for the current point

def _window_typical_speed(
    positions: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps_ns: np.ndarray,
) -> float:
    """Return median consecutive-point speed in knots for one rolling window."""
    indices = positions.astype(np.int64)
    lat = latitudes[indices]
    lon = longitudes[indices]
    timestamps = timestamps_ns[indices]

    # Calculate speed for each consecutive pair inside this time window.
    elapsed_hours = np.diff(timestamps) / 3_600_000_000_000.0
    distances_km = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    valid = (
        (elapsed_hours > 0)
        & np.isfinite(elapsed_hours)
        & np.isfinite(distances_km)
    )
    if not valid.any():
        return np.nan

    # Convert km/hour to knots, then use the median to reduce GPS-spike effects
    speeds_knots = distances_km[valid] / elapsed_hours[valid] / KM_PER_NAUTICAL_MILE
    return float(np.median(speeds_knots))

# calculated as net displacement divided by path length
def _window_straightness(
    positions: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    timestamps_ns: np.ndarray,
) -> float:
    """Return net displacement divided by path length for one rolling window.

    Straightness is 1 for a direct path and approaches 0 when a track loops
    back toward its starting point. A zero-length path is reported as NaN
    because stationary points do not provide a meaningful travel direction.
    """
    del timestamps_ns
    indices = positions.astype(np.int64)
    lat = latitudes[indices]
    lon = longitudes[indices]

    segment_distances = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    if not np.isfinite(segment_distances).all():
        return np.nan

    path_length_km = float(segment_distances.sum())
    if path_length_km == 0.0:
        return np.nan

    # endpoint distance / full traveled path: 1 is straight, 0 is looping.
    net_displacement_km = float(_haversine_km(lat[0], lon[0], lat[-1], lon[-1]))
    return float(np.clip(net_displacement_km / path_length_km, 0.0, 1.0))


# just creating the 30 minute rolling window for the metric we use earlier
def _rolling_metric(
    group: pd.DataFrame,
    window_minutes: int,
    min_points: int,
    metric: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
) -> pd.Series:
    latitudes = group["lat"].to_numpy(dtype=float)
    longitudes = group["lon"].to_numpy(dtype=float)
    timestamps_ns = group["timestamp"].astype("int64").to_numpy()

    # Store row positions so pandas can select them with a true time-based window.
    position_series = pd.Series(
        np.arange(len(group), dtype=float),
        index=pd.DatetimeIndex(group["timestamp"]),
    )
    rolling = position_series.rolling(
        window=f"{window_minutes}min",
        min_periods=min_points,
        closed="both",
    )
    values = rolling.apply(
        lambda positions: metric(
            positions,
            latitudes,
            longitudes,
            timestamps_ns,
        ),
        raw=True,
    )
    return pd.Series(values.to_numpy(), index=group.index)


# Compute rolling typical speed and straightness independently per fisher (calls previous function)
def compute_rolling_movement_metrics(
    df: pd.DataFrame,
    window_minutes: int = ROLLING_WINDOW_MINUTES,
    min_points: int = MIN_POINTS_IN_WINDOW,
) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")
    if window_minutes <= 0:
        raise ValueError("window_minutes must be greater than zero")
    if min_points < 2:
        raise ValueError("min_points must be at least 2")

    result = df.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"], errors="raise")
    if result[list(REQUIRED_COLUMNS)].isna().any().any():
        null_columns = result[list(REQUIRED_COLUMNS)].columns[
            result[list(REQUIRED_COLUMNS)].isna().any()
        ].tolist()
        raise ValueError(f"Required columns contain missing values: {null_columns}")

    for column in ("lat", "lon", "dist_to_shore", "dist_to_port"):
        result[column] = pd.to_numeric(result[column], errors="raise")

    result = result.sort_values(
        ["fisher_id", "timestamp"],
        kind="stable",
    ).reset_index(drop=True)
    result["typical_speed_knots"] = np.nan
    result["straightness"] = np.nan

    # Compute windows separately so coordinates from different canoes never mix.
    for _, group in result.groupby("fisher_id", sort=False):
        result.loc[group.index, "typical_speed_knots"] = _rolling_metric(
            group,
            window_minutes,
            min_points,
            _window_typical_speed,
        )
        result.loc[group.index, "straightness"] = _rolling_metric(
            group,
            window_minutes,
            min_points,
            _window_straightness,
        )

    return result


# ─────────────────────────────────────────────
#  SCATTERPLOT
# ─────────────────────────────────────────────

def plot_speed_vs_straightness(
    df: pd.DataFrame,
    window_minutes: int = ROLLING_WINDOW_MINUTES,
    min_points: int = MIN_POINTS_IN_WINDOW,
    color_metric: str = COLOR_METRIC,
    x_axis_min_knots: float = X_AXIS_MIN_KNOTS,
    x_axis_max_knots: float = X_AXIS_MAX_KNOTS,
) -> tuple[Figure, Axes, pd.DataFrame]:
    """Compute rolling movement metrics and create an exploratory scatterplot.

    Returns the Matplotlib figure, axes, and enriched DataFrame so callers can
    save the plot and inspect or export the rolling metrics.
    """
    if color_metric not in VALID_COLOR_METRICS:
        raise ValueError(
            f"color_metric must be one of {sorted(VALID_COLOR_METRICS)}; "
            f"received {color_metric!r}"
        )
    if x_axis_min_knots >= x_axis_max_knots:
        raise ValueError("x_axis_min_knots must be less than x_axis_max_knots")

    metrics = compute_rolling_movement_metrics(df, window_minutes, min_points)
    plot_data = metrics.dropna(
        subset=["typical_speed_knots", "straightness", color_metric]
    )
    if plot_data.empty:
        raise ValueError(
            "No valid rolling windows remain after applying the minimum-point "
            "and missing-value requirements"
        )

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(
        plot_data["typical_speed_knots"],
        plot_data["straightness"],
        c=plot_data[color_metric],
        cmap=COLORMAP,
        alpha=SCATTER_ALPHA,
        s=SCATTER_POINT_SIZE,
        edgecolors="none",
    )

    color_label = color_metric.replace("_", " ").title()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label(f"{color_label} ({DISTANCE_UNITS})")

    # Crop only the visible axis; the underlying speed calculations stay unchanged.
    ax.set_xlabel(f"Typical speed ({SPEED_UNITS}; rolling median)")
    ax.set_ylabel("Straightness (net displacement / path length)")
    ax.set_xlim(x_axis_min_knots, x_axis_max_knots)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        f"Speed vs. Straightness ({window_minutes}-min rolling window)"
    )
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, ax, metrics


# ─────────────────────────────────────────────
#  COMBINED DENSITY MAP
# ─────────────────────────────────────────────

def plot_speed_straightness_density(
    metrics: pd.DataFrame,
    window_minutes: int = ROLLING_WINDOW_MINUTES,
    grid_size: int = DENSITY_HEXBIN_GRID_SIZE,
) -> tuple[Figure, Axes]:
    """Plot a hexbin count-density map from valid rolling-window metrics."""
    if grid_size <= 0:
        raise ValueError("grid_size must be greater than zero")

    plot_data = metrics.dropna(subset=["typical_speed_knots", "straightness"])
    if plot_data.empty:
        raise ValueError("No valid rolling windows are available for the density map")

    fig, ax = plt.subplots(figsize=(10, 7))
    hexagons = ax.hexbin(
        plot_data["typical_speed_knots"],
        plot_data["straightness"],
        gridsize=grid_size,
        extent=(X_AXIS_MIN_KNOTS, X_AXIS_MAX_KNOTS, 0.0, 1.0),
        mincnt=1,
        cmap=DENSITY_COLORMAP,
        norm=LogNorm(),
    )
    colorbar = fig.colorbar(hexagons, ax=ax)
    colorbar.set_label("Rolling-window point count (log scale)")

    ax.set_xlabel(f"Typical speed ({SPEED_UNITS}; rolling median)")
    ax.set_ylabel("Straightness (net displacement / path length)")
    ax.set_xlim(X_AXIS_MIN_KNOTS, X_AXIS_MAX_KNOTS)
    ax.set_ylim(0.0, 1.02)
    ax.set_title(
        "All canoes density map\n"
        f"Speed vs. Straightness ({window_minutes}-min rolling window)"
    )
    ax.grid(alpha=0.2)
    fig.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────
#  GENERATE ONE PLOT PER CANOE
# ─────────────────────────────────────────────

def generate_all_canoe_scatterplots(
    metadata_folder: Path = FISHER_METADATA_FOLDER,
    fisher_depth_folder: Path = FISHER_DEPTH_FOLDER,
    output_folder: Path = SCATTERPLOT_OUTPUT_FOLDER,
    depth_grid_folder: Path = DEPTH_GRID_FOLDER,
    window_minutes: int = ROLLING_WINDOW_MINUTES,
    min_points: int = MIN_POINTS_IN_WINDOW,
    color_metric: str = COLOR_METRIC,
) -> pd.DataFrame:
    """Generate one scatterplot for every per-fisher metadata CSV.

    Metadata filenames identify which canoes to process, while the corresponding
    ``fisherDepth`` CSV supplies the required point-level GPS observations.
    Returns a summary containing each canoe's source and output paths.
    """
    metadata_paths = sorted(metadata_folder.glob("OMI_Fisher_*_metadata.csv"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No per-fisher metadata CSVs found in {metadata_folder}"
        )

    # Match each metadata filename with its point-level fisherDepth filename.
    source_pairs: list[tuple[Path, Path]] = []
    missing_depth_paths: list[Path] = []
    for metadata_path in metadata_paths:
        depth_name = metadata_path.name.replace("_metadata.csv", "_depth.csv")
        depth_path = fisher_depth_folder / depth_name
        if not depth_path.exists():
            missing_depth_paths.append(depth_path)
        source_pairs.append((metadata_path, depth_path))

    if missing_depth_paths:
        missing_list = "\n".join(f"  - {path}" for path in missing_depth_paths)
        raise FileNotFoundError(
            "Missing point-level fisherDepth files for metadata canoes:\n"
            f"{missing_list}"
        )

    output_folder.mkdir(parents=True, exist_ok=True)
    coastline_index, coastline_segments = _build_coastline_index(
        depth_grid_folder
    )

    # Reuse one coastline index for every canoe instead of rebuilding it 25 times.
    summary_rows = []
    for metadata_path, depth_path in source_pairs:
        canoe_df = load_fisher_depth_for_plot(
            depth_path,
            coastline_index,
            coastline_segments,
        )
        fig, ax, metrics = plot_speed_vs_straightness(
            canoe_df,
            window_minutes=window_minutes,
            min_points=min_points,
            color_metric=color_metric,
        )
        fisher_id = str(canoe_df["fisher_id"].iloc[0])
        ax.set_title(
            f"{fisher_id}\n"
            f"Speed vs. Straightness ({window_minutes}-min rolling window)"
        )
        fig.tight_layout()

        output_path = output_folder / metadata_path.name.replace(
            "_metadata.csv",
            "_speed_vs_straightness.png",
        )
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Record how many GPS points produced usable rolling-window observations.
        valid_windows = metrics[
            ["typical_speed_knots", "straightness", color_metric]
        ].dropna()
        summary_rows.append(
            {
                "fisher_id": fisher_id,
                "metadata_csv": str(metadata_path),
                "fisher_depth_csv": str(depth_path),
                "output_plot": str(output_path),
                "gps_points": len(canoe_df),
                "valid_rolling_windows": len(valid_windows),
                "window_minutes": window_minutes,
                "min_points": min_points,
                "color_metric": color_metric,
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_folder / "scatterplot_generation_summary.csv", index=False)
    return summary


# ─────────────────────────────────────────────
#  GENERATE ONE COMBINED PLOT
# ─────────────────────────────────────────────

def generate_combined_canoe_scatterplot(
    metadata_folder: Path = FISHER_METADATA_FOLDER,
    fisher_depth_folder: Path = FISHER_DEPTH_FOLDER,
    output_folder: Path = SCATTERPLOT_OUTPUT_FOLDER,
    depth_grid_folder: Path = DEPTH_GRID_FOLDER,
    window_minutes: int = ROLLING_WINDOW_MINUTES,
    min_points: int = MIN_POINTS_IN_WINDOW,
    color_metric: str = COLOR_METRIC,
) -> tuple[Path, Path]:
    """Generate combined scatterplot and hexbin density map for all canoes."""
    metadata_paths = sorted(metadata_folder.glob("OMI_Fisher_*_metadata.csv"))
    if not metadata_paths:
        raise FileNotFoundError(
            f"No per-fisher metadata CSVs found in {metadata_folder}"
        )

    coastline_index, coastline_segments = _build_coastline_index(
        depth_grid_folder
    )
    canoe_frames = []

    # Metadata filenames determine which corresponding fisherDepth files to load.
    for metadata_path in metadata_paths:
        depth_path = fisher_depth_folder / metadata_path.name.replace(
            "_metadata.csv",
            "_depth.csv",
        )
        if not depth_path.exists():
            raise FileNotFoundError(
                f"Missing point-level file for {metadata_path.name}: {depth_path}"
            )
        canoe_frames.append(
            load_fisher_depth_for_plot(
                depth_path,
                coastline_index,
                coastline_segments,
            )
        )

    combined = pd.concat(canoe_frames, ignore_index=True)
    fig, ax, metrics = plot_speed_vs_straightness(
        combined,
        window_minutes=window_minutes,
        min_points=min_points,
        color_metric=color_metric,
    )
    ax.set_title(
        "All canoes\n"
        f"Speed vs. Straightness ({window_minutes}-min rolling window)"
    )
    fig.tight_layout()

    output_folder.mkdir(parents=True, exist_ok=True)
    output_path = output_folder / "all_canoes_speed_vs_straightness.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    density_fig, _ = plot_speed_straightness_density(
        metrics,
        window_minutes=window_minutes,
    )
    density_output_path = (
        output_folder / "all_canoes_speed_vs_straightness_density.png"
    )
    density_fig.savefig(density_output_path, dpi=300, bbox_inches="tight")
    plt.close(density_fig)
    return output_path, density_output_path



def main() -> None:
    """Ask which plot mode to run, then generate the requested output."""
    print("Choose how to plot the canoe data:")
    print("  0 = one combined plot containing all canoes")
    print("  1 = one separate plot for each canoe")
    choice = input("Enter 0 or 1: ").strip()

    if choice == "0":
        scatterplot_path, density_map_path = generate_combined_canoe_scatterplot()
        print(f"Saved combined scatterplot to {scatterplot_path}")
        print(f"Saved combined density map to {density_map_path}")
    elif choice == "1":
        summary = generate_all_canoe_scatterplots()
        print(
            f"Saved {len(summary)} separate canoe plots to "
            f"{SCATTERPLOT_OUTPUT_FOLDER}"
        )
    else:
        raise ValueError("Invalid choice. Run the script again and enter 0 or 1.")


if __name__ == "__main__":
    main()
