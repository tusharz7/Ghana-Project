"""Compare the model depth-grid coastline with Natural Earth coastline data."""

from pathlib import Path

import contourpy
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, box
from shapely.ops import transform, unary_union


# ─────────────────────────────────────────────
#  FILES AND PLOT SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPTH_GRID_FOLDER = PROJECT_ROOT / "data"
OUTPUT_FOLDER = PROJECT_ROOT / "fisherMetadata"

NATURAL_EARTH_COASTLINE = (
    Path.home()
    / ".local/share/cartopy/shapefiles/natural_earth/physical"
    / "ne_10m_coastline.shp"
)

OUTPUT_PLOT = OUTPUT_FOLDER / "coastline_sanity_check.png"
OUTPUT_SUMMARY = OUTPUT_FOLDER / "coastline_sanity_check.txt"

# Ghana-region plotting extent: west, east, south, north.
PLOT_EXTENT = (-3.5, 1.5, 4.0, 6.5)
CONTOUR_SAMPLE_SPACING_METERS = 1000

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600


# ─────────────────────────────────────────────
#  EXTRACT DEPTH-GRID COASTLINE
# ─────────────────────────────────────────────

def load_depth_grid_coastline(depth_grid_folder: Path) -> list[LineString]:
    """Extract the continuous land/ocean boundary from the model depth grid."""
    xc = np.fromfile(
        depth_grid_folder / "XC.data",
        dtype=">f4",
    ).reshape(385, 1376)
    yc = np.fromfile(
        depth_grid_folder / "YC.data",
        dtype=">f4",
    ).reshape(385, 1376)
    depth = np.fromfile(
        depth_grid_folder / "Depth.data",
        dtype=">f4",
    ).reshape(385, 1376)

    # Confirm the binary files use rows for latitude and columns for longitude.
    if not np.allclose(xc, xc[0:1, :]):
        raise ValueError("XC.data does not contain longitude-constant rows")
    if not np.allclose(yc, yc[:, 0:1]):
        raise ValueError("YC.data does not contain latitude-constant columns")

    # Land is depth 0 and ocean is depth > 0. The 0.5 contour falls between them.
    ocean_mask = (depth > 0).astype("float32")
    contour_lines = contourpy.contour_generator(
        x=xc[0, :],
        y=yc[:, 0],
        z=ocean_mask,
        name="serial",
    ).lines(0.5)

    coastline = [
        LineString(contour)
        for contour in contour_lines
        if len(contour) >= 2
    ]
    if not coastline:
        raise ValueError(
            f"No coastline could be extracted from {depth_grid_folder}"
        )
    return coastline


# ─────────────────────────────────────────────
#  LOAD REFERENCE COASTLINE
# ─────────────────────────────────────────────

def load_reference_coastline(coastline_path: Path):
    """Load the Natural Earth coastline used as the visual reference."""
    if not coastline_path.exists():
        raise FileNotFoundError(
            "Natural Earth coastline is missing. Expected it at "
            f"{coastline_path}"
        )
    return gpd.read_file(coastline_path)


# ─────────────────────────────────────────────
#  QUANTIFY COASTLINE ALIGNMENT
# ─────────────────────────────────────────────

def coastline_alignment_distances_km(
    model_coastline: list[LineString],
    reference_coastline: gpd.GeoDataFrame,
    extent: tuple[float, float, float, float],
) -> np.ndarray:
    """Measure sampled model-coastline distances to the reference coastline."""
    west, east, south, north = extent
    comparison_region = box(west, south, east, north)

    model_clipped = [
        line.intersection(comparison_region)
        for line in model_coastline
        if line.intersects(comparison_region)
    ]
    model_clipped = [line for line in model_clipped if not line.is_empty]

    reference_clipped = [
        geometry.intersection(comparison_region)
        for geometry in reference_coastline.geometry
        if geometry.intersects(comparison_region)
    ]
    reference_clipped = [
        geometry for geometry in reference_clipped if not geometry.is_empty
    ]

    if not model_clipped or not reference_clipped:
        raise ValueError("No coastline was found inside the comparison extent")

    model_union = unary_union(model_clipped)
    reference_union = unary_union(reference_clipped)

    # Use a Ghana-centered equidistant projection so Shapely distances are meters.
    project_to_meters = Transformer.from_crs(
        "EPSG:4326",
        "+proj=aeqd +lat_0=5.4 +lon_0=-1 +datum=WGS84 +units=m +no_defs",
        always_xy=True,
    ).transform
    model_meters = transform(project_to_meters, model_union)
    reference_meters = transform(project_to_meters, reference_union)

    model_parts = (
        list(model_meters.geoms)
        if hasattr(model_meters, "geoms")
        else [model_meters]
    )
    distances_meters = []

    # Sample the model contour every kilometer and find reference-line distance.
    for part in model_parts:
        if part.length == 0:
            continue
        sample_positions = np.arange(
            0,
            part.length + 1,
            CONTOUR_SAMPLE_SPACING_METERS,
        )
        for position in sample_positions:
            point = part.interpolate(min(position, part.length))
            distances_meters.append(point.distance(reference_meters))

    if not distances_meters:
        raise ValueError("No model coastline points were available for comparison")
    return np.asarray(distances_meters, dtype=float) / 1000


# ─────────────────────────────────────────────
#  CREATE OVERLAY PLOT
# ─────────────────────────────────────────────

def create_overlay_plot(
    model_coastline: list[LineString],
    reference_coastline: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    """Plot the model coastline over the Natural Earth reference coastline."""
    west, east, south, north = PLOT_EXTENT
    fig, ax = plt.subplots(figsize=(12, 7))

    reference_coastline.plot(
        ax=ax,
        color="black",
        linewidth=1.8,
        label="Natural Earth 10m coastline",
        zorder=3,
    )

    for index, line in enumerate(model_coastline):
        longitude, latitude = line.xy
        ax.plot(
            longitude,
            latitude,
            color="red",
            linewidth=1.2,
            alpha=0.85,
            label="Depth-grid zero boundary" if index == 0 else None,
            zorder=4,
        )

    ax.scatter(
        [TEMA_PORT_LON],
        [TEMA_PORT_LAT],
        marker="*",
        s=120,
        color="gold",
        edgecolor="black",
        label="Tema Fishing Harbour",
        zorder=5,
    )
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(
        "Coastline sanity check: depth-grid contour vs Natural Earth 10m"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────
#  RUN CHECK
# ─────────────────────────────────────────────

def main() -> None:
    model_coastline = load_depth_grid_coastline(DEPTH_GRID_FOLDER)
    reference_coastline = load_reference_coastline(NATURAL_EARTH_COASTLINE)

    distances_km = coastline_alignment_distances_km(
        model_coastline,
        reference_coastline,
        PLOT_EXTENT,
    )
    create_overlay_plot(
        model_coastline,
        reference_coastline,
        OUTPUT_PLOT,
    )

    west, east, south, north = PLOT_EXTENT
    summary = (
        f"Comparison extent: lon {west} to {east}, "
        f"lat {south} to {north}\n"
        f"Model contour sample points: {len(distances_km)}\n"
        f"Median model-to-Natural-Earth distance: "
        f"{np.median(distances_km):.3f} km\n"
        f"90th percentile distance: "
        f"{np.percentile(distances_km, 90):.3f} km\n"
        f"95th percentile distance: "
        f"{np.percentile(distances_km, 95):.3f} km\n"
        f"Maximum distance: {np.max(distances_km):.3f} km\n"
    )
    OUTPUT_SUMMARY.write_text(summary)

    print(summary, end="")
    print(f"Saved overlay to {OUTPUT_PLOT}")
    print(f"Saved summary to {OUTPUT_SUMMARY}")


if __name__ == "__main__":
    main()
