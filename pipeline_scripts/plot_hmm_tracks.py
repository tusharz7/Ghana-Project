"""Map randomly sampled trips from the HMM output, once for each minimum fishing event length n.

Visual check of the 5-minute interpolated run (fit_hmm.R). Points are labelled exactly as in
fishing_min_length_stacked.png (plot_fishing_min_length.py): a state counts when its probability is
>= PI_MIN, single-fix gaps in fishing are filled, and fishing events need at least n fixes.
The same trips are drawn for every n, one PNG per trip in result_figures/hmm_tracks/n_<n>/.
Thin grey lines join consecutive points within a segment, so gaps over 30 minutes between
segments stay open. The coastline and Tema port appear when in view.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from plot_fishing_min_length import BELOW, N_VALUES, PI_MIN, ROLE_COLORS, ROLES, load_states as load_hmm_states, role_labels

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "result_figures" / "hmm_tracks"

NATURAL_EARTH_COASTLINE = (
    Path.home()
    / ".local/share/cartopy/shapefiles/natural_earth/physical"
    / "ne_10m_coastline.shp"
)

N_TRIPS = 5
MIN_TRIP_STEPS = 40         # steps across all of a trip's segments
RANDOM_SEED = 42            # same seed = same trips; change it to see others
MAP_PADDING_DEGREES = 0.03  # minimum space around the trip

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600

ROLE_MARKERS = {"Fishing": "o", "Near-stationary": "^", "Transit": "s", BELOW: "o"}

TO_WGS84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)


# ─────────────────────────────────────────────
#  LOAD AND SAMPLE
# ─────────────────────────────────────────────

def load_states() -> pd.DataFrame:
    """Read the HMM output, add trip ids, lat/lon and a role column for each n."""
    states, state_of_role = load_hmm_states()
    print(f"State numbers by role: {state_of_role}")
    for n, role in role_labels(states, state_of_role).items():
        states[f"role_n{n}"] = role

    states["time"] = pd.to_datetime(states["time"])
    states["trip"] = states["ID"].str.replace(r"_s\d+$", "", regex=True)
    # hmm_states.csv holds UTM 31N positions in km
    lon, lat = TO_WGS84.transform(states["x"].to_numpy() * 1000, states["y"].to_numpy() * 1000)
    states["lon"] = lon
    states["lat"] = lat
    return states


def sample_trips(states: pd.DataFrame) -> tuple[list[str], int]:
    """Pick N_TRIPS at random from the trips with at least MIN_TRIP_STEPS steps."""
    steps_per_trip = states.groupby("trip")["step_km"].count()
    eligible = steps_per_trip.index[steps_per_trip >= MIN_TRIP_STEPS].to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    return sorted(rng.choice(eligible, size=N_TRIPS, replace=False)), len(eligible)


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def map_extent(trip: pd.DataFrame) -> tuple[float, float, float, float]:
    """West, east, south, north around a trip, with a little padding."""
    pad_lon = max((trip["lon"].max() - trip["lon"].min()) * 0.08, MAP_PADDING_DEGREES)
    pad_lat = max((trip["lat"].max() - trip["lat"].min()) * 0.08, MAP_PADDING_DEGREES)
    return (
        trip["lon"].min() - pad_lon,
        trip["lon"].max() + pad_lon,
        trip["lat"].min() - pad_lat,
        trip["lat"].max() + pad_lat,
    )


def plot_trip(trip: pd.DataFrame, trip_id: str, n: int, coastline: gpd.GeoDataFrame) -> Path:
    """Save one map of a trip, coloured by role for minimum event length n."""
    west, east, south, north = map_extent(trip)
    fig, ax = plt.subplots(figsize=(10, 8))

    coastline.plot(ax=ax, color="grey", linewidth=1, zorder=1)

    # One line per segment, so gaps between segments are not drawn as travel
    for _, segment in trip.groupby("ID"):
        ax.plot(segment["lon"], segment["lat"], color="lightgrey", linewidth=0.8, zorder=2)

    role = trip[f"role_n{n}"]
    for name in ROLES:
        points = trip[role == name]
        share = len(points) / len(trip) * 100
        ax.scatter(
            points["lon"], points["lat"],
            s=10 if name == BELOW else 16, color=ROLE_COLORS[name], marker=ROLE_MARKERS[name],
            linewidths=0, zorder=2.5 if name == BELOW else 3,   # grey points sit under the states
            label=f"{name} ({share:.0f}%)",
        )

    if west <= TEMA_PORT_LON <= east and south <= TEMA_PORT_LAT <= north:
        ax.scatter(TEMA_PORT_LON, TEMA_PORT_LAT, s=70, color="black", marker="*", zorder=4)
        ax.annotate("Tema", (TEMA_PORT_LON, TEMA_PORT_LAT), xytext=(9, -3),
                    textcoords="offset points", color="black")

    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect(1 / np.cos(np.radians(trip["lat"].mean())))  # degrees of lon shrink away from the equator
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(color="0.92", zorder=0)
    ax.set_title(
        f"{trip_id}   n = {n} (min fixes per fishing event), Pi_min = {PI_MIN}\n"
        f"{trip['time'].min():%Y-%m-%d} to {trip['time'].max():%Y-%m-%d}   "
        f"{trip['step_km'].count():,} steps, {trip['ID'].nunique()} segment(s)"
    )
    # Below the map, a fixed distance under the axis label whatever the map's shape
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0), borderaxespad=3.2, ncol=4, frameon=False,
              markerscale=1.5, columnspacing=1.2, handletextpad=0.3)

    output_png = OUTPUT_FOLDER / f"n_{n}" / f"{trip_id}.png"
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    states = load_states()
    trip_ids, n_eligible = sample_trips(states)
    print(f"Trips with >= {MIN_TRIP_STEPS} steps: {n_eligible}; sampling {N_TRIPS} (seed {RANDOM_SEED})")

    bounds = (states["lon"].min() - 1, states["lat"].min() - 1, states["lon"].max() + 1, states["lat"].max() + 1)
    coastline = gpd.read_file(NATURAL_EARTH_COASTLINE, bbox=bounds)

    # Same trips for every n, so the n folders can be compared map by map
    for n in N_VALUES:
        (OUTPUT_FOLDER / f"n_{n}").mkdir(parents=True, exist_ok=True)
        for trip_id in trip_ids:
            trip = states[states["trip"] == trip_id].sort_values("time")
            plot_trip(trip, trip_id, n, coastline)
        print(f"Saved {len(trip_ids)} maps to {OUTPUT_FOLDER / f'n_{n}'}")


if __name__ == "__main__":
    main()
