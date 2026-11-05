"""Map randomly sampled trips from the HMM output, with points coloured by Viterbi state.

Visual check of the 5-minute interpolated run (fit_hmm.R). Each sampled trip gets its own PNG,
zoomed to the trip. Thin grey lines join consecutive points within a segment, so gaps over
30 minutes between segments stay open. The coastline and Tema port appear when in view.
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

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
STATES_CSV = PROJECT_ROOT / "hmm_states.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "hmm_track_maps"

NATURAL_EARTH_COASTLINE = (
    Path.home()
    / ".local/share/cartopy/shapefiles/natural_earth/physical"
    / "ne_10m_coastline.shp"
)

N_TRIPS = 12
MIN_TRIP_STEPS = 40         # steps across all of a trip's segments
RANDOM_SEED = 42            # same seed = same trips; change it to see others
MAP_PADDING_DEGREES = 0.03  # minimum space around the trip

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600

STATE_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}  # same as plot_state_sweep.R
STATE_MARKERS = {1: "o", 2: "^", 3: "s"}

TO_WGS84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)


# ─────────────────────────────────────────────
#  LOAD AND SAMPLE
# ─────────────────────────────────────────────

def load_states() -> pd.DataFrame:
    """Read the HMM output and add trip ids and lat/lon."""
    states = pd.read_csv(STATES_CSV, parse_dates=["time"])
    states["trip"] = states["ID"].str.replace(r"_s\d+$", "", regex=True)

    # hmm_states.csv holds UTM 31N positions in km
    lon, lat = TO_WGS84.transform(states["x"].to_numpy() * 1000, states["y"].to_numpy() * 1000)
    states["lon"] = lon
    states["lat"] = lat
    return states


def state_roles(states: pd.DataFrame) -> dict[int, str]:
    """Name states by mean step length, the same ranking rule as fit_hmm.R."""
    mean_step = states.groupby("viterbi_state")["step_km"].mean().sort_values()
    return dict(zip(mean_step.index, ["Near-stationary", "Fishing", "Transit"]))


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


def plot_trip(trip: pd.DataFrame, trip_id: str, roles: dict[int, str], coastline: gpd.GeoDataFrame) -> Path:
    """Save one map of a trip, coloured by Viterbi state."""
    west, east, south, north = map_extent(trip)
    fig, ax = plt.subplots(figsize=(10, 8))

    coastline.plot(ax=ax, color="grey", linewidth=1, zorder=1)

    # One line per segment, so gaps between segments are not drawn as travel
    for _, segment in trip.groupby("ID"):
        ax.plot(segment["lon"], segment["lat"], color="lightgrey", linewidth=0.8, zorder=2)

    for state in sorted(STATE_COLORS):
        points = trip[trip["viterbi_state"] == state]
        share = len(points) / len(trip) * 100
        ax.scatter(
            points["lon"], points["lat"],
            s=16, color=STATE_COLORS[state], marker=STATE_MARKERS[state],
            linewidths=0, zorder=3,
            label=f"State {state}: {roles[state]} ({share:.0f}%)",
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
        f"{trip_id}\n{trip['time'].min():%Y-%m-%d} to {trip['time'].max():%Y-%m-%d}   "
        f"{trip['step_km'].count():,} steps, {trip['ID'].nunique()} segment(s)"
    )
    # Below the map, so it never covers the track
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, frameon=False,
              markerscale=1.5, columnspacing=1.2, handletextpad=0.3)

    output_png = OUTPUT_FOLDER / f"{trip_id}.png"
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_png


def main() -> None:
    states = load_states()
    roles = state_roles(states)
    trip_ids, n_eligible = sample_trips(states)
    print(f"Trips with >= {MIN_TRIP_STEPS} steps: {n_eligible}; sampling {N_TRIPS} (seed {RANDOM_SEED})")
    print(f"State roles: {roles}")

    OUTPUT_FOLDER.mkdir(exist_ok=True)
    bounds = (states["lon"].min() - 1, states["lat"].min() - 1, states["lon"].max() + 1, states["lat"].max() + 1)
    coastline = gpd.read_file(NATURAL_EARTH_COASTLINE, bbox=bounds)

    for trip_id in trip_ids:
        trip = states[states["trip"] == trip_id].sort_values("time")
        print(f"Saved {plot_trip(trip, trip_id, roles, coastline)}")


if __name__ == "__main__":
    main()
