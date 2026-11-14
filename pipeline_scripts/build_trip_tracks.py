"""Build one track per boat-trip for crawlWrap, using the same trip split as the metadata.

Trips are defined exactly as in fisherMetadata (per trip)/generate_fisher_metadata.py:
a trip is a run of pings outside the 1 km Tema-port radius, split again wherever pings are
more than MAX_TRIP_GAP_DAYS apart. Each trip gets its own track_id so that separate trips
are never joined into one continuous track.
The original boat ID is kept in its own column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Geod, Transformer

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FISHER_DEPTH_FOLDER = PROJECT_ROOT / "fisherDepth"
OUTPUT_CSV = PROJECT_ROOT / "pipeline_outputs" / "gps_tracks" / "trip_tracks.csv"

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600
TEMA_PORT_RADIUS_KM = 1.0
MAX_GAP_MINUTES = 30  # observed gaps longer than this start a new track segment
MAX_TRIP_GAP_DAYS = 4  # observed gaps longer than this end the trip; the next ping starts a new one

GEOD = Geod(ellps="WGS84")
TO_UTM31N = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
TO_WGS84 = Transformer.from_crs("EPSG:32631", "EPSG:4326", always_xy=True)
MAX_SPEED_KMH = 30  # steps faster than this are implausible GPS jumps
CAP_SPEED_KMH = 20  # ...and are pulled back to this speed along the same bearing


# ─────────────────────────────────────────────
#  TRIP SPLITTING (same logic as generate_fisher_metadata.py)
# ─────────────────────────────────────────────

def distance_from_port_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Geodesic distance from Tema Fishing Harbour, in km."""
    _, _, distance_m = GEOD.inv(
        np.full(len(lat), TEMA_PORT_LON),
        np.full(len(lat), TEMA_PORT_LAT),
        lon,
        lat,
    )
    return np.asarray(distance_m) / 1000


def detect_tema_trips(in_port: np.ndarray) -> list[tuple[int, int, bool]]:
    """Return (start, end, returned_to_tema) row indices, end inclusive."""
    trips, row = [], 0
    while row < len(in_port):
        if in_port[row]:
            while row < len(in_port) and in_port[row]:
                row += 1
            if row == len(in_port):
                break
        start = row
        while row < len(in_port) and not in_port[row]:
            row += 1
        if row < len(in_port):
            end, returned = row, True
            while row < len(in_port) and in_port[row]:
                row += 1
        else:
            end, returned = len(in_port) - 1, False
        trips.append((start, end, returned))
    return trips or [(0, len(in_port) - 1, False)]


def split_at_long_gaps(
    times: np.ndarray, trips: list[tuple[int, int, bool]]
) -> list[tuple[int, int, bool, str]]:
    """Split each trip wherever consecutive pings are more than MAX_TRIP_GAP_DAYS apart.

    Returns (start, end, returned_to_tema, trip_end_reason), end inclusive.
    A piece that ends at a long gap did not return to Tema.
    """
    pieces = []
    for start, end, returned in trips:
        gap_days = np.diff(times[start:end + 1]) / np.timedelta64(1, "D")
        cuts = start + 1 + np.flatnonzero(gap_days > MAX_TRIP_GAP_DAYS)   # first row of each new piece
        bounds = [start, *cuts, end + 1]
        for k in range(len(bounds) - 1):
            last = k == len(bounds) - 2
            reason = ("port" if returned else "end_of_data") if last else "gap"
            pieces.append((bounds[k], bounds[k + 1] - 1, returned and last, reason))
    return pieces


# ─────────────────────────────────────────────
#  BUILD TRACKS
# ─────────────────────────────────────────────

def build_tracks() -> pd.DataFrame:
    pieces = []
    for path in sorted(FISHER_DEPTH_FOLDER.glob("*_depth.csv")):
        points = pd.read_csv(path, parse_dates=["timestamp"]).dropna(
            subset=["timestamp", "lat", "lon"]
        )
        points = points.sort_values("timestamp").reset_index(drop=True)
        boat_id = path.name.replace("_depth.csv", "")
        in_port = distance_from_port_km(
            points["lat"].to_numpy(), points["lon"].to_numpy()
        ) <= TEMA_PORT_RADIUS_KM

        trips = split_at_long_gaps(points["timestamp"].to_numpy(), detect_tema_trips(in_port))
        for number, (start, end, returned, end_reason) in enumerate(trips, start=1):
            trip = points.iloc[start:end + 1].copy()
            # Split long observed gaps inside a trip so crawlWrap never interpolates across them
            gap_min = trip["timestamp"].diff().dt.total_seconds().fillna(0) / 60
            trip["segment"] = (gap_min > MAX_GAP_MINUTES).cumsum() + 1
            trip["boat_id"] = boat_id
            trip["trip_index"] = number
            trip["track_id"] = f"{boat_id}_t{number}_s" + trip["segment"].astype(str)
            trip["returned_to_tema"] = returned
            trip["trip_end_reason"] = end_reason   # port, gap (> MAX_TRIP_GAP_DAYS) or end_of_data
            pieces.append(trip)

    tracks = pd.concat(pieces, ignore_index=True)
    x, y = TO_UTM31N.transform(tracks["lon"].to_numpy(), tracks["lat"].to_numpy())
    tracks["x"] = x
    tracks["y"] = y
    tracks = cap_implausible_speeds(tracks)
    return tracks[[
        "track_id", "boat_id", "trip_index", "segment", "returned_to_tema",
        "trip_end_reason", "timestamp", "lat", "lon", "x", "y", "speed", "depth_m", "on_land", "speed_capped",
    ]]


def cap_implausible_speeds(tracks: pd.DataFrame) -> pd.DataFrame:
    """Pull back any step faster than MAX_SPEED_KMH to CAP_SPEED_KMH along the same bearing.

    Works in metres (UTM 31N) within each track, in time order. Each step starts from the
    previous point after it has been capped, so one bad ping cannot shift the rest of the track.
    """
    tracks = tracks.sort_values(["track_id", "timestamp"]).reset_index(drop=True)
    x = tracks["x"].to_numpy().copy()
    y = tracks["y"].to_numpy().copy()
    times = tracks["timestamp"].to_numpy()
    capped = np.zeros(len(tracks), dtype=bool)

    for _, idx in tracks.groupby("track_id").indices.items():
        for k in range(1, len(idx)):
            i, p = idx[k], idx[k - 1]
            dt_h = (times[i] - times[p]) / np.timedelta64(1, "h")
            dx, dy = x[i] - x[p], y[i] - y[p]
            dist_km = np.hypot(dx, dy) / 1000
            if dt_h <= 0:
                speed_kmh = np.inf if dist_km > 0 else 0.0
            else:
                speed_kmh = dist_km / dt_h
            if speed_kmh > MAX_SPEED_KMH:
                allowed_km = CAP_SPEED_KMH * max(dt_h, 0.0)
                scale = allowed_km / dist_km if dist_km > 0 else 0.0
                x[i] = x[p] + dx * scale
                y[i] = y[p] + dy * scale
                capped[i] = True

    tracks["x"] = x
    tracks["y"] = y
    tracks["speed_capped"] = capped
    # Positions changed only where capped, so convert just those back to lat/lon
    if capped.any():
        lon, lat = TO_WGS84.transform(x[capped], y[capped])
        tracks.loc[capped, "lon"] = lon
        tracks.loc[capped, "lat"] = lat
    return tracks


def main() -> None:
    tracks = build_tracks()
    tracks.to_csv(OUTPUT_CSV, index=False)
    n_tracks = tracks["track_id"].nunique()
    n_open = tracks.drop_duplicates("track_id")["returned_to_tema"].eq(False).sum()
    print(f"Saved {OUTPUT_CSV}")
    print(f"Rows: {len(tracks):,}   tracks (trip segments): {n_tracks}   boats: {tracks['boat_id'].nunique()}")
    print(f"Pings capped to {CAP_SPEED_KMH} km/h (was > {MAX_SPEED_KMH} km/h): {int(tracks['speed_capped'].sum())}")
    print(f"Tracks not returned to Tema (kept, flagged): {n_open}")
    trips = tracks.drop_duplicates(["boat_id", "trip_index"])
    print(f"Trips: {len(trips)}   ended by a gap over {MAX_TRIP_GAP_DAYS} days: "
          f"{int((trips['trip_end_reason'] == 'gap').sum())}")


if __name__ == "__main__":
    main()
