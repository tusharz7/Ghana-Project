"""Build one track per boat-trip for the interpolation, using the same trip split as the metadata.

Trips are defined exactly as in fisherMetadata (per trip)/generate_fisher_metadata.py:
a trip is a run of pings outside the 1 km Tema-port radius, split again wherever pings are
more than MAX_TRIP_GAP_DAYS apart. Each trip gets its own track_id so that separate trips
are never joined into one continuous track.
The original boat ID is kept in its own column.

Only trips that both start by leaving Tema and end back in Tema are kept (trip_tracks.csv). The rest
(boats based elsewhere, pieces after a long gap, trips cut off by a gap or the end of the data) go to
dropped_non_tema_trips.csv. Trip numbers are kept from the full split, so kept trips can skip numbers.
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
DROPPED_CSV = PROJECT_ROOT / "pipeline_outputs" / "gps_tracks" / "dropped_non_tema_trips.csv"

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600
TEMA_PORT_RADIUS_KM = 1.0
MAX_GAP_MINUTES = 30  # observed gaps longer than this start a new track segment
MAX_TRIP_GAP_DAYS = 4  # observed gaps longer than this end the trip; the next ping starts a new one

GEOD = Geod(ellps="WGS84")
TO_UTM31N = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
MAX_SPEED_KMH = 30     # a step faster than this that comes back within a few pings is a GPS spike
MAX_SPIKE_PINGS = 3    # longest run of bad pings removed as one spike
MIN_PING_SECONDS = 60  # pings closer than this with a fast step between are near-duplicates
MIN_DETOUR_RATIO = 2   # a spike's path out and back is at least this many times the direct distance


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

def build_tracks() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (Tema-to-Tema trips, all other trips)."""
    pieces, n_spikes = [], 0
    for path in sorted(FISHER_DEPTH_FOLDER.glob("*_depth.csv")):
        points = pd.read_csv(path, parse_dates=["timestamp"]).dropna(
            subset=["timestamp", "lat", "lon"]
        )
        points = points.sort_values("timestamp").reset_index(drop=True)
        n_before = len(points)
        points = drop_speed_spikes(points)
        n_spikes += n_before - len(points)
        boat_id = path.name.replace("_depth.csv", "")
        in_port = distance_from_port_km(
            points["lat"].to_numpy(), points["lon"].to_numpy()
        ) <= TEMA_PORT_RADIUS_KM

        trips = split_at_long_gaps(points["timestamp"].to_numpy(), detect_tema_trips(in_port))
        for number, (start, end, returned, end_reason) in enumerate(trips, start=1):
            trip = points.iloc[start:end + 1].copy()
            # Split long observed gaps inside a trip so the interpolation never bridges them
            gap_min = trip["timestamp"].diff().dt.total_seconds().fillna(0) / 60
            trip["segment"] = (gap_min > MAX_GAP_MINUTES).cumsum() + 1
            trip["boat_id"] = boat_id
            trip["trip_index"] = number
            trip["track_id"] = f"{boat_id}_t{number}_s" + trip["segment"].astype(str)
            trip["returned_to_tema"] = returned
            trip["trip_end_reason"] = end_reason   # port, gap (> MAX_TRIP_GAP_DAYS) or end_of_data
            trip["started_from_tema"] = bool(start > 0 and in_port[start - 1])   # ping before the trip is in port
            pieces.append(trip)

    tracks = pd.concat(pieces, ignore_index=True)
    x, y = TO_UTM31N.transform(tracks["lon"].to_numpy(), tracks["lat"].to_numpy())
    tracks["x"] = x
    tracks["y"] = y
    tracks = tracks[[
        "track_id", "boat_id", "trip_index", "segment", "started_from_tema", "returned_to_tema",
        "trip_end_reason", "timestamp", "lat", "lon", "x", "y", "speed", "depth_m", "on_land",
    ]].copy()
    tema_trip = tracks["started_from_tema"] & tracks["returned_to_tema"]
    kept, dropped = tracks[tema_trip].reset_index(drop=True), tracks[~tema_trip].reset_index(drop=True)
    kept.attrs["spikes_dropped"] = n_spikes
    return kept, dropped


def step_speeds_kmh(lat: np.ndarray, lon: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Geodesic speed (km/h) of each step between consecutive pings; inf for a jump in zero time."""
    _, _, dist_m = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    hours = np.diff(times) / np.timedelta64(1, "h")
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(hours > 0, np.asarray(dist_m) / 1000 / hours, np.where(np.asarray(dist_m) > 0, np.inf, 0.0))
    return speed


def drop_speed_spikes(points: pd.DataFrame) -> pd.DataFrame:
    """Remove short GPS excursions from one boat's time-sorted pings; no ping is moved.

    After a step faster than MAX_SPEED_KMH, look up to MAX_SPIKE_PINGS pings ahead for the first
    ping the boat could have reached directly from the last good ping (at most MAX_SPEED_KMH, or
    at most half the speed of the jump) AND that the pings in between detour to (path at least
    MIN_DETOUR_RATIO times the direct distance) AND that is reached by a fast step too. Those
    pings are a jump out and back and are removed. A boat that is really fast for a while goes nearly straight, so it is kept.
    Two pings under MIN_PING_SECONDS apart with a fast step between are near-duplicates: the one
    farther from the next ping is removed. A fast first or last step drops the end ping when the
    step inward is normal. Repeats until nothing changes, since a removal changes the neighbours.
    """
    keep = np.ones(len(points), dtype=bool)
    lat_all, lon_all = points["lat"].to_numpy(float), points["lon"].to_numpy(float)
    times_all = points["timestamp"].to_numpy()

    def dist_km(i: int, j: int) -> float:
        return GEOD.inv(lon_all[i], lat_all[i], lon_all[j], lat_all[j])[2] / 1000

    def speed_kmh(i: int, j: int) -> float:
        d = dist_km(i, j)
        h = (times_all[j] - times_all[i]) / np.timedelta64(1, "h")
        return d / h if h > 0 else (np.inf if d > 0 else 0.0)

    while True:
        idx = np.flatnonzero(keep)
        if len(idx) < 3:
            break
        v = step_speeds_kmh(lat_all[idx], lon_all[idx], times_all[idx])   # v[k]: ping k to k+1
        secs = np.diff(times_all[idx]) / np.timedelta64(1, "s")
        drop: set[int] = set()
        next_k = 0                                                 # never search from a ping being dropped
        for k in np.flatnonzero(v > MAX_SPEED_KMH):
            if k < next_k or idx[k] in drop:
                continue
            if secs[k] < MIN_PING_SECONDS and k + 2 < len(idx):
                # near-duplicate pair: drop whichever is farther from the ping after them
                a_far = speed_kmh(idx[k], idx[k + 2]) > speed_kmh(idx[k + 1], idx[k + 2])
                drop.add(idx[k] if a_far else idx[k + 1])
                continue
            if k == 0 and not v[1] > MAX_SPEED_KMH:
                drop.add(idx[0])                                   # bad first ping
                continue
            limit = max(MAX_SPEED_KMH, 0.5 * v[k])
            for j in range(k + 2, min(k + 2 + MAX_SPIKE_PINGS, len(idx))):
                path = sum(dist_km(idx[m], idx[m + 1]) for m in range(k, j))
                if (v[j - 1] > MAX_SPEED_KMH and speed_kmh(idx[k], idx[j]) <= limit
                        and path >= MIN_DETOUR_RATIO * dist_km(idx[k], idx[j])):
                    drop.update(idx[k + 1:j])                      # out and back: drop the pings between
                    next_k = j
                    break
            else:
                if k == len(idx) - 2 and not v[k - 1] > MAX_SPEED_KMH:
                    drop.add(idx[-1])                              # bad last ping
        if not drop:
            break
        keep[list(drop)] = False
    return points[keep].reset_index(drop=True)


def main() -> None:
    tracks, dropped = build_tracks()
    tracks.to_csv(OUTPUT_CSV, index=False)
    dropped.to_csv(DROPPED_CSV, index=False)
    n_tracks = tracks["track_id"].nunique()
    print(f"Saved {OUTPUT_CSV}")
    print(f"Rows: {len(tracks):,}   tracks (trip segments): {n_tracks}   boats: {tracks['boat_id'].nunique()}")
    print(f"GPS spike pings dropped (> {MAX_SPEED_KMH} km/h out and back): {tracks.attrs['spikes_dropped']}")
    fast = [step_speeds_kmh(t.lat.to_numpy(), t.lon.to_numpy(), t.timestamp.to_numpy()) > MAX_SPEED_KMH
            for _, t in tracks.groupby("track_id") if len(t) > 1]
    print(f"Steps inside segments still faster than {MAX_SPEED_KMH} km/h (kept, not spikes): "
          f"{int(sum(f.sum() for f in fast))}")
    trips = tracks.drop_duplicates(["boat_id", "trip_index"])
    gone = dropped.drop_duplicates(["boat_id", "trip_index"])
    total = len(tracks) + len(dropped)
    print(f"Tema-to-Tema trips kept: {len(trips)} of {len(trips) + len(gone)}; "
          f"pings dropped: {len(dropped):,} of {total:,} ({len(dropped) / total:.1%}) -> {DROPPED_CSV.name}")
    why = np.select([~gone.started_from_tema & ~gone.returned_to_tema, ~gone.started_from_tema], ["neither", "did not start in Tema"],
                    "did not end in Tema")
    print("Dropped trips by reason:", pd.Series(why).value_counts().to_dict())
    lost = (dropped.groupby("boat_id").size() / (dropped.groupby("boat_id").size()
            .add(tracks.groupby("boat_id").size(), fill_value=0)) * 100).round(1).sort_values(ascending=False)
    print("Share of each boat's pings dropped (%):", lost[lost > 0].to_dict())


if __name__ == "__main__":
    main()
