"""Generate one trip-metadata CSV per fisherDepth file plus a combined CSV."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path
import re

import contourpy
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from pyproj import Geod, Transformer
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


# ─────────────────────────────────────────────
#  PROJECT PATHS AND TRIP PARAMETERS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FISHER_DEPTH_FOLDER = PROJECT_ROOT / "fisherDepth"
DEPTH_GRID_FOLDER = PROJECT_ROOT / "data"
OUTPUT_FOLDER = PROJECT_ROOT / "exploratory_analysis" / "fisherMetadata"
ENVIRONMENT_FOLDER = Path("/Volumes/Extreme SSD/Ghana Project data")

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600
TEMA_PORT_RADIUS_KM = 1.0
MAX_TRIP_GAP_DAYS = 4 # gaps between pings longer than this end the trip; the next ping starts a new one
FISHING_MIN_KNOTS = 0.0 # minimum speed considered as fishing
FISHING_MAX_KNOTS = 6.0 # maximum speed considered as fishing
OUTSIDE_SPEED_SPLIT_MINUTES = 3.0 # if the vessel is outside the fishing speed range for this many minutes, considered to be not fishing
STRAIGHTNESS_WINDOW_MINUTES = 30.0 # splitting straightness by every 30 minutes

GEOD = Geod(ellps="WGS84")
PROJECT_TO_METERS = Transformer.from_crs(
    "EPSG:4326",
    "+proj=aeqd +lat_0=5 +lon_0=0 +datum=WGS84 +units=m +no_defs",
    always_xy=True,
)

# Local Ghana monthly daylight estimates supplied for this project.
DAYLIGHT_HOURS = {
    1: ("06:05", "17:50"), 2: ("06:05", "18:00"), 3: ("05:55", "18:05"),
    4: ("05:45", "18:10"), 5: ("05:40", "18:15"), 6: ("05:45", "18:20"),
    7: ("05:50", "18:20"), 8: ("05:50", "18:10"), 9: ("05:45", "17:55"),
    10: ("05:45", "17:45"), 11: ("05:55", "17:40"), 12: ("06:05", "17:40"),
}
DAYLIGHT_HOURS = {
    month: (time(*map(int, sunrise.split(":"))), time(*map(int, sunset.split(":"))))
    for month, (sunrise, sunset) in DAYLIGHT_HOURS.items()
}


# ─────────────────────────────────────────────
#  COASTLINE AND DISTANCE HELPERS
# ─────────────────────────────────────────────

def build_coastline_index() -> tuple[STRtree, list[LineString]]:
    """Extract the zero-depth coastline as projected searchable line segments."""
    xc = np.fromfile(DEPTH_GRID_FOLDER / "XC.data", dtype=">f4").reshape(385, 1376)
    yc = np.fromfile(DEPTH_GRID_FOLDER / "YC.data", dtype=">f4").reshape(385, 1376)
    depth = np.fromfile(
        DEPTH_GRID_FOLDER / "Depth.data", dtype=">f4"
    ).reshape(385, 1376)
    contours = contourpy.contour_generator(
        x=xc[0], y=yc[:, 0], z=(depth > 0).astype("float32"), name="serial"
    ).lines(0.5)
    segments = []
    for contour in contours:
        if len(contour) < 2:
            continue
        x, y = PROJECT_TO_METERS.transform(contour[:, 0], contour[:, 1])
        points = list(zip(x, y))
        segments.extend(
            LineString([points[index], points[index + 1]])
            for index in range(len(points) - 1)
            if points[index] != points[index + 1]
        )
    if not segments:
        raise ValueError("No coastline could be extracted from the depth grid")
    return STRtree(segments), segments


def distance_from_port_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate geodesic distance from Tema Fishing Harbour."""
    _, _, distance_m = GEOD.inv(
        np.full(len(lat), TEMA_PORT_LON),
        np.full(len(lat), TEMA_PORT_LAT),
        lon,
        lat,
    )
    return np.asarray(distance_m) / 1000


def distance_from_shore_km(
    lat: np.ndarray, lon: np.ndarray, index: STRtree, segments: list[LineString]
) -> np.ndarray:
    """Calculate exact projected point-to-nearest-coastline-segment distance."""
    x, y = PROJECT_TO_METERS.transform(lon, lat)
    distances = np.empty(len(lat))
    for row, (point_x, point_y) in enumerate(zip(x, y)):
        point = Point(float(point_x), float(point_y))
        nearest = index.nearest(point)
        distances[row] = point.distance(segments[int(nearest)]) / 1000
    return distances


def consecutive_distance_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate great-circle distance between each consecutive GPS pair."""
    if len(lat) < 2:
        return np.array([])
    _, _, distance_m = GEOD.inv(lon[:-1], lat[:-1], lon[1:], lat[1:])
    return np.asarray(distance_m) / 1000


# ─────────────────────────────────────────────
#  DAILY ENVIRONMENTAL DATA
# ─────────────────────────────────────────────

def date_from_filename(path: Path) -> str | None:
    """Read either YYYY-MM-DD or YYYYMMDD date patterns from NetCDF filenames."""
    match = re.search(r"(20\d{2})-(\d{2})-(\d{2})", path.name)
    match = match or re.search(r"(20\d{2})(\d{2})(\d{2})", path.name)
    return "-".join(match.groups()) if match else None


def environmental_files(folder: str) -> dict[str, Path]:
    """Map each available environmental date to its daily NetCDF file."""
    return {
        date: path
        for path in (ENVIRONMENT_FOLDER / folder).glob("*.nc")
        if (date := date_from_filename(path))
    }


def nearest_index(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Return nearest coordinate index, including descending latitude arrays."""
    if values[0] > values[-1]:
        return len(values) - 1 - nearest_index(values[::-1], targets)
    positions = np.clip(np.searchsorted(values, targets), 1, len(values) - 1)
    return np.where(
        np.abs(targets - values[positions - 1]) <= np.abs(targets - values[positions]),
        positions - 1,
        positions,
    )


def sample_surface(
    path: Path, variable: str, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray:
    """Sample nearest same-day model surface cell, returning NaN if unreadable."""
    try:
        with Dataset(path) as dataset:
            yc = np.asarray(dataset.variables["YC"][:], dtype=float)
            xc = np.asarray(dataset.variables["XC"][:], dtype=float)
            values = np.ma.filled(dataset.variables[variable][0, 0], np.nan)
            return np.asarray(values, dtype=float)[
                nearest_index(yc, lat), nearest_index(xc, lon)
            ]
    except OSError:
        return np.full(len(lat), np.nan)


def sample_chlorophyll(path: Path, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Sample nearest same-day chlorophyll cell, returning NaN if unreadable."""
    try:
        with Dataset(path) as dataset:
            y = np.asarray(dataset.variables["lat"][:], dtype=float)
            x = np.asarray(dataset.variables["lon"][:], dtype=float)
            y_index = nearest_index(y, lat)
            x_index = nearest_index(x, lon)
            values = np.ma.filled(dataset.variables["chlor_a"][:], np.nan)
            return np.asarray(values, dtype=float)[y_index, x_index]
    except OSError:
        return np.full(len(lat), np.nan)


def enrich_environment(points: pd.DataFrame) -> pd.DataFrame:
    """Add same-date, nearest-cell surface temperature, salinity, and chlorophyll."""
    temperature_files = environmental_files("temperature data")
    salinity_files = environmental_files("salt data")
    chlorophyll_files = environmental_files("chlorophyll data")
    points = points.copy()
    points["date"] = points["timestamp"].dt.strftime("%Y-%m-%d")
    for column in ("temperature", "salinity", "chlorophyll"):
        points[column] = np.nan
    for date, rows in points.groupby("date").groups.items():
        subset = points.loc[rows]
        lat = subset["lat"].to_numpy()
        lon = subset["lon"].to_numpy()
        if date in temperature_files:
            points.loc[rows, "temperature"] = sample_surface(
                temperature_files[date], "THETA", lat, lon
            )
        if date in salinity_files:
            points.loc[rows, "salinity"] = sample_surface(
                salinity_files[date], "SALT", lat, lon
            )
        if date in chlorophyll_files:
            points.loc[rows, "chlorophyll"] = sample_chlorophyll(
                chlorophyll_files[date], lat, lon
            )
    return points.drop(columns="date")


# ─────────────────────────────────────────────
#  TRIP AND FISHING-EVENT DETECTION
# ─────────────────────────────────────────────

def detect_tema_trips(points: pd.DataFrame) -> list[tuple[int, int, bool]]:
    """Split tracks when leaving/re-entering the 1 km Tema-port radius."""
    in_port = points["distance_from_tema_port_km"].to_numpy() <= TEMA_PORT_RADIUS_KM
    trips, row = [], 0
    while row < len(points):
        if in_port[row]:
            while row < len(points) and in_port[row]:
                row += 1
            if row == len(points):
                break
        start = row
        while row < len(points) and not in_port[row]:
            row += 1
        if row < len(points):
            end, returned_to_tema = row, True
            while row < len(points) and in_port[row]:
                row += 1
        else:
            end, returned_to_tema = len(points) - 1, False
        trips.append((start, end, returned_to_tema))
    return trips or [(0, len(points) - 1, False)]


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


def daylight_hours(start: pd.Timestamp, end: pd.Timestamp) -> float:
    """Calculate overlap with the project monthly Ghana sunrise/sunset schedule."""
    total, day = 0.0, start.date()
    while day <= end.date():
        sunrise, sunset = DAYLIGHT_HOURS[day.month]
        overlap_start = max(start.to_pydatetime(), datetime.combine(day, sunrise))
        overlap_end = min(end.to_pydatetime(), datetime.combine(day, sunset))
        if overlap_end > overlap_start:
            total += (overlap_end - overlap_start).total_seconds() / 3600
        day += timedelta(days=1)
    return total


def fishing_events(
    trip: pd.DataFrame,
) -> tuple[list[tuple[pd.Timestamp, pd.Timestamp, float, float]], np.ndarray, float]:
    """Identify events, splitting only after speed exceeds 6 knots for >3 minutes."""
    times = trip["timestamp"].to_numpy()
    speed = pd.to_numeric(trip["speed"], errors="coerce").fillna(0).to_numpy()
    if len(trip) < 2:
        return [], np.zeros(len(trip), dtype=bool), 0.0
    hours = np.maximum(
        np.diff(times).astype("timedelta64[s]").astype(float) / 3600, 0
    )
    in_range = (speed[:-1] >= FISHING_MIN_KNOTS) & (speed[:-1] <= FISHING_MAX_KNOTS)
    fishing = in_range | ((~in_range) & (hours * 60 <= OUTSIDE_SPEED_SPLIT_MINUTES))
    events, active_start, duration, daytime = [], None, 0.0, 0.0
    for row, is_fishing in enumerate(fishing):
        if is_fishing:
            if active_start is None:
                active_start, duration, daytime = pd.Timestamp(times[row]), 0.0, 0.0
            duration += hours[row]
            daytime += daylight_hours(pd.Timestamp(times[row]), pd.Timestamp(times[row + 1]))
            active_end = pd.Timestamp(times[row + 1])
        elif active_start is not None:
            events.append((active_start, active_end, duration, daytime))
            active_start = None
    if active_start is not None:
        events.append((active_start, active_end, duration, daytime))
    fishing_points = np.zeros(len(trip), dtype=bool)
    for row, is_fishing in enumerate(fishing):
        if is_fishing:
            fishing_points[row:row + 2] = True
    return events, fishing_points, float(hours[fishing].sum())


# ─────────────────────────────────────────────
#  TRIP SUMMARY OUTPUT
# ─────────────────────────────────────────────

def stat(values: pd.Series | np.ndarray, method: str) -> float:
    """Return a NaN-safe numeric summary statistic."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan
    return float(getattr(np, method)(values))


def trip_metadata(
    trip: pd.DataFrame, number: int, returned: bool, end_reason: str
) -> dict:
    """Calculate requested behavior, depth, distance, and environmental metrics."""
    events, fishing_point_mask, fishing_hours = fishing_events(trip)
    fishing_points = trip.loc[fishing_point_mask]
    event_hours = np.array([event[2] for event in events])
    trip_hours = (trip["timestamp"].iloc[-1] - trip["timestamp"].iloc[0]).total_seconds() / 3600
    path_km = consecutive_distance_km(trip["lat"].to_numpy(), trip["lon"].to_numpy()).sum()
    final_window = trip[
        trip["timestamp"] >= trip["timestamp"].iloc[-1] - pd.Timedelta(minutes=STRAIGHTNESS_WINDOW_MINUTES)
    ]
    final_path = consecutive_distance_km(
        final_window["lat"].to_numpy(), final_window["lon"].to_numpy()
    ).sum()
    if len(final_window) > 1 and final_path > 0:
        _, _, direct_m = GEOD.inv(
            final_window["lon"].iloc[0], final_window["lat"].iloc[0],
            final_window["lon"].iloc[-1], final_window["lat"].iloc[-1],
        )
        straightness = direct_m / 1000 / final_path
    else:
        straightness = np.nan
    event_day_fraction = [
        DAYLIGHT_HOURS[event[0].month][0]
        <= (event[0] + (event[1] - event[0]) / 2).time()
        <= DAYLIGHT_HOURS[event[0].month][1]
        for event in events
    ]
    depth = fishing_points["depth_m"]
    return {
        "file_id": trip["file_id"].iloc[0], "file_name": trip["file_name"].iloc[0],
        "trip_index": number, "departure_time": trip["timestamp"].iloc[0],
        "return_time": trip["timestamp"].iloc[-1] if returned else "",
        "trip_end_time": trip["timestamp"].iloc[-1],
        "return_detected_to_tema": returned,
        "trip_end_reason": end_reason,
        "n_points": len(trip), "total_trip_duration_hours": trip_hours,
        "total_distance_traveled_km": path_km,
        "median_distance_from_shore_km": stat(trip["distance_from_shore_km"], "nanmedian"),
        "max_distance_from_shore_km": stat(trip["distance_from_shore_km"], "nanmax"),
        "p90_distance_from_shore_km": float(
            np.nanpercentile(trip["distance_from_shore_km"], 90)
        ),
        "max_distance_from_departure_port_km": stat(trip["distance_from_tema_port_km"], "nanmax"),
        "fraction_time_classified_as_fishing": fishing_hours / trip_hours if trip_hours else np.nan,
        "total_estimated_fishing_time_hours": fishing_hours,
        "number_of_individual_fishing_events": len(events),
        "median_fishing_event_duration_hours": stat(event_hours, "nanmedian"),
        "max_fishing_event_duration_hours": stat(event_hours, "nanmax"),
        "fraction_fishing_events_during_day": np.mean(event_day_fraction) if events else np.nan,
        "fraction_fishing_time_during_day": sum(event[3] for event in events) / fishing_hours if fishing_hours else np.nan,
        "median_fishing_depth_m": stat(depth, "nanmedian"), "min_fishing_depth_m": stat(depth, "nanmin"),
        "max_fishing_depth_m": stat(depth, "nanmax"),
        "range_fishing_depth_m": stat(depth, "nanmax") - stat(depth, "nanmin") if len(depth.dropna()) else np.nan,
        "std_fishing_depth_m": stat(depth, "nanstd"),
        "trip_straightness_last_30min": straightness,
        "median_temperature_during_fishing": stat(fishing_points["temperature"], "nanmedian"),
        "median_temperature_entire_trip": stat(trip["temperature"], "nanmedian"),
        "median_salinity_during_fishing": stat(fishing_points["salinity"], "nanmedian"),
        "median_salinity_entire_trip": stat(trip["salinity"], "nanmedian"),
        "median_chlorophyll_during_fishing": stat(fishing_points["chlorophyll"], "nanmedian"),
        "median_chlorophyll_entire_trip": stat(trip["chlorophyll"], "nanmedian"),
    }


def main() -> None:
    """Load all depth files, enrich them, detect trips, and write metadata CSVs."""
    coastline_index, coastline_segments = build_coastline_index()
    frames = []
    for path in sorted(FISHER_DEPTH_FOLDER.glob("*_depth.csv")):
        points = pd.read_csv(path, parse_dates=["timestamp"]).dropna(
            subset=["timestamp", "lat", "lon"]
        )
        points["source_csv"] = path.name
        frames.append(points.sort_values("timestamp"))
    points = pd.concat(frames, ignore_index=True)
    points["distance_from_tema_port_km"] = distance_from_port_km(
        points["lat"].to_numpy(), points["lon"].to_numpy()
    )
    points["distance_from_shore_km"] = distance_from_shore_km(
        points["lat"].to_numpy(), points["lon"].to_numpy(),
        coastline_index, coastline_segments,
    )
    points = enrich_environment(points)

    all_rows = []
    for source_csv, fisher in points.groupby("source_csv"):
        fisher = fisher.sort_values("timestamp").reset_index(drop=True)
        rows = [
            trip_metadata(fisher.iloc[start:end + 1].copy(), number, returned, end_reason)
            for number, (start, end, returned, end_reason)
            in enumerate(split_at_long_gaps(fisher["timestamp"].to_numpy(), detect_tema_trips(fisher)), start=1)
        ]
        pd.DataFrame(rows).to_csv(
            OUTPUT_FOLDER / source_csv.replace("_depth.csv", "_metadata.csv"),
            index=False,
        )
        all_rows.extend(rows)
    pd.DataFrame(all_rows).to_csv(
        OUTPUT_FOLDER / "all_fisher_trip_metadata.csv", index=False
    )
    print(f"Wrote metadata for {len(frames)} fishers to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
