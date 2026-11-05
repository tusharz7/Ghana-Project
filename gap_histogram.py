"""Histogram of time between consecutive GPS points, split by trip.

Trips are defined exactly as in fisherMetadata (per trip)/generate_fisher_metadata.py:
a trip is a run of pings outside the 1 km Tema-port radius. Gaps are only
measured between pings inside the same trip, so the multi-day gaps spent in
port are never counted. Gaps longer than MAX_GAP_MINUTES are dropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Geod

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent
FISHER_DEPTH_FOLDER = PROJECT_ROOT / "fisherDepth"
OUTPUT_PNG = PROJECT_ROOT / "gap_histogram.png"

TEMA_PORT_LAT = 5 + 38 / 60 + 14.6 / 3600
TEMA_PORT_LON = 0 + 0 / 60 + 58.5 / 3600
TEMA_PORT_RADIUS_KM = 1.0
MAX_GAP_MINUTES = 60

GEOD = Geod(ellps="WGS84")


# ─────────────────────────────────────────────
#  TRIP SPLITTING
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


def detect_tema_trips(in_port: np.ndarray) -> list[tuple[int, int]]:
    """Same logic as detect_tema_trips in generate_fisher_metadata.py.

    Returns (start, end) row indices, inclusive, for each trip.
    """
    trips, row = [], 0
    while row < len(in_port):
        if in_port[row]:
            while row < len(in_port) and in_port[row]:
                row += 1
            if row == len(in_port):
                break
            start = row
        else:
            start = row
        while row < len(in_port) and not in_port[row]:
            row += 1
        if row < len(in_port):
            end = row
            while row < len(in_port) and in_port[row]:
                row += 1
        else:
            end = len(in_port) - 1
        trips.append((start, end))
    return trips or [(0, len(in_port) - 1)]


# ─────────────────────────────────────────────
#  GAP COLLECTION
# ─────────────────────────────────────────────

def gaps_within_trips() -> tuple[np.ndarray, int]:
    """Return (gaps in minutes kept for histogram, number of gaps dropped as too long)."""
    kept, dropped = [], 0
    for path in sorted(FISHER_DEPTH_FOLDER.glob("*_depth.csv")):
        points = pd.read_csv(path, parse_dates=["timestamp"]).dropna(
            subset=["timestamp", "lat", "lon"]
        )
        points = points.sort_values("timestamp").reset_index(drop=True)
        in_port = (
            distance_from_port_km(points["lat"].to_numpy(), points["lon"].to_numpy())
            <= TEMA_PORT_RADIUS_KM
        )
        for start, end in detect_tema_trips(in_port):
            times = points["timestamp"].iloc[start:end + 1]
            gaps = times.diff().dropna().dt.total_seconds().to_numpy() / 60
            dropped += int((gaps > MAX_GAP_MINUTES).sum())
            kept.append(gaps[gaps <= MAX_GAP_MINUTES])
    return np.concatenate(kept), dropped


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def plot_histogram(gaps: np.ndarray, dropped: int) -> None:
    # One bin per whole minute, centred on the minute (0 = [-0.5, 0.5), 5 = [4.5, 5.5), ...)
    edges = np.arange(-0.5, MAX_GAP_MINUTES + 1.5, 1)
    counts, _ = np.histogram(gaps, bins=edges)
    centres = np.arange(0, MAX_GAP_MINUTES + 1)

    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, scale in ((ax_lin, "linear"), (ax_log, "log")):
        ax.bar(centres, counts, width=0.9, color="#4C72B0", edgecolor="white")
        ax.bar(centres[[5, 10]], counts[[5, 10]], width=0.9, color="#DD8452", edgecolor="white")
        ax.set_xticks(range(0, MAX_GAP_MINUTES + 1, 5))
        ax.set_xlabel("Time between consecutive GPS points (minutes)")
        ax.set_ylabel("Number of gaps")
        ax.set_yscale(scale)
        ax.set_title(f"{scale.capitalize()} count scale")
        ax.grid(axis="y", alpha=0.3)

    total = counts.sum()
    share_5 = counts[5] / total * 100
    share_10 = counts[10] / total * 100
    ax_lin.set_title(
        f"Time between GPS points, within trips (n={total:,})\n"
        f"5 min: {share_5:.1f}%   10 min: {share_10:.1f}%   (orange = 5 and 10 min bins)"
    )
    ax_log.set_title(f"Same data, log count scale (gaps > {MAX_GAP_MINUTES} min dropped: {dropped:,})")

    fig.tight_layout()
    fig.savefig(OUTPUT_PNG, dpi=150)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Gaps used: {total:,}   dropped (> {MAX_GAP_MINUTES} min): {dropped:,}")
    print(f"5 min bin: {counts[5]:,} ({share_5:.1f}%)   10 min bin: {counts[10]:,} ({share_10:.1f}%)")
    print(f"Most common whole-minute gap: {centres[counts.argmax()]} min")


def main() -> None:
    gaps, dropped = gaps_within_trips()
    plot_histogram(gaps, dropped)


if __name__ == "__main__":
    main()
