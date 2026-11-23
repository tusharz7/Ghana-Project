"""Match NASA PACE OCI chlorophyll-a to every GPS ping in trip_tracks.csv.

Source: PACE OCI Level-3 Global Mapped Ocean Biogeochemical Properties, version 3.2 (PACE_OCI_L3M_BGC),
4 km grid, variable chlor_a (mg m^-3). Each ping gets the nearest cell from the daily file for its UTC day;
where that is empty (cloud, glint), it gets the nearest cell from the 8-day file that contains the day.
chl_source records which one was used.

Step 1 (download): for each needed file, download the global file, crop it to the Ghana box, save the crop
to data/chl_pace/ and delete the full file. Crops on disk are skipped.
Step 2 (match).

Needs a free NASA Earthdata login in ~/.netrc:
    machine urs.earthdata.nasa.gov login <user> password <pass>

Output: pipeline_outputs/environment/ping_chl_pace.csv

Usage: /usr/bin/python3 pipeline_scripts/match_chl_pace.py [--download] [--match]
(no flag runs both steps)
"""

from __future__ import annotations

import argparse
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_CSV = PROJECT_ROOT / "pipeline_outputs" / "gps_tracks" / "trip_tracks.csv"
CHL_DIR = PROJECT_ROOT / "data" / "chl_pace"                  # cropped files (git-ignored *.nc)
OUTPUT_CSV = PROJECT_ROOT / "pipeline_outputs" / "environment" / "ping_chl_pace.csv"

SHORT_NAME = "PACE_OCI_L3M_BGC"
VERSION = "3.2"
CMR_URL = "https://cmr.earthdata.nasa.gov/search/granules.json"
DOWNLOAD_URL = "https://obdaac-tea.earthdatacloud.nasa.gov/ob-cumulus-prod-public/{name}"
LAT_RANGE = (2.0, 7.0)              # same crop box as the SST files
LON_RANGE = (-6.0, 2.0)
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRIES = 2
NETCDF_LOCK = threading.Lock()


# ─────────────────────────────────────────────
#  FIND AND DOWNLOAD FILES
# ─────────────────────────────────────────────

def list_files(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """All 4 km DAY and 8D file names overlapping the period, with the dates each covers."""
    rows, page = [], 1
    while True:
        r = requests.get(CMR_URL, params={
            "short_name": SHORT_NAME, "version": VERSION, "page_size": 2000, "page_num": page,
            "temporal": f"{start:%Y-%m-%d}T00:00:00Z,{end:%Y-%m-%d}T23:59:59Z"}, timeout=120)
        r.raise_for_status()
        entries = r.json()["feed"]["entry"]
        for e in entries:
            name = e["title"]                      # e.g. PACE_OCI.20251008_20251015.L3m.8D.BGC.V3_2.4km.nc
            parts = name.split(".")
            if parts[-2] != "4km" or parts[3] not in ("DAY", "8D"):
                continue
            dates = parts[1].split("_")
            rows.append({"name": name, "period": parts[3],
                         "first": pd.Timestamp(dates[0]), "last": pd.Timestamp(dates[-1])})
        if len(entries) < 2000:
            break
        page += 1
    return pd.DataFrame(rows)


def crop_path(name: str) -> Path:
    return CHL_DIR / name.replace(".nc", ".ghana.nc")


def fetch_and_crop(session: requests.Session, name: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / name
        with session.get(DOWNLOAD_URL.format(name=name), stream=True, timeout=600) as r:
            r.raise_for_status()
            with open(full, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        part = crop_path(name).with_suffix(".part")
        with NETCDF_LOCK:                                          # the netCDF library is not thread-safe
            with xr.open_dataset(full) as ds:                      # lat runs north to south
                crop = ds[["chlor_a"]].sel(lat=slice(LAT_RANGE[1], LAT_RANGE[0]), lon=slice(*LON_RANGE)).load()
            crop.to_netcdf(part, encoding={"chlor_a": {"zlib": True, "complevel": 4}})
        part.rename(crop_path(name))


def download(names: list[str]) -> None:
    CHL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in CHL_DIR.glob("*.part"):
        stale.unlink()
    todo = [n for n in names if not crop_path(n).exists()]
    print(f"Files needed: {len(names)}; already cropped: {len(names) - len(todo)}; to download: {len(todo)}", flush=True)
    session = requests.Session()                                   # Earthdata login from ~/.netrc on redirect
    failed = []
    for attempt in range(1 + DOWNLOAD_RETRIES):
        failed = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(fetch_and_crop, session, n): n for n in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    future.result()
                except Exception as e:
                    failed.append(futures[future])
                    print(f"  failed {futures[future]}: {type(e).__name__}: {e}", flush=True)
                if done % 25 == 0 or done == len(futures):
                    print(f"  attempt {attempt + 1}: {done}/{len(futures)} done", flush=True)
        if not failed:
            break
        todo = failed
    print(f"Download done; failed after retries: {len(failed)}", flush=True)
    if failed:
        pd.DataFrame({"name": failed}).to_csv(CHL_DIR / "failed_files.csv", index=False)


# ─────────────────────────────────────────────
#  MATCH TO PINGS
# ─────────────────────────────────────────────

def sample(name: str, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    path = crop_path(name)
    if not path.exists():
        return np.full(len(lat), np.nan)
    with xr.open_dataset(path) as ds:
        lat_i = np.abs(ds.lat.values[None, :] - lat[:, None]).argmin(axis=1)
        lon_i = np.abs(ds.lon.values[None, :] - lon[:, None]).argmin(axis=1)
        return ds["chlor_a"].values[lat_i, lon_i]


def file_for_day(files: pd.DataFrame, period: str) -> dict:
    """Day -> file name for the given period (DAY or 8D)."""
    out = {}
    for f in files[files["period"] == period].itertuples():
        for day in pd.date_range(f.first, f.last):
            out[day] = f.name
    return out


def match_pings(tracks: pd.DataFrame, daily: dict, eight_day: dict) -> pd.DataFrame:
    out = tracks[["track_id", "boat_id", "trip_index", "segment", "timestamp", "lat", "lon"]].copy()
    out["chl_day"] = out["timestamp"].dt.normalize()
    out["chl_daily"] = np.nan
    out["chl_8day"] = np.nan
    for day, rows in out.groupby("chl_day").groups.items():
        lat, lon = out.loc[rows, "lat"].values, out.loc[rows, "lon"].values
        if day in daily:
            out.loc[rows, "chl_daily"] = sample(daily[day], lat, lon)
        if day in eight_day:
            out.loc[rows, "chl_8day"] = sample(eight_day[day], lat, lon)
    out["chlor_a"] = out["chl_daily"].fillna(out["chl_8day"])     # mg m^-3
    out["chl_source"] = np.where(out["chl_daily"].notna(), "daily",
                                 np.where(out["chl_8day"].notna(), "8day", pd.NA))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--match", action="store_true")
    args = parser.parse_args()
    both = not (args.download or args.match)

    tracks = pd.read_csv(TRACKS_CSV, parse_dates=["timestamp"])
    days = sorted(tracks["timestamp"].dt.normalize().unique())
    files = list_files(pd.Timestamp(days[0]), pd.Timestamp(days[-1]))
    daily, eight_day = file_for_day(files, "DAY"), file_for_day(files, "8D")
    needed = sorted({daily[d] for d in days if d in daily} | {eight_day[d] for d in days if d in eight_day})
    print(f"Pings: {len(tracks):,} on {len(days)} days; days with a daily file: {sum(d in daily for d in days)}, "
          f"with an 8-day file: {sum(d in eight_day for d in days)}")
    if args.download or both:
        download(needed)
    if args.match or both:
        out = match_pings(tracks, daily, eight_day)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {OUTPUT_CSV}")
        print("chl_source counts:", out["chl_source"].value_counts(dropna=False).to_dict())
        print("chlor_a (mg m^-3):", out["chlor_a"].describe().round(3).to_dict())


if __name__ == "__main__":
    main()
