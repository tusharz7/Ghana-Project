"""Match Meteosat-12 (MTG-I1) FCI sea surface temperature to every GPS ping in trip_tracks.csv.

Source: OSI SAF OSI-206-b, hourly L3C GHRSST sub-skin SST on a 0.05 deg grid (EUMETSAT Data Store
collection EO:EUM:DAT:0694). Pre-operational product: every pixel is kept, and pixels below MIN_QUALITY
are marked low_quality = True.

Step 1 (download): for each hour that has a ping, download that hour's file, crop it to the Ghana box,
save the crop to data/sst/ and delete the full file. Crops already on disk are skipped, so the script
can be stopped and restarted.
Step 2 (match): each ping gets the SST of the nearest grid cell in the file for its nearest hour.

Needs a free EUMETSAT account: put the Data Store API key in ~/.eumdac/credentials as one line
"consumer_key,consumer_secret" (or set EUMDAC_KEY and EUMDAC_SECRET).

Usage: /usr/bin/python3 pipeline_scripts/match_sst_meteosat12.py [--download] [--match] [--dry-run]
(no flag runs both steps)
"""

from __future__ import annotations

import argparse
import os
import threading
import shutil
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKS_CSV = PROJECT_ROOT / "pipeline_outputs" / "gps_tracks" / "trip_tracks.csv"
SST_DIR = PROJECT_ROOT / "data" / "sst_meteosat12"          # cropped hourly files (git-ignored *.nc)
OUTPUT_CSV = PROJECT_ROOT / "pipeline_outputs" / "environment" / "ping_sst_meteosat12.csv"
CREDENTIALS = Path.home() / ".eumdac" / "credentials"

COLLECTION = "EO:EUM:DAT:0694"      # OSI-206-b, Meteosat-12 FCI L3C SST, hourly
LAT_RANGE = (2.0, 7.0)              # crop box: covers all pings (2.56-5.77 N, 5.59 W-1.66 E)
LON_RANGE = (-6.0, 2.0)
DOWNLOAD_WORKERS = 6                # files downloaded at once
DOWNLOAD_RETRIES = 2                # extra passes over hours that failed
NETCDF_LOCK = threading.Lock()
MIN_QUALITY = 4                     # GHRSST quality_level: 4 = acceptable, 5 = best; lower is kept but flagged low_quality


# ─────────────────────────────────────────────
#  DOWNLOAD AND CROP
# ─────────────────────────────────────────────

def data_store():
    """Log in to the EUMETSAT Data Store with the saved API key."""
    import eumdac

    key, secret = os.environ.get("EUMDAC_KEY"), os.environ.get("EUMDAC_SECRET")
    if not (key and secret):
        if not CREDENTIALS.exists():
            raise SystemExit(f"No EUMETSAT API key: put 'key,secret' in {CREDENTIALS} or set EUMDAC_KEY/EUMDAC_SECRET")
        key, secret = CREDENTIALS.read_text().strip().split(",")
    return eumdac.DataStore(eumdac.AccessToken((key, secret)))


def crop_path(hour: pd.Timestamp) -> Path:
    return SST_DIR / f"{hour:%Y%m%d%H}_meteosat12_sst.nc"


def crop_to_ghana(nc_file: Path, out_file: Path) -> None:
    """Keep SST, its bias estimate and quality level inside the crop box; drop the rest."""
    with xr.open_dataset(nc_file) as ds:
        lat_ok = (ds.lat >= LAT_RANGE[0]) & (ds.lat <= LAT_RANGE[1])
        lon_ok = (ds.lon >= LON_RANGE[0]) & (ds.lon <= LON_RANGE[1])
        keep = [v for v in ("sea_surface_temperature", "sses_bias", "quality_level") if v in ds]
        crop = ds[keep].isel(lat=np.flatnonzero(lat_ok.values), lon=np.flatnonzero(lon_ok.values)).load()
    encoding = {v: {"zlib": True, "complevel": 4} for v in crop.data_vars}
    crop.to_netcdf(out_file, encoding=encoding)


def product_hour(product) -> pd.Timestamp:
    """Nominal hour from the product name, e.g. 20260115100000-OSISAF-L3C_GHRSST-... -> 2026-01-15 10:00."""
    return pd.Timestamp(str(product)[:14])


def fetch_and_crop(product, hour: pd.Timestamp) -> None:
    """Download one hourly file to a temporary folder, crop it, save the crop and delete the full file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with product.open() as src, open(tmp / src.name, "wb") as dst:
            shutil.copyfileobj(src, dst)
            name = src.name
        if zipfile.is_zipfile(tmp / name):
            with zipfile.ZipFile(tmp / name) as z:
                z.extractall(tmp)
        nc = next(tmp.rglob("*.nc"))
        part = crop_path(hour).with_suffix(".part")
        with NETCDF_LOCK:                                          # the netCDF library is not thread-safe
            crop_to_ghana(nc, part)
        part.rename(crop_path(hour))                               # only complete crops get the final name


def download_hours(hours: list[pd.Timestamp]) -> None:
    store = data_store()
    collection = store.get_collection(COLLECTION)
    SST_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SST_DIR.glob("*.part"):
        stale.unlink()                                             # half-written crops from an interrupted run
    todo = [h for h in hours if not crop_path(h).exists()]
    print(f"Hours with pings: {len(hours)}; already cropped: {len(hours) - len(todo)}; to download: {len(todo)}",
          flush=True)

    # Find each wanted hour's product: one search per day
    jobs, missing = [], []
    for day, day_hours in pd.Series(todo).groupby(pd.Series(todo).dt.date):
        start = pd.Timestamp(day)
        products = collection.search(dtstart=start.to_pydatetime(), dtend=(start + timedelta(days=1)).to_pydatetime())
        found = {product_hour(p): p for p in products}
        for hour in day_hours:
            (jobs.append((found[hour], hour)) if hour in found else missing.append(hour))
    print(f"Products found: {len(jobs)}; hours with no product: {len(missing)}", flush=True)

    # Download in parallel; failed hours are retried, then listed
    failed = []
    for attempt in range(1 + DOWNLOAD_RETRIES):
        failed = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(fetch_and_crop, product, hour): (product, hour) for product, hour in jobs}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    future.result()
                except Exception as e:                             # network errors, bad files
                    failed.append(futures[future])
                    print(f"  failed {futures[future][1]}: {type(e).__name__}: {e}", flush=True)
                if done % 100 == 0 or done == len(futures):
                    print(f"  attempt {attempt + 1}: {done}/{len(futures)} done", flush=True)
        if not failed:
            break
        jobs = failed
    print(f"Download done; no product: {len(missing)}; failed after retries: {len(failed)}", flush=True)
    if missing or failed:
        pd.DataFrame({"hour": missing + [h for _, h in failed],
                      "reason": ["no product"] * len(missing) + ["download failed"] * len(failed)}
                     ).to_csv(SST_DIR / "missing_hours.csv", index=False)


# ─────────────────────────────────────────────
#  MATCH TO PINGS
# ─────────────────────────────────────────────

def match_pings(tracks: pd.DataFrame) -> pd.DataFrame:
    """Nearest hour, nearest 0.05 deg cell; all pixels kept, low_quality = quality_level < MIN_QUALITY."""
    out = tracks[["track_id", "boat_id", "trip_index", "segment", "timestamp", "lat", "lon"]].copy()
    out["sst_hour"] = out["timestamp"].dt.round("h")
    out["sst_c"] = np.nan              # bias-corrected: sea_surface_temperature - sses_bias
    out["sst_raw_c"] = np.nan          # as delivered, no bias correction
    out["quality_level"] = np.nan
    for hour, rows in out.groupby("sst_hour").groups.items():
        path = crop_path(hour)
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            ds = ds.isel(time=0) if "time" in ds.dims else ds
            lat_i = np.abs(ds.lat.values[None, :] - out.loc[rows, "lat"].values[:, None]).argmin(axis=1)
            lon_i = np.abs(ds.lon.values[None, :] - out.loc[rows, "lon"].values[:, None]).argmin(axis=1)
            sst = ds["sea_surface_temperature"].values[lat_i, lon_i]          # kelvin (scale applied by xarray)
            bias = ds["sses_bias"].values[lat_i, lon_i] if "sses_bias" in ds else np.zeros(len(rows))
            quality = ds["quality_level"].values[lat_i, lon_i].astype(float)
        out.loc[rows, "quality_level"] = quality
        out.loc[rows, "sst_raw_c"] = sst - 273.15                            # every pixel kept; NaN only where no SST
        out.loc[rows, "sst_c"] = sst - np.nan_to_num(bias) - 273.15
    # Pixels below MIN_QUALITY are kept but flagged; empty when there is no SST value at all
    has_sst = out["sst_raw_c"].notna()
    out["low_quality"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[has_sst, "low_quality"] = out.loc[has_sst, "quality_level"] < MIN_QUALITY
    out["minutes_from_sst_hour"] = (out["timestamp"] - out["sst_hour"]).dt.total_seconds() / 60
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--match", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only count the hours needed")
    args = parser.parse_args()
    both = not (args.download or args.match)

    tracks = pd.read_csv(TRACKS_CSV, parse_dates=["timestamp"])
    hours = sorted(tracks["timestamp"].dt.round("h").unique())
    hours = [pd.Timestamp(h) for h in hours]
    print(f"Pings: {len(tracks):,}; {tracks.timestamp.min()} to {tracks.timestamp.max()}; "
          f"{len(hours):,} hours on {len({h.date() for h in hours})} days")
    if args.dry_run:
        print(f"Already cropped: {sum(crop_path(h).exists() for h in hours)}")
        return
    if args.download or both:
        download_hours(hours)
    if args.match or both:
        matched = match_pings(tracks)
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        matched.to_csv(OUTPUT_CSV, index=False)
        have_file = matched["sst_hour"].map(lambda h: crop_path(h).exists())
        print(f"Saved {OUTPUT_CSV}")
        has_sst = matched["sst_c"].notna()
        print(f"Pings with an SST file for their hour: {have_file.mean():.1%}; with an SST value: {has_sst.mean():.1%} "
              f"(good quality: {(has_sst & ~matched['low_quality'].fillna(True)).mean():.1%}, "
              f"low_quality: {matched['low_quality'].fillna(False).mean():.1%})")
        good = matched.loc[matched["low_quality"] == False, "sst_c"]   # noqa: E712 (nullable boolean)
        print("Good-quality SST (deg C, bias-corrected):", good.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
