"""Fill gaps in the Meteosat-12 SST with RSS MW_IR optimally interpolated SST.

Source: Remote Sensing Systems MW_IR OI SST v05.1, daily L4 GHRSST foundation SST on a 9 km grid
(microwave + infrared, gap-free). Open access at https://data.remss.com/SST/daily/mw_ir/v05.1/netcdf/
(no account needed).

Pings that need filling: no Meteosat-12 SST value, or Meteosat quality_level below MIN_QUALITY.
Every other ping keeps its Meteosat-12 value.

Note: MW_IR is a daily foundation SST (no daytime warming); Meteosat-12 is hourly sub-skin SST.
The two can differ by a few tenths of a degree in the daytime, so sst_source records which one was used.

Step 1 (download): for each day that has a ping to fill, download the global daily file, crop it to the
Ghana box, save the crop to data/sst_mwir/ and delete the full file. Crops on disk are skipped.
Step 2 (match): each ping to fill gets the nearest 9 km cell for its UTC day.

Input:  pipeline_outputs/environment/ping_sst_meteosat12.csv (from match_sst_meteosat12.py)
Output: pipeline_outputs/environment/ping_sst_combined.csv

Usage: /usr/bin/python3 pipeline_scripts/match_sst_mwir.py [--download] [--match]
(no flag runs both steps)
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

# ─────────────────────────────────────────────
#  PATHS AND SETTINGS
# ─────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parents[1]
METEOSAT_CSV = PROJECT_ROOT / "pipeline_outputs" / "environment" / "ping_sst_meteosat12.csv"
SST_DIR = PROJECT_ROOT / "data" / "sst_mwir"                 # cropped daily files (git-ignored *.nc)
OUTPUT_CSV = PROJECT_ROOT / "pipeline_outputs" / "environment" / "ping_sst_combined.csv"

BASE_URL = "https://data.remss.com/SST/daily/mw_ir/v05.1/netcdf"
FILE_NAME = "{day:%Y%m%d}120000-REMSS-L4_GHRSST-SSTfnd-MW_IR_OI-GLOB-v02.0-fv05.1.nc"
LAT_RANGE = (2.0, 7.0)              # same crop box as the Meteosat-12 files
LON_RANGE = (-6.0, 2.0)
MIN_QUALITY = 3                     # Meteosat quality_level below this (or no value) is filled from MW_IR
DOWNLOAD_WORKERS = 4
DOWNLOAD_RETRIES = 2
NETCDF_LOCK = threading.Lock()


# ─────────────────────────────────────────────
#  DOWNLOAD AND CROP
# ─────────────────────────────────────────────

def crop_path(day: pd.Timestamp) -> Path:
    return SST_DIR / f"{day:%Y%m%d}_mwir_sst.nc"


def fetch_and_crop(day: pd.Timestamp) -> None:
    """Download one global daily file, keep SST, its error estimate and the mask inside the crop box."""
    url = f"{BASE_URL}/{day:%Y}/{FILE_NAME.format(day=day)}"
    with tempfile.TemporaryDirectory() as tmp:
        full = Path(tmp) / "full.nc"
        with urllib.request.urlopen(url, timeout=300) as src, open(full, "wb") as dst:
            shutil.copyfileobj(src, dst)
        part = crop_path(day).with_suffix(".part")
        with NETCDF_LOCK:                                          # the netCDF library is not thread-safe
            with xr.open_dataset(full) as ds:
                crop = ds[["analysed_sst", "analysis_error", "mask"]].sel(
                    lat=slice(*LAT_RANGE), lon=slice(*LON_RANGE)).load()
            crop.to_netcdf(part, encoding={v: {"zlib": True, "complevel": 4} for v in crop.data_vars})
        part.rename(crop_path(day))                                # only complete crops get the final name


def download_days(days: list[pd.Timestamp]) -> None:
    SST_DIR.mkdir(parents=True, exist_ok=True)
    for stale in SST_DIR.glob("*.part"):
        stale.unlink()
    todo = [d for d in days if not crop_path(d).exists()]
    print(f"Days needed: {len(days)}; already cropped: {len(days) - len(todo)}; to download: {len(todo)}", flush=True)
    failed = []
    for attempt in range(1 + DOWNLOAD_RETRIES):
        failed = []
        with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(fetch_and_crop, d): d for d in todo}
            for done, future in enumerate(as_completed(futures), start=1):
                try:
                    future.result()
                except Exception as e:                             # network errors, missing days
                    failed.append(futures[future])
                    print(f"  failed {futures[future]:%Y-%m-%d}: {type(e).__name__}: {e}", flush=True)
                if done % 25 == 0 or done == len(futures):
                    print(f"  attempt {attempt + 1}: {done}/{len(futures)} done", flush=True)
        if not failed:
            break
        todo = failed
    print(f"Download done; failed after retries: {len(failed)}", flush=True)
    if failed:
        pd.DataFrame({"day": failed}).to_csv(SST_DIR / "missing_days.csv", index=False)


# ─────────────────────────────────────────────
#  MATCH TO PINGS
# ─────────────────────────────────────────────

def needs_fill(pings: pd.DataFrame) -> pd.Series:
    return pings["sst_c"].isna() | (pings["quality_level"] < MIN_QUALITY)


def match_pings(pings: pd.DataFrame) -> pd.DataFrame:
    out = pings.copy()
    out["mwir_day"] = out["timestamp"].dt.normalize()
    out["sst_mwir_c"] = np.nan
    out["mwir_error_c"] = np.nan
    fill = needs_fill(out)
    for day, rows in out[fill].groupby("mwir_day").groups.items():
        path = crop_path(day)
        if not path.exists():
            continue
        with xr.open_dataset(path) as ds:
            ds = ds.isel(time=0)
            lat_i = np.abs(ds.lat.values[None, :] - out.loc[rows, "lat"].values[:, None]).argmin(axis=1)
            lon_i = np.abs(ds.lon.values[None, :] - out.loc[rows, "lon"].values[:, None]).argmin(axis=1)
            out.loc[rows, "sst_mwir_c"] = ds["analysed_sst"].values[lat_i, lon_i] - 273.15   # NaN on land cells
            out.loc[rows, "mwir_error_c"] = ds["analysis_error"].values[lat_i, lon_i]
    # Best available value: Meteosat-12 unless it needed filling, then MW_IR
    out["sst_best_c"] = np.where(fill, out["sst_mwir_c"], out["sst_c"])
    out["sst_source"] = np.where(~fill, "meteosat12", np.where(out["sst_mwir_c"].notna(), "mw_ir", pd.NA))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--match", action="store_true")
    args = parser.parse_args()
    both = not (args.download or args.match)

    pings = pd.read_csv(METEOSAT_CSV, parse_dates=["timestamp", "sst_hour"])
    fill = needs_fill(pings)
    days = [pd.Timestamp(d) for d in sorted(pings.loc[fill, "timestamp"].dt.normalize().unique())]
    print(f"Pings: {len(pings):,}; to fill from MW_IR: {fill.sum():,} on {len(days)} days")
    if args.download or both:
        download_days(days)
    if args.match or both:
        out = match_pings(pings)
        out.to_csv(OUTPUT_CSV, index=False)
        print(f"Saved {OUTPUT_CSV}")
        print("sst_source counts:", out["sst_source"].value_counts(dropna=False).to_dict())
        print("MW_IR SST filled (deg C):", out.loc[out["sst_source"] == "mw_ir", "sst_mwir_c"].describe().round(2).to_dict())


if __name__ == "__main__":
    main()
