"""Animated maps of daily Meteosat-12 SST over the Ghana box, one GIF per minimum quality level.

Input:  data/sst_meteosat12/<YYYYMMDDHH>_meteosat12_sst.nc (hourly crops from match_sst_meteosat12.py)
Output: result_figures/sst_gifs/sst_quality_5.gif, sst_quality_4plus.gif, sst_quality_3plus.gif, sst_quality_2plus.gif

Each frame is one UTC day: every 0.05 deg pixel shows the mean bias-corrected SST (sea_surface_temperature -
sses_bias) over that day's hours whose quality_level passes the cut. Grey = no passing value that day (cloud).
Hourly files exist only for hours with canoe pings, so days with no pings are missing, and the number of
hours averaged differs between days (shown on each frame). Quality 0 and 1 never occur, so there is no "1+" GIF.

Usage: /usr/bin/python3 pipeline_scripts/plot_sst_gifs.py
"""

from __future__ import annotations

import io
from collections import defaultdict
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SST_DIR = PROJECT_ROOT / "data" / "sst_meteosat12"
OUT_DIR = PROJECT_ROOT / "result_figures" / "sst_gifs"
NATURAL_EARTH_COASTLINE = (Path.home() / ".local/share/cartopy/shapefiles/natural_earth/physical"
                           / "ne_10m_coastline.shp")

QUALITY_CUTS = {5: "sst_quality_5", 4: "sst_quality_4plus", 3: "sst_quality_3plus", 2: "sst_quality_2plus"}
VMIN, VMAX = 24, 31                 # fixed colour scale (deg C) for every frame and GIF
FPS = 10
TEMA = (-0.0166, 5.6398)            # lon, lat of Tema port


def daily_means():
    """One pass over the hourly files: per day and quality cut, the sum and count of passing SST values."""
    files = sorted(f for f in SST_DIR.glob("*_meteosat12_sst.nc") if not f.name.startswith("._"))  # skip macOS ._ files on the SSD
    sums = {q: defaultdict(lambda: 0.0) for q in QUALITY_CUTS}
    counts = {q: defaultdict(lambda: 0) for q in QUALITY_CUTS}
    hours = defaultdict(int)
    lat = lon = None
    for i, f in enumerate(files, start=1):
        day = pd.Timestamp(f.name[:8])
        with xr.open_dataset(f) as ds:
            ds = ds.isel(time=0) if "time" in ds.dims else ds
            sst = ds["sea_surface_temperature"].values - np.nan_to_num(ds["sses_bias"].values) - 273.15
            quality = ds["quality_level"].values
            if lat is None:
                lat, lon = ds.lat.values, ds.lon.values
        hours[day] += 1
        for q in QUALITY_CUTS:
            ok = np.isfinite(sst) & (quality >= q)
            sums[q][day] = sums[q][day] + np.where(ok, sst, 0.0)
            counts[q][day] = counts[q][day] + ok
        if i % 500 == 0:
            print(f"  read {i}/{len(files)} files", flush=True)
    days = sorted(hours)
    means = {q: {d: np.where(counts[q][d] > 0, sums[q][d] / np.maximum(counts[q][d], 1), np.nan) for d in days}
             for q in QUALITY_CUTS}
    return days, hours, means, lat, lon


def render_frame(grid, lat, lon, day, n_hours, q, coastline) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=90)
    ax.set_facecolor("#d9d9d9")
    mesh = ax.pcolormesh(lon, lat, grid, cmap="turbo", vmin=VMIN, vmax=VMAX, shading="nearest")
    coastline.plot(ax=ax, color="black", linewidth=0.8)
    ax.plot(*TEMA, marker="*", color="white", markeredgecolor="black", markersize=12)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    label = "quality 5" if q == 5 else f"quality {q}+"
    coverage = np.isfinite(grid).mean()
    ax.set_title(f"Meteosat-12 SST, {label}   {day:%Y-%m-%d}\n"
                 f"daily mean of {n_hours} hourly file(s); {coverage:.0%} of box has a value", fontsize=10)
    fig.colorbar(mesh, ax=ax, label="SST (°C, bias-corrected)", shrink=0.85)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    days, hours, means, lat, lon = daily_means()
    print(f"Days with files: {len(days)} ({days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d})")
    coastline = gpd.read_file(NATURAL_EARTH_COASTLINE, bbox=(lon.min(), lat.min(), lon.max(), lat.max()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q, name in QUALITY_CUTS.items():
        frames = [render_frame(means[q][d], lat, lon, d, hours[d], q, coastline) for d in days]
        # one shared palette keeps colours identical across frames
        palette_src = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        frames = [f.quantize(palette=palette_src, dither=Image.Dither.NONE) for f in frames]
        out = OUT_DIR / f"{name}.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
        values = np.concatenate([means[q][d][np.isfinite(means[q][d])] for d in days])
        print(f"Saved {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB); "
              f"SST 1st-99th pct {np.percentile(values, 1):.1f}-{np.percentile(values, 99):.1f} C, "
              f"outside 24-31: {((values < VMIN) | (values > VMAX)).mean():.1%}")


if __name__ == "__main__":
    main()
