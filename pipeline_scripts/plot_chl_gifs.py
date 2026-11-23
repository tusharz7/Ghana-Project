"""Animated map of daily NASA PACE chlorophyll-a over the Ghana box (same layout as the SST GIFs).

Input:  data/chl_pace/PACE_OCI.<YYYYMMDD>.L3m.DAY.BGC.V3_2.4km.ghana.nc (daily crops from match_chl_pace.py)
Output: result_figures/chl_gifs/chl_pace_daily.gif

Each frame is one UTC day of the PACE OCI L3M daily 4 km product, chlor_a as delivered (no 8-day fill).
Grey = no value that day (cloud, Harmattan dust, glint, land). Daily files were only downloaded for days with
canoe pings, so days without pings (all of March 2026) are missing.

Usage: /usr/bin/python3 pipeline_scripts/plot_chl_gifs.py
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import LogNorm
from PIL import Image

from plot_sst_gifs import NATURAL_EARTH_COASTLINE, TEMA

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHL_DIR = PROJECT_ROOT / "data" / "chl_pace"
OUT_GIF = PROJECT_ROOT / "result_figures" / "chl_gifs" / "chl_pace_daily.gif"

VMIN, VMAX = 0.05, 10               # fixed log colour scale (mg m^-3); values outside are clipped to the end colours
FPS = 10


def render_frame(chl, lat, lon, day, coastline) -> Image.Image:
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=90)
    ax.set_facecolor("#d9d9d9")
    mesh = ax.pcolormesh(lon, lat, np.clip(chl, VMIN, VMAX), cmap="viridis",
                         norm=LogNorm(vmin=VMIN, vmax=VMAX), shading="nearest")
    coastline.plot(ax=ax, color="black", linewidth=0.8)
    ax.plot(*TEMA, marker="*", color="white", markeredgecolor="black", markersize=12)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"PACE OCI chlorophyll-a, daily 4 km   {day:%Y-%m-%d}\n"
                 f"{np.isfinite(chl).mean():.1%} of box has a value", fontsize=10)
    bar = fig.colorbar(mesh, ax=ax, label="Chlorophyll-a (mg m$^{-3}$, log scale)", shrink=0.85)
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
    bar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    bar.ax.minorticks_off()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    files = sorted(f for f in CHL_DIR.glob("PACE_OCI.*.L3m.DAY.*.ghana.nc") if not f.name.startswith("._"))
    lat = lon = coastline = None
    frames, all_values = [], []
    for f in files:
        day = pd.Timestamp(f.name.split(".")[1])
        with xr.open_dataset(f) as ds:
            chl = ds["chlor_a"].values
            if lat is None:
                lat, lon = ds.lat.values, ds.lon.values
                coastline = gpd.read_file(NATURAL_EARTH_COASTLINE,
                                          bbox=(lon.min(), lat.min(), lon.max(), lat.max()))
        all_values.append(chl[np.isfinite(chl)])
        frames.append(render_frame(chl, lat, lon, day, coastline))
    palette_src = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames = [fr.quantize(palette=palette_src, dither=Image.Dither.NONE) for fr in frames]
    OUT_GIF.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT_GIF, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
    values = np.concatenate(all_values)
    print(f"Saved {OUT_GIF} ({len(frames)} frames, {OUT_GIF.stat().st_size / 1e6:.1f} MB)")
    print(f"chlor_a 1st-99th pct {np.percentile(values, 1):.3f}-{np.percentile(values, 99):.2f} mg m^-3; "
          f"outside {VMIN}-{VMAX}: {((values < VMIN) | (values > VMAX)).mean():.2%}")


if __name__ == "__main__":
    main()
