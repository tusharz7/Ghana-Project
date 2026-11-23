"""Same daily Meteosat-12 SST GIFs as plot_sst_gifs.py, with gaps filled from RSS MW_IR to see how they line up.

Input:  data/sst_meteosat12/*.nc (hourly Meteosat-12 crops), data/sst_mwir/*_mwir_sst.nc (daily MW_IR crops)
Output: result_figures/sst_gifs_mwir_fill/sst_quality_<cut>_mwir.gif

Each GIF fills its own grey pixels (no Meteosat value passing that GIF's quality cut that day) with the
MW_IR daily foundation SST from the nearest 9 km cell. MW_IR pixels use the same colour scale, drawn faded
with a dot pattern. Pixels MW_IR doesn't cover (within ~9 km of the coast) stay grey, and so do days with no
MW_IR file (MW_IR was only downloaded for days with pings that needed filling).

Usage: /usr/bin/python3 pipeline_scripts/plot_sst_gifs_mwir.py
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from PIL import Image

from plot_sst_gifs import FPS, NATURAL_EARTH_COASTLINE, QUALITY_CUTS, TEMA, VMAX, VMIN, daily_means

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MWIR_DIR = PROJECT_ROOT / "data" / "sst_mwir"
OUT_DIR = PROJECT_ROOT / "result_figures" / "sst_gifs_mwir_fill"
MWIR_ALPHA = 0.55


def mwir_on_grid(day, lat, lon):
    """MW_IR SST (deg C) for the day, put on the Meteosat 0.05 deg grid by nearest cell; None if no file."""
    path = MWIR_DIR / f"{day:%Y%m%d}_mwir_sst.nc"
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        ds = ds.isel(time=0) if "time" in ds.dims else ds
        lat_i = np.abs(ds.lat.values[None, :] - lat[:, None]).argmin(axis=1)
        lon_i = np.abs(ds.lon.values[None, :] - lon[:, None]).argmin(axis=1)
        return ds["analysed_sst"].values[np.ix_(lat_i, lon_i)] - 273.15


def render_frame(meteosat, mwir, lat, lon, day, n_hours, q, coastline) -> Image.Image:
    fill = np.isnan(meteosat) & np.isfinite(mwir) if mwir is not None else np.zeros_like(meteosat, bool)
    fig, ax = plt.subplots(figsize=(8, 5.4), dpi=90)
    ax.set_facecolor("#d9d9d9")
    mesh = ax.pcolormesh(lon, lat, meteosat, cmap="turbo", vmin=VMIN, vmax=VMAX, shading="nearest")
    if fill.any():
        ax.pcolormesh(lon, lat, np.where(fill, mwir, np.nan), cmap="turbo", vmin=VMIN, vmax=VMAX,
                      shading="nearest", alpha=MWIR_ALPHA)
        ax.contourf(lon, lat, fill.astype(float), levels=[0.5, 1.5], colors="none", hatches=[".."])
    coastline.plot(ax=ax, color="black", linewidth=0.8)
    ax.plot(*TEMA, marker="*", color="white", markeredgecolor="black", markersize=12)
    ax.set_xlim(lon.min(), lon.max())
    ax.set_ylim(lat.min(), lat.max())
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    label = "quality 5" if q == 5 else f"quality {q}+"
    mw_text = f"MW_IR fill (faded, dotted) {fill.mean():.0%}" if mwir is not None else "no MW_IR file for this day"
    ax.set_title(f"Meteosat-12 SST, {label}   {day:%Y-%m-%d}\n"
                 f"Meteosat {np.isfinite(meteosat).mean():.0%} of box ({n_hours} hourly files); {mw_text}",
                 fontsize=10)
    fig.colorbar(mesh, ax=ax, label="SST (°C)", shrink=0.85)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    days, hours, means, lat, lon = daily_means()
    mwir = {d: mwir_on_grid(d, lat, lon) for d in days}
    print(f"Days: {len(days)}; with an MW_IR file: {sum(v is not None for v in mwir.values())}")
    coastline = gpd.read_file(NATURAL_EARTH_COASTLINE, bbox=(lon.min(), lat.min(), lon.max(), lat.max()))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for q, name in QUALITY_CUTS.items():
        frames = [render_frame(means[q][d], mwir[d], lat, lon, d, hours[d], q, coastline) for d in days]
        palette_src = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        frames = [f.quantize(palette=palette_src, dither=Image.Dither.NONE) for f in frames]
        out = OUT_DIR / f"{name}_mwir.gif"
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / FPS), loop=0)
        # Where both exist: how far MW_IR is from Meteosat (same pixel, same day)
        diffs = np.concatenate([(mwir[d] - means[q][d])[np.isfinite(means[q][d]) & np.isfinite(mwir[d])]
                                for d in days if mwir[d] is not None])
        print(f"Saved {out} ({len(frames)} frames, {out.stat().st_size / 1e6:.1f} MB); "
              f"MW_IR minus Meteosat where both exist: mean {diffs.mean():+.2f} C, sd {diffs.std():.2f} C")


if __name__ == "__main__":
    main()
