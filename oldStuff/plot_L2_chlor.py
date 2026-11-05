"""Plot chlorophyll-a swath from a NASA Ocean Color L2 OLCI netCDF file."""
import netCDF4 as nc
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

FPATH = "/Users/tushy/Downloads/S3A_OLCI_EFRNT.20250121T001617.L2.OC.nc"

ds = nc.Dataset(FPATH)
lon = ds.groups["navigation_data"].variables["longitude"][:]
lat = ds.groups["navigation_data"].variables["latitude"][:]
chlor_a = ds.groups["geophysical_data"].variables["chlor_a"][:]
ds.close()

chlor_a = np.ma.masked_invalid(chlor_a)
chlor_a = np.ma.masked_less_equal(chlor_a, 0)

fig = plt.figure(figsize=(10, 8))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_feature(cfeature.LAND, facecolor="lightgray")
ax.add_feature(cfeature.COASTLINE)
ax.gridlines(draw_labels=True, linestyle="--", alpha=0.5)

mesh = ax.pcolormesh(
    lon, lat, chlor_a,
    transform=ccrs.PlateCarree(),
    cmap="viridis",
    norm=plt.matplotlib.colors.LogNorm(vmin=0.01, vmax=20),
    shading="auto",
)
cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", pad=0.05, shrink=0.8)
cbar.set_label("Chlorophyll-a (mg m$^{-3}$)")

ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()], crs=ccrs.PlateCarree())
ax.set_title("S3A OLCI L2 Chlorophyll-a\n2025-01-21T00:16:17Z")

plt.tight_layout()
plt.savefig("/Users/tushy/Documents/Research/Ghana Project/oldStuff/S3A_OLCI_chlor_a_20250121.png", dpi=200)
plt.show()
