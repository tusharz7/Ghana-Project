import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  LOAD DEPTH GRID
# ─────────────────────────────────────────────

def loadDepthGrid(cross_ref_folder: Path):
    xc    = np.fromfile(cross_ref_folder / "XC.data",    dtype=">f4").reshape(385, 1376)
    depth = np.fromfile(cross_ref_folder / "Depth.data", dtype=">f4").reshape(385, 1376)

    lons      = xc[0]
    lat_step  = 0.015625
    lat_start = 5.0 - 239 * lat_step
    lats      = lat_start + np.arange(385) * lat_step

    return lats, lons, depth


def lookupDepth(lat: float, lon: float,
                lats: np.ndarray, lons: np.ndarray,
                depth_grid: np.ndarray) -> float:
    if (lat < lats.min() or lat > lats.max() or
            lon < lons.min() or lon > lons.max()):
        return np.nan
    i_lat = int(np.argmin(np.abs(lats - lat)))
    i_lon = int(np.argmin(np.abs(lons - lon)))
    return float(depth_grid[i_lat, i_lon])


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def loadData(folder: Path, lats, lons, depth_grid) -> pd.DataFrame:
    allData = []

    for i, file_path in enumerate(folder.glob("*.xlsx")):
        try:
            df = pd.read_excel(file_path, header=4)
            if len(df) >= 100:
                df.columns = ["events", "timestamp", "address",
                              "lat_lng", "speed", "heading",
                              "altitude", "via", "drivers"]

                df = df[df["timestamp"] != "Date"].copy()
                df = df.dropna(subset=["lat_lng", "speed"])

                lat_lon   = df["lat_lng"].str.split(",", expand=True)
                df["lat"] = pd.to_numeric(lat_lon[0].str.strip(), errors="coerce")
                df["lon"] = pd.to_numeric(lat_lon[1].str.strip(), errors="coerce")
                df = df.dropna(subset=["lat", "lon"])

                df["speed"]     = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
                df["file_id"]   = i + 1
                df["file_name"] = file_path.stem   # ← store fisher name

                # Look up depth for every GPS point
                df["depth_m"] = df.apply(
                    lambda row: lookupDepth(row["lat"], row["lon"], lats, lons, depth_grid),
                    axis=1
                )

                df = df.sort_values("timestamp")
                df["time_diff"]   = df["timestamp"].diff()
                df["new_segment"] = df["time_diff"] > pd.Timedelta(hours=2)
                df["segment_id"]  = df["new_segment"].cumsum()

                allData.append(df[["file_id", "file_name", "timestamp",
                                   "lat", "lon", "speed", "heading",
                                   "depth_m", "segment_id"]])

        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    combined = pd.concat(allData, ignore_index=True)
    combined.sort_values(["file_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extractFeatures(combined: pd.DataFrame) -> pd.DataFrame:
    records = []
    for fid, grp in combined.groupby("file_id"):
        grp = grp.sort_values("timestamp")

        # Total time at sea in hours
        total_hours  = (grp["timestamp"].max() - grp["timestamp"].min()).total_seconds() / 3600
        num_segments = grp["segment_id"].nunique()

        # Average trip duration = total time at sea / number of trip segments
        avg_trip_duration = total_hours / num_segments if num_segments > 0 else 0

        # Percent of timepoints in water deeper than 500m (off the continental shelf)
        valid_depth   = grp["depth_m"].dropna()
        pct_off_shelf = (valid_depth > 500).sum() / len(valid_depth) * 100 if len(valid_depth) > 0 else 0

        records.append({
            "file_id":           fid,
            "file_name":         grp["file_name"].iloc[0],
            "mean_lat":          grp["lat"].mean(),
            "mean_lon":          grp["lon"].mean(),
            "std_lat":           grp["lat"].std(),
            "std_lon":           grp["lon"].std(),
            "mean_speed":        grp["speed"].mean(),
            "max_speed":         grp["speed"].max(),
            "avg_trip_duration": avg_trip_duration,   # hours per trip segment
            "num_segments":      num_segments,         # number of trip segments
            "mean_depth":        grp["depth_m"].mean(),
            "max_depth":         grp["depth_m"].max(),
            "pct_off_shelf":     pct_off_shelf,        # % of time in water > 500m
        })

    return pd.DataFrame(records).set_index("file_id")


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

CLUSTER_COLORS = [
    "#e63946",  # red
    "#2a9d8f",  # teal
    "#f4a261",  # orange
    "#6a0dad",  # purple
    "#ffffff",  # white
    "#faff00",  # yellow
]


def plotClusteredTrajectories(combined: pd.DataFrame, labels_map: dict):
    imagery = GoogleTiles(style="satellite")
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(1, 1, 1, projection=imagery.crs)

    pad = 0.3
    ax.set_extent([
        combined["lon"].min() - pad, combined["lon"].max() + pad,
        combined["lat"].min() - pad, combined["lat"].max() + pad,
    ])
    ax.add_image(imagery, 10)

    legend_handles = {}
    for fid, grp in combined.groupby("file_id"):
        cluster = labels_map.get(fid, 0)
        color   = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        label   = f"Cluster {cluster + 1}"
        for _, seg in grp.groupby("segment_id"):
            line, = ax.plot(
                seg["lon"], seg["lat"],
                color=color, linewidth=1.5, alpha=0.7,
                transform=ccrs.PlateCarree(),
            )
            legend_handles[label] = line

    ax.legend(handles=list(legend_handles.values()),
              labels=list(legend_handles.keys()),
              fontsize=8, loc="upper left")
    plt.title("Fisher Trajectories – K-Means Clusters (k=5)")
    plt.tight_layout()
    plt.show()


def printClusterMembers(features: pd.DataFrame):
    """Print which fisher belongs to each cluster."""
    print("\n" + "=" * 50)
    print("CLUSTER MEMBERSHIP")
    print("=" * 50)
    for cluster_id, grp in features.groupby("cluster"):
        print(f"\nCluster {cluster_id + 1}  ({len(grp)} fishers):")
        for fid, row in grp.iterrows():
            print(f"  [{fid}] {row['file_name']}")
            print(f"        mean depth: {row['mean_depth']:.0f}m  |  "
                  f"max depth: {row['max_depth']:.0f}m  |  "
                  f"mean speed: {row['mean_speed']:.1f}  |  "
                  f"trip duration: {row['avg_trip_duration']:.1f} hours")
    print("=" * 50)


def plotClusterDepthProfile(features: pd.DataFrame):
    """Bar chart of mean depth per cluster."""
    cluster_depth = features.groupby("cluster")["mean_depth"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(
        [f"Cluster {c + 1}" for c in cluster_depth.index],
        cluster_depth.values,
        color=[CLUSTER_COLORS[c % len(CLUSTER_COLORS)] for c in cluster_depth.index],
        edgecolor="black", linewidth=0.5
    )
    ax.set_ylabel("Mean Depth (m)")
    ax.set_title("Average Fishing Depth by Cluster")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    fisher_folder    = Path("/Users/tushy/Documents/Research/Ghana Project/Exported Files")
    cross_ref_folder = Path("/Users/tushy/Documents/Research/Ghana Project/archive_old_code/cross_reference")

    # 1. Load depth grid
    lats, lons, depth_grid = loadDepthGrid(cross_ref_folder)

    # 2. Load fisher data with depth lookup
    combined = loadData(fisher_folder, lats, lons, depth_grid)
    print(f"Loaded {combined['file_id'].nunique()} trajectories "
          f"({len(combined):,} total timepoints).")

    # 3. Extract features (now includes depth)
    features = extractFeatures(combined)
    print(f"Feature matrix shape: {features.shape}")

    # 4. Select only numeric columns for clustering
    feature_cols = ["mean_lat", "mean_lon", "std_lat", "std_lon",
                    "mean_speed", "max_speed", "avg_trip_duration", 
                    "mean_depth", "max_depth", "pct_off_shelf"]

    scaler = StandardScaler()
    X = scaler.fit_transform(features[feature_cols].values)

    # 5. Run K-Means
    k = 5
    km     = KMeans(n_clusters=k, init="k-means++", n_init=10)
    labels = km.fit_predict(X)

    features["cluster"] = labels
    labels_map = dict(zip(features.index, labels))

    # 6. Print which fisher is in each cluster
    printClusterMembers(features)

    # 7. Visualise
    plotClusteredTrajectories(combined, labels_map)
    plotClusterDepthProfile(features)


if __name__ == "__main__":
    main()