import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles


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
            df.columns = ["events", "timestamp", "address",
                          "lat_lng", "speed", "heading",
                          "altitude", "via", "drivers"]

            df = df[df["timestamp"] != "Date"].copy()
            df = df.dropna(subset=["lat_lng", "speed"])

            lat_lon   = df["lat_lng"].str.split(",", expand=True)
            df["lat"] = pd.to_numeric(lat_lon[0].str.strip(), errors="coerce")
            df["lon"] = pd.to_numeric(lat_lon[1].str.strip(), errors="coerce")
            df = df.dropna(subset=["lat", "lon"])

            if df.shape[0] < 100:
                continue

            df["speed"]     = pd.to_numeric(df["speed"], errors="coerce").fillna(0)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df["file_id"]   = i + 1
            df["file_name"] = file_path.stem

            # Look up depth for every GPS point
            df["depth_m"] = df.apply(
                lambda row: lookupDepth(row["lat"], row["lon"], lats, lons, depth_grid),
                axis=1
            )

            df = df.sort_values("timestamp")
            df["time_diff"]   = df["timestamp"].diff()
            df["new_segment"] = df["time_diff"] > pd.Timedelta(hours=2)
            df["segment_id"]  = df["new_segment"].cumsum()

            allData.append(df[["file_id", "file_name", "timestamp", "lat", "lon",
                                "speed", "heading", "depth_m", "segment_id"]])

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

        records.append({
            "file_id":        fid,
            "file_name":      grp["file_name"].iloc[0],
            "mean_lat":       grp["lat"].mean(),
            "mean_lon":       grp["lon"].mean(),
            "std_lat":        grp["lat"].std(),
            "std_lon":        grp["lon"].std(),
            "mean_speed":     grp["speed"].mean(),
            "max_speed":      grp["speed"].max(),
            "total_distance": grp["speed"].sum(),
            "num_segments":   grp["segment_id"].nunique(),
            "mean_depth":     grp["depth_m"].mean(),   
            "max_depth":      grp["depth_m"].max(),
            "min_depth":      grp["depth_m"].min(),
        })

    return pd.DataFrame(records).set_index("file_id")


# ─────────────────────────────────────────────
#  INITIAL CENTROID SELECTION
# ─────────────────────────────────────────────

def initialCentroids(X: np.ndarray, k: int) -> np.ndarray:
    n = X.shape[0]
    diff        = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))

    sum_dists  = dist_matrix.sum(axis=1)
    chosen_idx = [int(np.argmin(sum_dists))]

    for _ in range(1, k):
        last_idx          = chosen_idx[-1]
        distances         = dist_matrix[:, last_idx].copy()
        distances[chosen_idx] = -np.inf
        chosen_idx.append(int(np.argmax(distances)))

    return X[chosen_idx]


# ─────────────────────────────────────────────
#  RWC-K-MEANS
# ─────────────────────────────────────────────

def rwcKMeans(
    X: np.ndarray,
    k: int,
    max_iter: int = 300,
    tol: float = 1e-6,
    seed: int = 42,
):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    centroids      = initialCentroids(X, k)
    prev_centroids = centroids.copy()
    labels         = np.zeros(n, dtype=int)
    cost_history   = []

    for iteration in range(max_iter):

        diff       = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances  = np.sqrt((diff ** 2).sum(axis=2))
        new_labels = np.argmin(distances, axis=1)

        J = distances[np.arange(n), new_labels].sum()
        cost_history.append(J)
        labels = new_labels

        mean_centroids = np.zeros_like(centroids)
        for i in range(k):
            members = X[labels == i]
            mean_centroids[i] = members.mean(axis=0) if len(members) > 0 else centroids[i]

        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            r         = rng.uniform(0.0, 1.0)
            delta     = prev_centroids[i] - mean_centroids[i]
            magnitude = np.linalg.norm(delta)

            if magnitude > 1e-12:
                unit             = delta / magnitude
                new_centroids[i] = mean_centroids[i] + r * magnitude * unit
            else:
                new_centroids[i] = mean_centroids[i]

        shift          = np.sqrt(((new_centroids - centroids) ** 2).sum(axis=1)).max()
        prev_centroids = centroids.copy()
        centroids      = new_centroids

        if shift < tol:
            print(f"Converged at iteration {iteration + 1}  (shift={shift:.2e})")
            break
    else:
        print(f"Reached max_iter={max_iter} without full convergence.")

    return labels, centroids, cost_history


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
        label   = f"Cluster {cluster}"

        for _, seg in grp.groupby("segment_id"):
            line, = ax.plot(
                seg["lon"], seg["lat"],
                color=color, linewidth=1.5, alpha=0.7,
                transform=ccrs.PlateCarree(),
                label=label if label not in legend_handles else "_nolegend_",
            )
            legend_handles[label] = line

    ax.legend(handles=list(legend_handles.values()),
              labels=list(legend_handles.keys()),
              fontsize=8, loc="upper left")
    plt.title("Fisher Trajectories – RWC-K-Means Clusters (with Depth)")
    plt.tight_layout()
    plt.show()


def printClusterMembers(features: pd.DataFrame):
    print("\n" + "=" * 50)
    print("CLUSTER MEMBERSHIP")
    print("=" * 50)
    for cluster_id, grp in features.groupby("cluster"):
        print(f"\nCluster {cluster_id}  ({len(grp)} fishers):")
        for fid, row in grp.iterrows():
            print(f"  [{fid}] {row['file_name']}")
            print(f"        mean depth: {row['mean_depth']:.0f}m  |  "
                  f"max depth: {row['max_depth']:.0f}m  |  "
                  f"mean speed: {row['mean_speed']:.1f}  |  "
                  f"segments: {int(row['num_segments'])}")
    print("=" * 50)


def plotClusterDepthProfile(features: pd.DataFrame):
    cluster_depth = features.groupby("cluster")["mean_depth"].mean()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        [f"Cluster {c}" for c in cluster_depth.index],
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
    print("Loading depth grid...")
    lats, lons, depth_grid = loadDepthGrid(cross_ref_folder)

    # 2. Load fisher data with depth lookup
    print("Loading fisher data and looking up depths...")
    combined = loadData(fisher_folder, lats, lons, depth_grid)
    print(f"Loaded {combined['file_id'].nunique()} trajectories "
          f"({len(combined):,} total timepoints).")

    # 3. Extract features
    features = extractFeatures(combined)
    print(f"Feature matrix shape: {features.shape}")

    # 4. Normalize (numeric columns only)
    feature_cols = ["mean_lat", "mean_lon", "std_lat", "std_lon",
                    "mean_speed", "max_speed", "total_distance",
                    "num_segments", "mean_depth", "max_depth", "min_depth"]

    X_raw  = features[feature_cols].values.astype(float)
    mu     = X_raw.mean(axis=0)
    sigma  = X_raw.std(axis=0) + 1e-8
    X      = (X_raw - mu) / sigma

    # 5. Run RWC-K-Means
    numCluster = 6
    labels, centroids, cost_history = rwcKMeans(X, k=numCluster, max_iter=300, seed=42)

    features["cluster"] = labels
    labels_map = dict(zip(features.index, labels))

    # 6. Print which fisher is in each cluster
    printClusterMembers(features)

    # 7. Visualise
    plotClusteredTrajectories(combined, labels_map)
    plotClusterDepthProfile(features)


if __name__ == "__main__":
    main()