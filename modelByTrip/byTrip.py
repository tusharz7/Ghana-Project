import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

TRIP_GAP_HOURS     = 24    # gap in hours that signals a new trip
MIN_TRIP_POINTS    = 10    # trip must have at least this many points to be considered a "new trip"
SHALLOW_THRESHOLD  = 20    # metres — at port if depth <= this
OFFSHORE_THRESHOLD = 500   # metres — off the continental shelf


# ─────────────────────────────────────────────
#  DATA LOADING FROM fisherDepth CSVs
# ─────────────────────────────────────────────

def loadData(fisher_depth_folder: Path) -> pd.DataFrame:
    """
    Trip segmentation: a new trip starts whenever there is a
    gap of >= 24 hours between consecutive timestamps.
    """
    allData = []

    for file_path in sorted(fisher_depth_folder.glob("*.csv")):
        try:
            df = pd.read_csv(file_path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp", "lat", "lon", "depth_m"])
            df = df.sort_values("timestamp").reset_index(drop=True)

            if len(df) < MIN_TRIP_POINTS:
                continue

            # ── Assign trip segments based on 24-hour gaps ──────────
            df["time_diff"]   = df["timestamp"].diff()
            df["new_trip"]    = df["time_diff"] >= pd.Timedelta(hours=TRIP_GAP_HOURS)
            df["segment_id"]  = df["new_trip"].cumsum()

            allData.append(df[["file_id", "file_name", "timestamp",
                                "lat", "lon", "speed", "depth_m", "segment_id"]])

            n_trips = df["segment_id"].nunique()
            print(f"  {file_path.name:<50} {len(df):>6} rows  {n_trips:>3} trips")

        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    if not allData:
        raise ValueError("No valid CSV files found in fisherDepth folder.")

    combined = pd.concat(allData, ignore_index=True)
    combined.sort_values(["file_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ─────────────────────────────────────────────
#  TRIP FEATURE EXTRACTION
# ─────────────────────────────────────────────

def extractTripFeatures(combined: pd.DataFrame) -> pd.DataFrame:
    """Each row = one trip (file_id + segment_id)."""
    records = []

    for (fid, seg_id), grp in combined.groupby(["file_id", "segment_id"]):
        grp = grp.sort_values("timestamp")

        if len(grp) < MIN_TRIP_POINTS:
            continue

        valid_depth = grp["depth_m"].dropna()

        # Skip port stays — trips that never left shallow water
        if len(valid_depth) == 0 or valid_depth.max() <= SHALLOW_THRESHOLD:
            continue

        duration_hours = (
            grp["timestamp"].max() - grp["timestamp"].min()
        ).total_seconds() / 3600

        # Did fisher return to port? (last 10% of points back in shallow water)
        n_tail     = max(1, int(len(grp) * 0.1))
        tail_depth = grp["depth_m"].iloc[-n_tail:].dropna()
        returned   = int(len(tail_depth) > 0 and tail_depth.mean() <= SHALLOW_THRESHOLD)

        pct_off_shelf = (valid_depth > OFFSHORE_THRESHOLD).sum() / len(valid_depth) * 100

        records.append({
            "file_id":        fid,
            "file_name":      grp["file_name"].iloc[0],
            "segment_id":     seg_id,
            "trip_id":        f"{grp['file_name'].iloc[0]}_trip{int(seg_id)+1}",
            "duration_hours": duration_hours,
            "mean_lat":       grp["lat"].mean(),
            "mean_lon":       grp["lon"].mean(),
            "std_lat":        grp["lat"].std(),
            "std_lon":        grp["lon"].std(),
            "mean_speed":     grp["speed"].mean(),
            "max_speed":      grp["speed"].max(),
            "mean_depth":     valid_depth.mean(),
            "max_depth":      valid_depth.max(),
            "pct_off_shelf":  pct_off_shelf,
            "returned":       returned,
        })

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  VISUALISATION
# ─────────────────────────────────────────────

CLUSTER_COLORS = [
    "#e63946",  # red
    "#2a9d8f",  # teal
    "#f4a261",  # orange
    "#6a0dad",  # purple
    "#faff00",  # yellow
]

def plotClusteredTrips(combined: pd.DataFrame, trip_features: pd.DataFrame):
    """Trajectories coloured by cluster."""
    imagery = GoogleTiles(style="satellite")
    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(1, 1, 1, projection=imagery.crs)

    pad = 0.3
    ax.set_extent([
        combined["lon"].min() - pad, combined["lon"].max() + pad,
        combined["lat"].min() - pad, combined["lat"].max() + pad,
    ])
    ax.add_image(imagery, 10)

    legend_handles = {}
    for _, trip in trip_features.iterrows():
        fid     = trip["file_id"]
        seg_id  = trip["segment_id"]
        cluster = int(trip["cluster"])
        color   = CLUSTER_COLORS[cluster % len(CLUSTER_COLORS)]
        label   = f"Cluster {cluster + 1}"

        seg = combined[(combined["file_id"] == fid) &
                       (combined["segment_id"] == seg_id)]
        if seg.empty:
            continue

        line, = ax.plot(
            seg["lon"], seg["lat"],
            color=color, linewidth=1.5, alpha=0.7,
            transform=ccrs.PlateCarree(),
        )
        legend_handles[label] = line

    ax.legend(handles=list(legend_handles.values()),
              labels=list(legend_handles.keys()),
              fontsize=8, loc="upper left")
    plt.title("Individual Trips — K-Means Clusters")
    plt.tight_layout()
    plt.show()


def plotClusterProfiles(trip_features: pd.DataFrame):
    metrics = {
        "Avg Duration (hrs)": "duration_hours",
        "Avg Mean Depth (m)": "mean_depth",
        "Avg % Off Shelf":    "pct_off_shelf",
    }
    cluster_means = trip_features.groupby("cluster")[list(metrics.values())].mean()

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 4))
    for ax, (title, col) in zip(axes, metrics.items()):
        colors = [CLUSTER_COLORS[int(c) % len(CLUSTER_COLORS)] for c in cluster_means.index]
        ax.bar([f"C{int(c)+1}" for c in cluster_means.index],
               cluster_means[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Trip Cluster Profiles", fontsize=11)
    plt.tight_layout()
    plt.show()


def printClusterSummary(trip_features: pd.DataFrame):
    print("\n" + "=" * 65)
    print("TRIP CLUSTER SUMMARY")
    print("=" * 65)

    overall = trip_features.groupby("cluster").agg(
        num_trips      = ("trip_id",       "count"),
        num_fishers    = ("file_name",     "nunique"),
        avg_duration   = ("duration_hours","mean"),
        avg_mean_depth = ("mean_depth",    "mean"),
        avg_max_depth  = ("max_depth",     "mean"),
        avg_pct_shelf  = ("pct_off_shelf", "mean"),
        avg_speed      = ("mean_speed",    "mean"),
        pct_returned   = ("returned",      "mean"),
    )

    for cluster_id, row in overall.iterrows():
        print(f"\nCluster {int(cluster_id) + 1}")
        print(f"  Trips          : {int(row['num_trips'])}")
        print(f"  Unique fishers : {int(row['num_fishers'])}")
        print(f"  Avg duration   : {row['avg_duration']:.1f} hrs")
        print(f"  Avg mean depth : {row['avg_mean_depth']:.0f} m")
        print(f"  Avg max depth  : {row['avg_max_depth']:.0f} m")
        print(f"  Avg % off shelf: {row['avg_pct_shelf']:.1f}%")
        print(f"  Avg speed      : {row['avg_speed']:.2f}")

    # Fisher breakdown per cluster
    print("\n" + "=" * 65)
    print("FISHER BREAKDOWN PER CLUSTER")
    print("=" * 65)
    for cluster_id, grp in trip_features.groupby("cluster"):
        print(f"\nCluster {int(cluster_id) + 1}:")
        fisher_counts = grp.groupby("file_name")["trip_id"].count().sort_values(ascending=False)
        for fisher, n_trips in fisher_counts.items():
            print(f"  {fisher:<50} {n_trips} trip{'s' if n_trips > 1 else ''}")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    fisher_depth_folder = Path("/Users/tushy/Documents/Research/Ghana Project/fisherDepth")

    # 1. Load from fisherDepth CSVs (depth already computed)
    combined = loadData(fisher_depth_folder)
    print(f"\nLoaded {combined['file_id'].nunique()} fishers "
          f"({len(combined):,} total timepoints).")

    # 2. Extract per-trip features
    print("\nExtracting trip features...")
    trip_features = extractTripFeatures(combined)
    print(f"Extracted {len(trip_features)} valid trips across "
          f"{trip_features['file_name'].nunique()} fishers.")

    # 3. Cluster
    feature_cols = [
        "duration_hours",
        "mean_lat", "mean_lon",
        "std_lat",  "std_lon",
        "mean_speed", "max_speed",
        "mean_depth", "max_depth",
        "pct_off_shelf",
    ]

    scaler = StandardScaler()
    X = scaler.fit_transform(trip_features[feature_cols].values)

    k  = 5
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    trip_features["cluster"] = km.fit_predict(X)

    printClusterSummary(trip_features)
    plotClusteredTrips(combined, trip_features)
    plotClusterProfiles(trip_features)


if __name__ == "__main__":
    main()