import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def loadData(folder: Path) -> pd.DataFrame:
    allData = []

    for i, file_path in enumerate(folder.glob("*.xlsx")):
        try:
            df = pd.read_excel(file_path, header=4)

            df.columns = ["events", "timestamp", "address",
                          "lat_lng", "speed", "heading",
                          "altitude", "via", "drivers"]

            df = df[df["timestamp"] != "Date"].copy()

            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df.dropna(subset=["timestamp"])
            if len(df) < 100:
                continue
            df["file_id"]   = i + 1
            df["file_name"] = file_path.stem

            df = df.sort_values("timestamp")
            allData.append(df[["file_id", "file_name", "timestamp"]])

        except Exception as e:
            print(f"  Skipping {file_path.name}: {e}")

    combined = pd.concat(allData, ignore_index=True)
    combined.sort_values(["file_id", "timestamp"], inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined


# ─────────────────────────────────────────────
#  COMPUTE dT
# ─────────────────────────────────────────────

def computeDT(combined: pd.DataFrame) -> pd.Series:
    dt = combined.groupby("file_id")["timestamp"].diff().dt.total_seconds()
    return dt.dropna()


# ─────────────────────────────────────────────
#  PLOT
# ─────────────────────────────────────────────

def plotDTHistogram(dt: pd.Series):
    dt_minutes = dt / 60

    # Cut off at 99th percentile so extreme outliers don't squash the histogram
    upper_limit = np.percentile(dt_minutes, 99)

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.hist(
        dt_minutes[dt_minutes <= upper_limit],
        bins=60,
        color="#2a9d8f",
        edgecolor="none",
        alpha=0.85
    )

    ax.set_xlabel("dT (minutes)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("dT Distribution — Time Between Consecutive GPS Points", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Print summary
    print("\ndT Summary (all fishers with >= 100 datapoints):")
    print(f"  Average dT         : {dt.mean():.1f} sec  ({dt.mean()/60:.2f} min)")
    print(f"  Median dT          : {dt.median():.1f} sec  ({dt.median()/60:.2f} min)")
    print(f"  99th percentile dT : {np.percentile(dt, 99):.1f} sec  "
          f"({np.percentile(dt, 99)/60:.2f} min)")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    folder = Path("/Users/tushy/Documents/Research/Ghana Project/Exported Files")
    combined = loadData(folder)
    print(f"Loaded {combined['file_id'].nunique()} fishers "
          f"({len(combined):,} total timepoints).")

    dt = computeDT(combined)
    plotDTHistogram(dt)


if __name__ == "__main__":
    main()
