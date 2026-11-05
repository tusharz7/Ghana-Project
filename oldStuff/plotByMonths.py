import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
from cartopy.io.img_tiles import GoogleTiles

# 12 distinct colors, one per month
MONTH_COLORS = {
    1:  "#1f77b4",  # Jan - blue
    2:  "#aec7e8",  # Feb - light blue
    3:  "#2ca02c",  # Mar - green
    4:  "#98df8a",  # Apr - light green
    5:  "#ffbb78",  # May - light orange
    6:  "#ff7f0e",  # Jun - orange
    7:  "#d62728",  # Jul - red
    8:  "#ff9896",  # Aug - light red/pink
    9:  "#9467bd",  # Sep - purple
    10: "#c5b0d5",  # Oct - light purple
    11: "#8c564b",  # Nov - brown
    12: "#c49c94",  # Dec - light brown
}

MONTH_NAMES = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
}

def graphIndividual():
    ghana = Path("/Users/tushy/Documents/Research/Ghana Project/fwcheckin")
    output_dir = Path("/Users/tushy/Documents/Research/Ghana Project/individual_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(ghana.glob("*.csv"))
    fisher_num = 1

    for file_path in files:
        df = pd.read_csv(file_path)

        print(f"Processing Fisher {fisher_num}: {file_path.name}")

        df = df.iloc[:, :3].copy()
        df.columns = ["Var1", "Var2", "Var3"]
        df["Var3"] = pd.to_datetime(df["Var3"])

        # Segment by month 
        df = df.sort_values("Var3").reset_index(drop=True)
        df["month"] = df["Var3"].dt.month
        df["segment_id"] = df["month"]  # segment ID is just the month

        
        # --- Plot ---
        imagery = GoogleTiles(style="satellite")
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

        lon_min, lon_max = df["Var2"].min(), df["Var2"].max()
        lat_min, lat_max = df["Var1"].min(), df["Var1"].max()
        padding = 0.3
        ax.set_extent([lon_min - padding, lon_max + padding,
                       lat_min - padding, lat_max + padding])
        ax.add_image(imagery, 10)

        # Plot each segment, colored by its dominant month
        for seg_id, seg_data in df.groupby("segment_id"):
            color = MONTH_COLORS[seg_id]  # seg_id IS the month now
            ax.plot(
                seg_data["Var2"], seg_data["Var1"],
                color=color,
                transform=ccrs.PlateCarree(),
                linewidth=1.5,
                alpha=0.85
            )

        # Legend: only months actually present in this file
        present_months = sorted(df["month"].unique())
        legend_handles = [
            plt.Line2D([0], [0], color=MONTH_COLORS[m], linewidth=2,
                       label=MONTH_NAMES[m])
            for m in present_months
        ]
        ax.legend(handles=legend_handles, title="Month",
                  fontsize=7, title_fontsize=8, loc="lower left")

        plt.title(f"Fisher {fisher_num} — {df.shape[0]}", fontsize=11)

        out_path = output_dir / f"Fisher_{fisher_num}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved → {out_path}")

        fisher_num += 1

    print("Done")


if __name__ == "__main__":
    graphIndividual()