import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd  
import cartopy.crs as ccrs
import cartopy.feature as cfeature

def cleanData():
    ghana = Path("/Users/tushy/Documents/Research/Ghana Project/fwcheckin")

    for file_path in ghana.glob("*.csv"):
        df = pd.read_csv(file_path)
            
        if df.shape[0] < 100: # checking number of time stamps  
            file_path.unlink()

    print("done iterating")
    

def graphAll():
    allData = []
    ghana = Path("/Users/tushy/Documents/Research/Ghana Project/fwcheckin")
    
    for i, file_path in enumerate(ghana.glob("*.csv")):
        df = pd.read_csv(file_path)
        
        if df.shape[0] >= 4000:
            df = df.iloc[:, :3].copy()
            df["file_id"] = i + 1
            allData.append(df)
    
    combined = pd.concat(allData, ignore_index=True)
    combined["Var3"] = pd.to_datetime(combined["Var3"])
    
    combined["time_diff"] = combined.groupby("file_id")["Var3"].diff()
    combined["new_segment"] = combined["time_diff"] > pd.Timedelta(hours=2)
    combined["segment_id"] = combined.groupby("file_id")["new_segment"].cumsum()
    
    graphing(combined)
    
    
from cartopy.io.img_tiles import OSM, GoogleTiles

def graphing(combined):
    imagery = GoogleTiles(style="satellite") 

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)

    lon_min, lon_max = combined["Var2"].min(), combined["Var2"].max()
    lat_min, lat_max = combined["Var1"].min(), combined["Var1"].max()
    padding = 0.3 # basically just tells matplot how much of the map to show 
    ax.set_extent([lon_min - padding, lon_max + padding, 
                   lat_min - padding, lat_max + padding])

    ax.add_image(imagery, 10)   # 10 = zoom level (higher = more detail)

    for file_id, file_data in combined.groupby("file_id"):
        line, = ax.plot([], [], label=f"File {file_id}", transform=ccrs.PlateCarree()) #basically telling it to use the lat/lon data 
        color = line.get_color()
        for _, segment in file_data.groupby("segment_id"):
            ax.plot(segment["Var2"], segment["Var1"],
                    color=color, transform=ccrs.PlateCarree(),
                    linewidth=2)

    plt.legend()
    plt.legend(fontsize = 7)
    plt.title("Trajectories (Greater Than 4000 Datapoints)")
    plt.show()

if __name__ == "__main__":
    graphAll()
    
    
    
