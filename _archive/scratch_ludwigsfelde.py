import pandas as pd
from pathlib import Path

def run():
    stops = pd.read_csv("Fahrplan_VBB-2021/stops.txt", dtype=str)
    matches = stops[stops['stop_name'].str.contains('Ludwigsfelde', case=False, na=False)]
    print(matches[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].to_string())
            
if __name__ == "__main__":
    run()
