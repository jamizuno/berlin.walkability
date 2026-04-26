import pandas as pd
from pathlib import Path
import os

GTFS_DIR = Path("Fahrplan_VBB-2021")
CACHE_DIR = Path("cache")

def compute_frequencies():
    print("Loading GTFS files... (This might take ~10-20 seconds as stop_times.txt is large)")
    stops = pd.read_csv(GTFS_DIR / "stops.txt", dtype={'stop_id': str})
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=['trip_id', 'stop_id'], dtype={'trip_id': str, 'stop_id': str})
    trips = pd.read_csv(GTFS_DIR / "trips.txt", usecols=['route_id', 'service_id', 'trip_id'], dtype=str)
    
    # Read calendar dates as some services might only exist in calendar_dates.txt
    # We will just look at calendar.txt for regular weekday services to keep it simple and robust
    calendar = pd.read_csv(GTFS_DIR / "calendar.txt", dtype=str)

    print("Filtering for regular weekday service...")
    # Find service_ids that run mon-fri
    weekday_services = calendar[
        (calendar['monday'] == '1') & 
        (calendar['tuesday'] == '1') & 
        (calendar['wednesday'] == '1') & 
        (calendar['thursday'] == '1') & 
        (calendar['friday'] == '1')
    ]['service_id']

    print("Counting departures per stop...")
    weekday_trips = trips[trips['service_id'].isin(weekday_services)]
    weekday_stop_times = stop_times[stop_times['trip_id'].isin(weekday_trips['trip_id'])]

    # Count departures per stop_id
    departures = weekday_stop_times.groupby('stop_id').size().reset_index(name='daily_departures')

    print("Consolidating by station name...")
    # Merge with stops to get coordinates and names
    stop_freq = pd.merge(stops, departures, on='stop_id', how='inner')

    # Many stations have multiple platforms (each with a stop_id). 
    # Group by the exact stop_name and average the coordinates.
    station_freq = stop_freq.groupby('stop_name').agg({
        'stop_lat': 'mean',
        'stop_lon': 'mean',
        'daily_departures': 'sum'
    }).reset_index()

    CACHE_DIR.mkdir(exist_ok=True)
    out_file = CACHE_DIR / "station_frequencies.csv"
    station_freq.to_csv(out_file, index=False)
    print(f"Done! Saved frequency data for {len(station_freq)} stations to {out_file}")

if __name__ == "__main__":
    compute_frequencies()
