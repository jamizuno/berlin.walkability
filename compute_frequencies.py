import pandas as pd
from pathlib import Path
import os

GTFS_DIR = Path("Fahrplan_VBB-2026")
CACHE_DIR = Path("cache")

def compute_frequencies():
    print("Loading GTFS files... (This might take ~10-20 seconds as stop_times.txt is large)")
    stops = pd.read_csv(GTFS_DIR / "stops.txt", dtype={'stop_id': str})
    stop_times = pd.read_csv(GTFS_DIR / "stop_times.txt", usecols=['trip_id', 'stop_id'], dtype={'trip_id': str, 'stop_id': str})
    trips = pd.read_csv(GTFS_DIR / "trips.txt", usecols=['route_id', 'service_id', 'trip_id'], dtype=str)
    routes = pd.read_csv(GTFS_DIR / "routes.txt", usecols=['route_id', 'route_short_name', 'route_type'], dtype=str)
    
    calendar = pd.read_csv(GTFS_DIR / "calendar.txt", dtype=str)

    print("Filtering for regular weekday service...")
    weekday_services = calendar[
        (calendar['monday'] == '1') & 
        (calendar['tuesday'] == '1') & 
        (calendar['wednesday'] == '1') & 
        (calendar['thursday'] == '1') & 
        (calendar['friday'] == '1')
    ]['service_id']

    print("Filtering for Rail and Tram lines only (excluding buses)...")
    weekday_trips = trips[trips['service_id'].isin(weekday_services)]
    weekday_trips = weekday_trips.merge(routes, on='route_id', how='left')
    
    rail_tram_types = ['0', '1', '2', '100', '101', '102', '109', '400', '900']
    weekday_trips = weekday_trips[weekday_trips['route_type'].isin(rail_tram_types)]

    print("Counting rail/tram departures per stop...")
    weekday_stop_times = stop_times[stop_times['trip_id'].isin(weekday_trips['trip_id'])]
    departures = weekday_stop_times.groupby('stop_id').size().reset_index(name='daily_departures')

    print("Gathering unique lines per stop...")
    stop_time_routes = weekday_stop_times.merge(weekday_trips[['trip_id', 'route_short_name', 'route_type']], on='trip_id', how='left')
    
    unique_stop_routes = stop_time_routes[['stop_id', 'route_short_name', 'route_type']].dropna(subset=['route_short_name']).drop_duplicates()
    unique_stop_routes['line_str'] = unique_stop_routes['route_short_name'] + '|' + unique_stop_routes['route_type']
    lines_per_stop = unique_stop_routes.groupby('stop_id')['line_str'].apply(','.join).reset_index(name='lines')

    stop_stats = pd.merge(departures, lines_per_stop, on='stop_id', how='left')

    print("Consolidating by station name...")
    stop_freq = pd.merge(stops, stop_stats, on='stop_id', how='inner')

    def combine_lines(series):
        all_lines = set()
        for s in series.dropna():
            if str(s) != 'nan' and s:
                all_lines.update([l.strip() for l in str(s).split(',') if l.strip()])
        return ','.join(all_lines)

    station_freq = stop_freq.groupby('stop_name').agg({
        'stop_lat': 'mean',
        'stop_lon': 'mean',
        'daily_departures': 'sum',
        'lines': combine_lines
    }).reset_index()

    CACHE_DIR.mkdir(exist_ok=True)
    out_file = CACHE_DIR / "station_frequencies.csv"
    station_freq.to_csv(out_file, index=False)
    print(f"Done! Saved frequency data for {len(station_freq)} stations to {out_file}")

if __name__ == "__main__":
    compute_frequencies()
