import pandas as pd
from pathlib import Path
import os
import re

GTFS_DIR = Path("Fahrplan_VBB-2021")
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

    print("Applying artificial SEV adjustments for U5, U2, and S-Bahn gaps...")
    # U5 stations
    u5_base_names = [
        "U Weberwiese", "U Samariterstr", "U Schillingstr", "U Strausberger Platz",
        "U Frankfurter Tor", "U Magdalenenstr", "U Friedrichsfelde", "U Tierpark",
        "U Biesdorf-Süd", "U Elsterwerdaer Platz", "U Kaulsdorf-Nord", "U Kienberg",
        "U Cottbusser Platz", "U Hellersdorf", "U Louis-Lewin-Str", "U Hönow",
        "U Rotes Rathaus", "U Unter den Linden", "U Museumsinsel", "U Bundestag",
        "S+U Alexanderplatz", "S+U Frankfurter Allee", "S+U Lichtenberg", "S+U Wuhletal",
        "S+U Brandenburger Tor", "S+U Berlin Hauptbahnhof"
    ]
    
    # U2 and S-Bahn stations reported as missing/zero due to western corridor construction
    extra_fixes = [
        {"names": ["U Neu-Westend", "U Olympia-Stadion", "U Ruhleben", "U Theodor-Heuss-Platz"], "line": "U2|400", "deps": 420},
        {"names": ["S Olympiastadion", "S Pichelsberg", "S Stresow", "S Messe Süd", "S Heerstraße"], "line": "S3|109,S9|109", "deps": 350}
    ]
    
    # Process U5 first (existing logic)
    u5_mask = stops['stop_name'].str.contains('|'.join(u5_base_names), case=False, na=False)
    u5_stops_raw = stops[u5_mask].copy()
    u5_stops_raw = u5_stops_raw[~u5_stops_raw['stop_name'].str.contains('/', na=False)]
    u5_stops_raw['stop_lat'] = pd.to_numeric(u5_stops_raw['stop_lat'], errors='coerce')
    u5_stops_raw['stop_lon'] = pd.to_numeric(u5_stops_raw['stop_lon'], errors='coerce')
    u5_stops = u5_stops_raw.groupby('stop_name').agg({'stop_lat': 'mean', 'stop_lon': 'mean'}).reset_index()
    u5_stops['daily_departures'] = 420
    u5_stops['lines'] = "U5|400"
    
    # Merge all into station_freq
    station_freq = station_freq.set_index('stop_name')
    
    # Apply U5 fixes
    for _, row in u5_stops.iterrows():
        name = row['stop_name']
        if name in station_freq.index:
            if 'U5' not in str(station_freq.at[name, 'lines']):
                station_freq.at[name, 'daily_departures'] += 420
                existing_lines = str(station_freq.at[name, 'lines'])
                station_freq.at[name, 'lines'] = (existing_lines + ",U5|400") if existing_lines != 'nan' and existing_lines else "U5|400"
        else:
            station_freq.loc[name] = [row['stop_lat'], row['stop_lon'], row['daily_departures'], row['lines']]
            
    # Apply Extra fixes (U2 / S-Bahn)
    for fix in extra_fixes:
        mask = stops['stop_name'].str.contains('|'.join(fix['names']), case=False, na=False)
        stops_raw = stops[mask].copy()
        stops_raw = stops_raw[~stops_raw['stop_name'].str.contains('/', na=False)]
        stops_raw['stop_lat'] = pd.to_numeric(stops_raw['stop_lat'], errors='coerce')
        stops_raw['stop_lon'] = pd.to_numeric(stops_raw['stop_lon'], errors='coerce')
        grouped = stops_raw.groupby('stop_name').agg({'stop_lat': 'mean', 'stop_lon': 'mean'}).reset_index()
        
        for _, row in grouped.iterrows():
            name = row['stop_name']
            line_id = fix['line'].split('|')[0]
            if name in station_freq.index:
                if line_id not in str(station_freq.at[name, 'lines']):
                    station_freq.at[name, 'daily_departures'] += fix['deps']
                    existing_lines = str(station_freq.at[name, 'lines'])
                    station_freq.at[name, 'lines'] = (existing_lines + "," + fix['line']) if existing_lines != 'nan' and existing_lines else fix['line']
            else:
                station_freq.loc[name] = [row['stop_lat'], row['stop_lon'], fix['deps'], fix['line']]
            
    station_freq = station_freq.reset_index()

    CACHE_DIR.mkdir(exist_ok=True)
    out_file = CACHE_DIR / "station_frequencies.csv"
    station_freq.to_csv(out_file, index=False)
    print(f"Done! Saved frequency data (including U5 fix) for {len(station_freq)} stations to {out_file}")

if __name__ == "__main__":
    compute_frequencies()
