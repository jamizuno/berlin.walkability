#!/usr/bin/env python3
"""
Berlin Transport Station Extraction Pipeline
Extracts all stations from VBB GTFS: S-Bahn, RE, FEX, U-Bahn, Tram
"""

import pandas as pd
import os
from collections import defaultdict

# Configuration
GTFS_DIR = r"C:\Users\jermo\Documents\BERLINMAP-DISTANCE\Haltestellen_VBB\GTFS"
OUTPUT_FILE = r"C:\Users\jermo\Documents\BERLINMAP-DISTANCE\Haltestellen_VBB\stations_reference.txt"

# Target lines for rail
TARGET_RAIL_LINES = {
    'S1', 'S2', 'S25', 'S26', 'S3', 'S5', 'S7', 'S8', 'S9',
    'S41', 'S42', 'S45', 'S46', 'S47', 'S85',
    'RE1', 'RE2', 'RE3', 'RE4', 'RE5', 'RE7',
    'FEX'
}

# GTFS route_type values
ROUTE_TYPE_RAIL = {109, 100, 106}  # S-Bahn, RE, RB, FEX
ROUTE_TYPE_SUBWAY = {400}   # U-Bahn
ROUTE_TYPE_TRAM = {900}     # Tram

# Optional: RB lines
OPTIONAL_RB = True

def load_gtfs_files():
    """Load required GTFS files."""
    print("Loading GTFS files...")
    
    # Load routes
    routes = pd.read_csv(os.path.join(GTFS_DIR, "routes.txt"))
    print(f"  Routes: {len(routes)} rows")
    
    # Load trips
    trips = pd.read_csv(os.path.join(GTFS_DIR, "trips.txt"))
    print(f"  Trips: {len(trips)} rows")
    
    # Load stops
    stops = pd.read_csv(os.path.join(GTFS_DIR, "stops.txt"))
    print(f"  Stops: {len(stops)} rows")
    
    return routes, trips, stops

def get_route_type_name(route_type, route_short_name):
    """Map GTFS route_type to transport type name."""
    if route_type == 109:  # S-Bahn
        return "S-Bahn"
    elif route_type == 100:  # RE/FEX
        if route_short_name and route_short_name.startswith('FEX'):
            return "FEX"
        return "Regio-RE"
    elif route_type == 106:  # RB
        return "Regio-RB"
    elif route_type == 400:  # U-Bahn
        return "U-Bahn"
    elif route_type == 900:  # Tram
        return "Tram"
    return None

def filter_routes_by_type(routes, route_types):
    """Filter routes by route_type."""
    return routes[routes['route_type'].isin(route_types)].copy()

def load_stop_times_for_routes(trips_df):
    """Load stop_times efficiently by filtering to relevant trips."""
    print("\nLoading stop_times for target routes...")
    
    target_trip_ids = set(trips_df['trip_id'].unique())
    print(f"  Target trip IDs: {len(target_trip_ids)}")
    
    # Load in chunks
    chunk_size = 5_000_000
    stop_times_chunks = []
    
    for chunk in pd.read_csv(
        os.path.join(GTFS_DIR, "stop_times.txt"),
        chunksize=chunk_size,
        usecols=['trip_id', 'stop_id']
    ):
        filtered = chunk[chunk['trip_id'].isin(target_trip_ids)]
        stop_times_chunks.append(filtered)
        print(f"    Processed chunk, kept {len(filtered)} stop_times")
    
    stop_times = pd.concat(stop_times_chunks, ignore_index=True)
    print(f"  Total stop_times: {len(stop_times)}")
    
    return stop_times

def build_stop_to_station_mapping(stops):
    """Build mapping from stop_id to station_id using parent_station."""
    print("\nBuilding stop → station mapping...")
    
    stop_to_station = {}
    station_info = {}
    
    for _, stop in stops.iterrows():
        stop_id = stop['stop_id']
        
        # Determine station ID
        if pd.notna(stop.get('parent_station')) and stop['parent_station'] != '':
            station_id = stop['parent_station']
        else:
            station_id = stop_id
        
        stop_to_station[stop_id] = station_id
        
        # Store station info
        if station_id not in station_info:
            station_info[station_id] = {
                'name': stop['stop_name'],
                'lat': stop['stop_lat'],
                'lon': stop['stop_lon'],
                'stop_ids': set()
            }
        
        station_info[station_id]['stop_ids'].add(stop_id)
    
    print(f"  Unique stops: {len(stop_to_station)}")
    print(f"  Unique stations: {len(station_info)}")
    
    return stop_to_station, station_info

def extract_stations(stop_times, stop_to_station, station_info, routes_df, trips_df):
    """Extract all stations served by target routes."""
    print("\nExtracting stations...")
    
    # Get unique stop_ids from target trips
    target_stop_ids = set(stop_times['stop_id'].unique())
    print(f"  Unique stops in target trips: {len(target_stop_ids)}")
    
    # Map to station IDs
    target_station_ids = set()
    for stop_id in target_stop_ids:
        if stop_id in stop_to_station:
            target_station_ids.add(stop_to_station[stop_id])
    
    print(f"  Unique stations served: {len(target_station_ids)}")
    
    # Build route_id -> line name and route_type mapping
    route_to_line = dict(zip(routes_df['route_id'], routes_df['route_short_name']))
    route_to_type = dict(zip(routes_df['route_id'], routes_df['route_type']))
    
    # Get trip_id -> route_id mapping
    trip_to_route = dict(zip(trips_df['trip_id'], trips_df['route_id']))
    
    # Build stop_id -> lines and types mapping
    stop_id_lines = defaultdict(set)
    stop_id_types = defaultdict(set)
    
    for _, st in stop_times.iterrows():
        trip_id = st['trip_id']
        stop_id = st['stop_id']
        
        if trip_id in trip_to_route:
            route_id = trip_to_route[trip_id]
            if route_id in route_to_line:
                line = route_to_line[route_id]
                route_type = route_to_type.get(route_id)
                transport = get_route_type_name(route_type, line)
                
                if line:
                    stop_id_lines[stop_id].add(line)
                if transport:
                    stop_id_types[stop_id].add(transport)
    
    # Aggregate per station
    station_data = []
    
    for station_id in target_station_ids:
        info = station_info.get(station_id, {})
        
        # Get all stop_ids for this station
        station_stop_ids = info.get('stop_ids', set()) & target_stop_ids
        
        # Collect all lines and types serving this station
        lines = set()
        transport_types = set()
        
        for stop_id in station_stop_ids:
            lines.update(stop_id_lines.get(stop_id, set()))
            transport_types.update(stop_id_types.get(stop_id, set()))
        
        station_data.append({
            'station_id': station_id,
            'station_name': info.get('name', 'Unknown'),
            'lat': info.get('lat'),
            'lon': info.get('lon'),
            'lines': sorted(lines),
            'transport_types': sorted(transport_types)
        })
    
    print(f"  Station data entries: {len(station_data)}")
    
    return station_data

def aggregate_and_deduplicate(station_data):
    """Aggregate stations with the same name and deduplicate."""
    print("\nAggregating and deduplicating...")
    
    df = pd.DataFrame(station_data)
    
    # Aggregate by station_name
    aggregated = []
    
    for station_name, group in df.groupby('station_name', sort=False):
        all_lines = set()
        all_types = set()
        lats = []
        lons = []
        
        for _, row in group.iterrows():
            all_lines.update(row['lines'])
            all_types.update(row['transport_types'])
            if pd.notna(row['lat']):
                lats.append(row['lat'])
            if pd.notna(row['lon']):
                lons.append(row['lon'])
        
        avg_lat = sum(lats) / len(lats) if lats else None
        avg_lon = sum(lons) / len(lons) if lons else None
        
        aggregated.append({
            'station_name': station_name,
            'lines': sorted(all_lines),
            'transport_types': sorted(all_types),
            'lat': avg_lat,
            'lon': avg_lon
        })
    
    output_df = pd.DataFrame(aggregated)
    output_df = output_df.sort_values('station_name').reset_index(drop=True)
    
    print(f"  After deduplication: {len(output_df)} unique stations")
    
    return output_df

def validate_and_output(output_df):
    """Validate and output the station list."""
    print("\nValidating output...")
    
    # Show transport type distribution
    print("\n  Transport types:")
    for t in ['S-Bahn', 'U-Bahn', 'Tram', 'Regio-RE', 'Regio-RB', 'FEX']:
        count = sum(t in tp for tp in output_df['transport_types'])
        print(f"    {t}: {count} stations")
    
    # Show line distribution
    print("\n  Lines found:")
    all_lines = set()
    for l in output_df['lines']:
        all_lines.update(l)
    
    # Group by transport type
    s_lines = [x for x in sorted(all_lines) if x.startswith('S') or x.startswith('RE') or x == 'FEX']
    u_lines = [x for x in sorted(all_lines) if x.startswith('U')]
    m_lines = [x for x in sorted(all_lines) if x.startswith('M')]
    
    print(f"    S/RE/FEX: {len(s_lines)} lines")
    print(f"    U-Bahn: {len(u_lines)} lines")
    print(f"    Tram (M): {len(m_lines)} lines")
    
    # Check key interchanges
    print("\n  Key interchanges:")
    key_names = ['Alexanderplatz', 'Hauptbahnhof', 'Warschauer', 'Kottbusser', 'Nollendorfplatz']
    for name in key_names:
        matches = output_df[output_df['station_name'].str.contains(name, case=False, na=False)]
        if len(matches) > 0:
            for _, row in matches.head(3).iterrows():
                print(f"    {row['station_name']}: {row['lines']} ({row['transport_types']})")
    
    # Format output
    output_rows = []
    for _, row in output_df.iterrows():
        lines_str = '; '.join(row['lines'])
        types_str = '; '.join(row['transport_types'])
        output_rows.append({
            'station_name': row['station_name'],
            'lines': lines_str,
            'transport_types': types_str,
            'lat': row['lat'],
            'lon': row['lon']
        })
    
    final_df = pd.DataFrame(output_rows)
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nOutput written to: {OUTPUT_FILE}")
    print(f"Total stations: {len(final_df)}")
    
    return final_df

def main():
    print("=" * 60)
    print("Berlin Transport Station Extraction Pipeline")
    print("=" * 60)
    
    # Step 1: Load GTFS files
    routes, trips, stops = load_gtfs_files()
    
    # Step 2: Filter routes for all transport types (rail + subway + tram)
    print("\nFiltering routes by transport type...")
    
    # Get all relevant routes (rail + U-Bahn + tram)
    all_route_types = ROUTE_TYPE_RAIL | ROUTE_TYPE_SUBWAY | ROUTE_TYPE_TRAM
    filtered_routes = filter_routes_by_type(routes, all_route_types)
    
    print(f"  Filtered routes: {len(filtered_routes)}")
    
    # Show route types found
    route_types_found = filtered_routes['route_type'].unique()
    print(f"  Route types: {route_types_found}")
    
    # Show unique route names by type
    for rt in sorted(route_types_found):
        rt_routes = filtered_routes[filtered_routes['route_type'] == rt]
        names = rt_routes['route_short_name'].unique()
        type_name = get_route_type_name(rt, None)
        print(f"    {type_name}: {len(names)} routes")
    
    # Step 3: Filter trips to target routes
    print("\nFiltering trips to target routes...")
    target_trip_ids = set(filtered_routes['route_id'])
    target_trips = trips[trips['route_id'].isin(target_trip_ids)].copy()
    print(f"  Target trips: {len(target_trips)}")
    
    # Step 4: Load stop_times
    stop_times = load_stop_times_for_routes(target_trips)
    
    # Step 5: Build stop → station mapping
    stop_to_station, station_info = build_stop_to_station_mapping(stops)
    
    # Step 6: Extract stations
    station_data = extract_stations(
        stop_times, stop_to_station, station_info,
        filtered_routes, target_trips
    )
    
    # Step 7: Aggregate and deduplicate
    output_df = aggregate_and_deduplicate(station_data)
    
    # Step 8: Validate and output
    final_df = validate_and_output(output_df)
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print("=" * 60)
    
    return final_df

if __name__ == "__main__":
    main()