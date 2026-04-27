import io
import os
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.ops import unary_union
from shapely.validation import make_valid


warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.cache_folder = "cache/osmnx"

# -- SETTINGS -----------------------------------------------------------------
PLACE = "Berlin, Germany"
OUTPUT_HTML = Path("index.html")
CACHE_DIR = Path("cache")
GRAPH_CACHE = CACHE_DIR / "berlin_walk_graph.graphml"
TRAM_STOPS_CACHE = CACHE_DIR / "tram_stops.geojson"

WALK_LEVELS = [5, 10]
WALK_SPEED_KMH = 4.5

BASEMAPDE_TILE_URL = (
    "https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/"
    "de_basemapde_web_raster_farbe/default/GLOBAL_WEBMERCATOR/{z}/{y}/{x}.png"
)
BASEMAPDE_ATTRIBUTION = "© GeoBasis-DE / BKG, CC BY 4.0"
BASEMAPS = [
    ("OpenStreetMap", "OpenStreetMap", None),
    (BASEMAPDE_TILE_URL, "basemap.de Web Raster Farbe", BASEMAPDE_ATTRIBUTION),
    ("CartoDB positron", "CartoDB Positron", None),
    ("CartoDB dark_matter", "CartoDB Dark Matter", None),
]
ENVIRONMENTAL_JUSTICE_WMS_URL = "https://gdi.berlin.de/services/wms/ua_umweltgerechtigkeit2023"
ENVIRONMENTAL_JUSTICE_LAYER = "z_gesamt_umwelt2023"
ENVIRONMENTAL_JUSTICE_LAYER_NAME = "Umweltgerechtigkeit 2023/2024"
ENVIRONMENTAL_JUSTICE_OPACITY = 0.75

WOHNLAGEN_WMS_URL = "https://gdi.berlin.de/services/wms/wohnlagenadr2024"
WOHNLAGEN_LAYER = "wohnlagenadr2024"
WOHNLAGEN_LAYER_NAME = "Mietspiegel 2024 (Wohnlagen)"
WOHNLAGEN_OPACITY = 0.75
TRAM_STOPS_WFS_URL = (
    "https://gdi.berlin.de/services/wfs/oepnv_ungestoert"
    "?SERVICE=WFS"
    "&VERSION=2.0.0"
    "&REQUEST=GetFeature"
    "&TYPENAMES=oepnv_ungestoert:b_tramstopp"
    "&OUTPUTFORMAT=application/json"
    "&SRSNAME=EPSG:4326"
)
TRAM_WALK_MINUTES = 3

# Berlin is in UTM zone 33N. Buffering in a projected CRS keeps distances in m.
METRIC_CRS = "EPSG:25833"
NODE_BUFFER_METERS = 70
EDGE_BUFFER_METERS = 35
STATION_ACCESS_BUFFER_METERS = 90
SIMPLIFY_TOLERANCE_METERS = 18

STATION_TAGS = {
    "railway": ["station", "halt"],
    "station": ["subway", "light_rail"],
    "subway": "yes",
}

WALK_COLORS = {
    5: "#f075c3",
    10: "#ae54c4",
}
WALK_OPACITY = 0.5
EDGE_DIFFUSION_METERS = {
    5: 60,
    10: 120,
}
EDGE_FEATHER_OPACITIES = [0.18, 0.12, 0.07, 0.035]
STATION_FILL = "#313873"
STATION_ICON_SIZE = 12

TRAM_WALK_COLOR = "#d4a0dc"
TRAM_WALK_OPACITY = 0.65
TRAM_EDGE_DIFFUSION_METERS = 80
TRAM_STOP_FILL = "#a870c0"
TRAM_STOP_ICON_SIZE = 9

REGIONAL_STATIONS_CSV = Path("Haltestellen_VBB/UMBW.CSV")
OUTSIDE_S_SEARCH_TERMS = [
    # North
    "Oranienburg", "Lehnitz", "Borgsdorf", "Birkenwerder", "Hohen Neuendorf", 
    "Bergfelde", "Schönfließ", "Mühlenbeck-Mönchmühle", "Hennigsdorf",
    # Northeast
    "Bernau", "Bernau-Friedenstal", "Zepernick", "Röntgental",
    # East / Southeast
    "Strausberg Nord", "Strausberg Stadt", "Hegermühle", "Strausberg", 
    "Petershagen Nord", "Fredersdorf", "Neuenhagen", "Hoppegarten", "Birkenstein", 
    "Erkner", "Königs Wusterhausen", "Wildau", "Zeuthen", "Eichwalde",
    # South
    "Flughafen BER", "Waßmannsdorf", "Schönefeld (bei Berlin)", "Blankenfelde", 
    "Mahlow", "Teltow Stadt",
    # Southwest
    "Potsdam Hauptbahnhof", "Babelsberg", "Griebnitzsee"
]
REGIONAL_SEARCH_TERMS = [
    "Oranienburg", "Bernau", "Königs Wusterhausen", "Ludwigsfelde",
    "Potsdam Hbf", "Nauen", "Brieselang", "Falkensee", "Erkner", "Strausberg", 
    "Fürstenwalde", "Flughafen BER", "Dallgow-Döberitz", "Elstal", "Wustermark",
    "Werder (Havel)", "Teltow", "Großbeeren", "Birkengrund", "Rangsdorf", "Dahlewitz",
    "Blankenfelde", "Schönefeld (bei Berlin)", "Hennigsdorf"
]
REGIONAL_WALK_MINUTES = 20
REGIONAL_WALK_COLOR = "#e85db4"
REGIONAL_WALK_OPACITY = 0.4
REGIONAL_EDGE_DIFFUSION_METERS = 150
REGIONAL_STATION_FILL = "#9c27b0"

WALK_ZONE_5_CACHE = CACHE_DIR / "walk_zone_5.geojson"
WALK_ZONE_10_CACHE = CACHE_DIR / "walk_zone_10.geojson"
TRAM_ZONE_3_CACHE = CACHE_DIR / "tram_zone_3.geojson"
REGIONAL_ZONE_20_CACHE = CACHE_DIR / "regional_zone_20.geojson"

FREQUENCY_CACHE = CACHE_DIR / "station_frequencies.csv"
FREQUENCY_COLOR = "#00008b"
# -----------------------------------------------------------------------------


def log(message):
    print(message, flush=True)


def robust_union(geoms):
    """Safely union a list of geometries, handling invalid or empty ones."""
    if not geoms:
        return None
        
    clean_geoms = []
    for g in geoms:
        if g is None or g.is_empty:
            continue
        
        # Ensure validity
        if not g.is_valid:
            try:
                g = make_valid(g)
            except Exception:
                g = g.buffer(0)
            
        # For walk zones, we primarily care about polygons
        if g.geom_type == 'GeometryCollection':
            for sub in g.geoms:
                if sub.geom_type in ['Polygon', 'MultiPolygon'] and not sub.is_empty:
                    clean_geoms.append(sub)
        elif not g.is_empty:
            clean_geoms.append(g)
            
    if not clean_geoms:
        return None
        
    try:
        res = unary_union(clean_geoms)
        if not res.is_valid:
            res = make_valid(res)
        return res
    except Exception as e:
        log(f"Warning: robust_union encountered an error: {e}. Attempting iterative union.")
        try:
            res = clean_geoms[0]
            for g in clean_geoms[1:]:
                res = res.union(g)
            return res
        except Exception:
            return None


def text_value(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple, set)):
        return " ".join(str(item) for item in value).lower()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).lower()


def row_text(row):
    fields = [
        "name",
        "station",
        "network",
        "operator",
        "line",
        "route_ref",
        "railway",
        "subway",
        "train",
        "tram",
    ]
    return " ".join(text_value(row.get(field)) for field in fields)


def is_tram(row):
    text = row_text(row)
    return (
        text_value(row.get("railway")) == "tram_stop"
        or text_value(row.get("station")) == "tram"
        or text_value(row.get("tram")) == "yes"
        or "tram" in text
        or "strassenbahn" in text
        or "straßenbahn" in text
    )


def is_subway_station(row):
    text = row_text(row)
    return (
        text_value(row.get("station")) == "subway"
        or text_value(row.get("subway")) == "yes"
        or "u-bahn" in text
        or "ubahn" in text
    )


def is_sbahn_station(row):
    text = row_text(row)
    return (
        text_value(row.get("station")) == "light_rail"
        or "s-bahn" in text
        or "sbahn" in text
        or "s-bahn berlin" in text
    )


def station_kind(row):
    kinds = []
    if is_sbahn_station(row):
        kinds.append("S-Bahn")
    if is_subway_station(row):
        kinds.append("U-Bahn")
    return " + ".join(kinds) if kinds else "S/U-Bahn"


def station_name(row):
    name = row.get("name")
    if name is None or pd.isna(name) or not str(name).strip():
        return "S/U-Bahn station"
    return str(name)


def combine_station_kinds(values):
    kinds = []
    text = " ".join(str(value) for value in values)
    if "S-Bahn" in text:
        kinds.append("S-Bahn")
    if "U-Bahn" in text:
        kinds.append("U-Bahn")
    return " + ".join(kinds) if kinds else "S/U-Bahn"


def station_symbol(kind):
    return "S" if "S-Bahn" in kind else "U"


def fetch_su_stations(place):
    log("Downloading S-Bahn and U-Bahn stations...")
    raw = ox.features_from_place(place, tags=STATION_TAGS)
    raw = raw[raw.geometry.notna()].copy()
    raw["is_su_station"] = raw.apply(
        lambda row: is_sbahn_station(row) or is_subway_station(row),
        axis=1,
    )
    raw["is_tram_only"] = raw.apply(
        lambda row: is_tram(row) and not row["is_su_station"],
        axis=1,
    )
    raw = raw[raw["is_su_station"] & ~raw["is_tram_only"]]

    if raw.empty:
        return gpd.GeoDataFrame(columns=["name", "kind", "geometry"], crs="EPSG:4326")

    stations = raw.to_crs("EPSG:4326").copy()
    stations["geometry"] = stations.geometry.apply(
        lambda geometry: geometry if geometry.geom_type == "Point" else geometry.centroid
    )
    stations["name"] = stations.apply(station_name, axis=1)
    stations["kind"] = stations.apply(station_kind, axis=1)
    stations["lon"] = stations.geometry.x.round(5)
    stations["lat"] = stations.geometry.y.round(5)
    stations = stations.drop_duplicates(subset=["name", "kind", "lon", "lat"])
    stations["station_key"] = stations.apply(
        lambda row: (
            f"unnamed-{row['lon']:.3f}-{row['lat']:.3f}"
            if row["name"] == "S/U-Bahn station"
            else row["name"]
        ),
        axis=1,
    )

    stations = stations.to_crs(METRIC_CRS).dissolve(
        by="station_key",
        aggfunc={"name": "first", "kind": combine_station_kinds},
    )
    stations["geometry"] = stations.geometry.centroid
    stations = stations.to_crs("EPSG:4326")[["name", "kind", "geometry"]].reset_index(drop=True)

    log(f"Found {len(stations)} S-Bahn/U-Bahn station points")
    return stations


def fetch_stations_from_csv(search_terms):
    log(f"Fetching {len(search_terms)} station groups from VBB CSV...")
    if not REGIONAL_STATIONS_CSV.exists():
        log(f"Warning: CSV not found at {REGIONAL_STATIONS_CSV}")
        return gpd.GeoDataFrame(columns=["name", "geometry"], crs="EPSG:4326")

    stations = []
    import csv
    try:
        with open(REGIONAL_STATIONS_CSV, mode='r', encoding='latin-1') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if len(row) < 7: continue
                name = row[0]
                # if row[2] != 'Bauwerk': continue
                
                for term in search_terms:
                    if term.lower() in name.lower():
                        # NEVER use replacement or bus services
                        if any(x in name.lower() for x in ["ersatz", "bus", "sev", "ersatzverkehr", "(ers)"]):
                            continue
                            
                        # Priority for actual stations over bus stops sharing the name
                        is_likely_station = any(x in name.lower() for x in ["bahnhof", "hbf", "bhf", "s ", "s+u"])
                        is_exact = name.strip().lower() == term.lower()
                        
                        if is_likely_station or is_exact:
                            try:
                                lon = float(row[5].replace(',', '.'))
                                lat = float(row[6].replace(',', '.'))
                                stations.append({"name": name, "geometry": Point(lon, lat), "term": term})
                                break 
                            except (ValueError, IndexError):
                                continue
    except Exception as e:
        log(f"Error reading CSV: {e}")

    df = gpd.GeoDataFrame(stations, crs="EPSG:4326")
    if not df.empty:
        # Score candidates to pick the best one for each term (e.g. prefer Hbf over a bus stop)
        def score_name(n, term):
            n = n.lower()
            term = term.lower()
            score = 0
            if "hbf" in n or "hauptbahnhof" in n: score += 10
            if "bahnhof" in n or "bhf" in n: score += 8
            if "s " in n or "s+u" in n: score += 7
            
            # Penalize replacement/bus services heavily
            if any(x in n for x in ["ersatz", "bus", "sev", "ersatzverkehr"]):
                score -= 15
                
            # Bonus for exact or very close matches
            if n == term: score += 5
            elif n.startswith(term): score += 2
            
            return score

        df["score"] = df.apply(lambda r: score_name(r["name"], r["term"]), axis=1)
        df = df.sort_values("score", ascending=False).drop_duplicates(subset=["term"])
    
    log(f"Found {len(df)} stations in CSV")
    return df


def load_cached_polygon(cache_path):
    if cache_path.exists():
        log(f"Loading cached walk zone from {cache_path}...")
        gdf = gpd.read_file(cache_path)
        if not gdf.empty:
            return gdf.geometry.unary_union
    return None


def save_cached_polygon(polygon, cache_path):
    log(f"Saving computed walk zone to {cache_path}...")
    # polygon might be a single Polygon or MultiPolygon. 
    # Wrap it in a list to create a GeoDataFrame.
    gpd.GeoDataFrame({"geometry": [polygon]}, crs="EPSG:4326").to_file(str(cache_path), driver="GeoJSON")


def add_walk_travel_times(graph, walk_speed_kmh):
    meters_per_second = walk_speed_kmh * 1000 / 3600
    for _, _, _, data in graph.edges(keys=True, data=True):
        data["travel_time"] = float(data.get("length", 0) or 0) / meters_per_second
    return graph


def load_walk_graph(place):
    if GRAPH_CACHE.exists():
        log(f"Loading cached walking network from {GRAPH_CACHE}...")
        return ox.load_graphml(str(GRAPH_CACHE))

    log("Downloading Berlin walking network. The first Berlin-wide run can take a while...")
    graph = ox.graph_from_place(place, network_type="walk", simplify=True)
    graph = add_walk_travel_times(graph, WALK_SPEED_KMH)
    GRAPH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, filepath=str(GRAPH_CACHE))
    return graph


def graph_edge_geometry(graph, u, v, data):
    geometry = data.get("geometry")
    if geometry is not None:
        return geometry

    start = graph.nodes[u]
    end = graph.nodes[v]
    return LineString([(start["x"], start["y"]), (end["x"], end["y"])])


def load_tram_stops():
    if TRAM_STOPS_CACHE.exists():
        log(f"Loading cached tram stops from {TRAM_STOPS_CACHE}...")
        return gpd.read_file(TRAM_STOPS_CACHE)

    log("Downloading tram stops from Berlin WFS...")
    response = requests.get(TRAM_STOPS_WFS_URL, timeout=60)
    response.raise_for_status()
    gdf = gpd.read_file(io.StringIO(response.text))
    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf.to_crs("EPSG:4326")
    gdf["geometry"] = gdf.geometry.apply(
        lambda g: g if g.geom_type == "Point" else g.centroid
    )
    # Normalise stop name — column may be 'name', 'bezeichnung', or similar
    name_col = next((c for c in gdf.columns if c.lower() in ("name", "bezeichnung", "hst_name")), None)
    gdf["name"] = gdf[name_col].fillna("Tram stop").astype(str) if name_col else "Tram stop"
    gdf = gdf[["name", "geometry"]]
    TRAM_STOPS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(str(TRAM_STOPS_CACHE), driver="GeoJSON")
    log(f"Found {len(gdf)} tram stops")
    return gdf


def build_walk_zones(graph, stations, walk_levels):
    max_walk_minutes = max(walk_levels)
    log(f"Computing {', '.join(str(level) for level in walk_levels)}-minute walking reachability...")
    max_seconds = max_walk_minutes * 60

    nearest_nodes = ox.distance.nearest_nodes(
        graph,
        X=stations.geometry.x.to_numpy(),
        Y=stations.geometry.y.to_numpy(),
    )
    source_nodes = list(dict.fromkeys(nearest_nodes))
    if not source_nodes:
        raise RuntimeError("No station origins could be matched to the walking network.")

    log(f"Using {len(source_nodes)} unique walking-network origins")
    travel_times = nx.multi_source_dijkstra_path_length(
        graph,
        source_nodes,
        cutoff=max_seconds,
        weight="travel_time",
    )
    if not travel_times:
        raise RuntimeError("No reachable walking-network nodes were found.")

    nodes = ox.graph_to_gdfs(graph, edges=False)
    reached_nodes = nodes.loc[list(travel_times)].copy()
    reached_nodes["travel_time"] = reached_nodes.index.map(travel_times)
    reached_buffers = reached_nodes.to_crs(METRIC_CRS)
    reached_buffers["geometry"] = reached_buffers.geometry.buffer(NODE_BUFFER_METERS)
    station_buffers = gpd.GeoSeries(
        stations.geometry,
        crs="EPSG:4326",
    ).to_crs(METRIC_CRS).buffer(STATION_ACCESS_BUFFER_METERS)

    zones = {}
    for level in sorted(walk_levels):
        max_seconds = level * 60
        selected = reached_buffers[reached_buffers["travel_time"] <= level * 60]
        if selected.empty:
            raise RuntimeError(f"No reachable walking-network nodes were found for {level} minutes.")

        log(f"Building {level}-minute walking polygon from {len(selected)} network nodes")
        selected_node_ids = set(selected.index)
        edge_geometries = [
            graph_edge_geometry(graph, u, v, data)
            for u, v, _, data in graph.edges(keys=True, data=True)
            if u in selected_node_ids
            and v in selected_node_ids
            and travel_times.get(u, float("inf")) <= max_seconds
            and travel_times.get(v, float("inf")) <= max_seconds
        ]
        edge_buffers = gpd.GeoSeries(
            edge_geometries,
            crs="EPSG:4326",
        ).to_crs(METRIC_CRS).buffer(EDGE_BUFFER_METERS)
        
        zone = robust_union(
            list(selected.geometry)
            + list(edge_buffers)
            + list(station_buffers)
        )

        if zone:
            zone = zone.simplify(SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)
            zones[level] = gpd.GeoSeries([zone], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        else:
            zones[level] = None

    return zones


def build_outside_walk_zones(stations, minutes_per_station):
    """
    stations: GeoDataFrame of stations.
    minutes_per_station: dict {station_name: [min1, min2, ...]}
    Returns: dict {min: [polygons]}
    """
    results = {}
    walk_speed_mps = WALK_SPEED_KMH * 1000 / 3600
    
    for station in stations.itertuples():
        levels = minutes_per_station.get(station.name)
        if not levels: continue
        
        log(f"Processing outside station: {station.name} for levels {levels}...")
        max_min = max(levels)
        buffer_dist = (max_min * 60 * walk_speed_mps) + 500
        
        try:
            G = ox.graph_from_point((station.geometry.y, station.geometry.x), 
                                     dist=buffer_dist, network_type='walk', simplify=True)
            G = add_walk_travel_times(G, WALK_SPEED_KMH)
            source_node = ox.distance.nearest_nodes(G, X=station.geometry.x, Y=station.geometry.y)
            
            # Precompute all needed travel times up to the max requested level
            all_travel_times = nx.single_source_dijkstra_path_length(G, source_node, cutoff=max_min*60, weight="travel_time")
            
            nodes = ox.graph_to_gdfs(G, edges=False)
            
            for mins in levels:
                travel_times = {node: dist for node, dist in all_travel_times.items() if dist <= mins*60}
                if not travel_times: continue
                
                reached_nodes = nodes.loc[list(travel_times.keys())].to_crs(METRIC_CRS)
                reached_buffers = reached_nodes.buffer(NODE_BUFFER_METERS)
                
                selected_node_ids = set(travel_times.keys())
                edge_geometries = [
                    graph_edge_geometry(G, u, v, data)
                    for u, v, _, data in G.edges(keys=True, data=True)
                    if u in selected_node_ids and v in selected_node_ids
                ]
                # Filter degenerate edges
                edge_geometries = [g for g in edge_geometries if g.length > 1e-9]
                edge_buffers = gpd.GeoSeries(edge_geometries, crs="EPSG:4326").to_crs(METRIC_CRS).buffer(EDGE_BUFFER_METERS)
                
                station_buffer = gpd.GeoSeries([station.geometry], crs="EPSG:4326").to_crs(METRIC_CRS).buffer(STATION_ACCESS_BUFFER_METERS)
                
                zone = robust_union(list(reached_buffers) + list(edge_buffers) + list(station_buffer))
                if zone:
                    zone = zone.simplify(SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)
                    zone = gpd.GeoSeries([zone], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
                    if mins not in results: results[mins] = []
                    results[mins].append(zone)
                
        except Exception as e:
            log(f"Warning: Could not compute zones for {station.name}: {e}")
            
    return results


def build_edge_feathers(geometry, spread_meters):
    if geometry is None or geometry.is_empty:
        return []
    
    try:
        if not geometry.is_valid:
            geometry = make_valid(geometry)
        
        # Convert to metric CRS for buffering
        metric_series = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(METRIC_CRS)
        metric_geometry = metric_series.iloc[0]
        
        # Clean up metric geometry
        if not metric_geometry.is_valid:
            metric_geometry = make_valid(metric_geometry)
        metric_geometry = metric_geometry.buffer(0)
        
        # Ensure we only have Polygons/MultiPolygons
        if metric_geometry.geom_type == 'GeometryCollection':
            polys = [g for g in metric_geometry.geoms if g.geom_type in ['Polygon', 'MultiPolygon']]
            if not polys: return []
            metric_geometry = unary_union(polys)

        previous_geometry = metric_geometry
        feather_geometries = []

        num_steps = len(EDGE_FEATHER_OPACITIES)
        for step in range(1, num_steps + 1):
            distance = spread_meters * step / num_steps
            
            try:
                expanded_geometry = metric_geometry.buffer(distance)
            except Exception:
                # Fallback: tiny simplification can often fix topological errors
                expanded_geometry = metric_geometry.simplify(0.05).buffer(distance)
                
            if not expanded_geometry.is_valid:
                expanded_geometry = make_valid(expanded_geometry)
            
            # difference() can also be sensitive
            try:
                feather_geometry = expanded_geometry.difference(previous_geometry)
            except Exception:
                feather_geometry = expanded_geometry.buffer(0).difference(previous_geometry.buffer(0))

            if not feather_geometry.is_empty:
                if not feather_geometry.is_valid:
                    feather_geometry = make_valid(feather_geometry)
                
                feather_geometry = feather_geometry.simplify(
                    SIMPLIFY_TOLERANCE_METERS,
                    preserve_topology=True,
                )
                
                # Filter for polygons only
                if feather_geometry.geom_type in ['Polygon', 'MultiPolygon', 'GeometryCollection']:
                    feather_geometries.append(
                        gpd.GeoSeries([feather_geometry], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
                    )

            previous_geometry = expanded_geometry

        return feather_geometries
        
    except Exception as e:
        log(f"Warning: build_edge_feathers failed for a layer: {e}")
        return []


def filter_lines_for_category(lines_str, category):
    if not isinstance(lines_str, str) or not lines_str:
        return ""
        
    valid_lines = set()
    for item in lines_str.split(','):
        if not item: continue
        parts = item.split('|')
        name = parts[0].strip()
        rtype = parts[1].strip() if len(parts) > 1 else ""
        
        if category == "su":
            if name.startswith('S') and name[1:].isdigit(): valid_lines.add(name)
            elif name.startswith('U') and name[1:].isdigit(): valid_lines.add(name)
            elif rtype in ('109', '400', '1'): valid_lines.add(name)
        elif category == "tram":
            if rtype in ('0', '900'): valid_lines.add(name)
            elif name in ('M1', 'M2', 'M4', 'M5', 'M6', 'M8', 'M10', 'M13', 'M17'): valid_lines.add(name)
            elif name.isdigit() and len(name) <= 2: valid_lines.add(name)
        elif category == "regional":
            if name.startswith(('RE', 'RB', 'FEX', 'HBX', 'IRE', 'IC', 'ICE')): valid_lines.add(name)
            elif rtype in ('100', '101', '102'): valid_lines.add(name)
            
    import re
    def natural_keys(text):
        return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
        
    return ', '.join(sorted(valid_lines, key=natural_keys))


def assign_frequencies(stations, freq_df, category="su"):
    if freq_df is None or freq_df.empty or stations.empty:
        stations["departures"] = 0
        stations["lines"] = ""
        return stations

    stations_metric = stations.to_crs(METRIC_CRS)
    freq_gdf = gpd.GeoDataFrame(
        freq_df, 
        geometry=gpd.points_from_xy(freq_df.stop_lon, freq_df.stop_lat),
        crs="EPSG:4326"
    ).to_crs(METRIC_CRS)

    tree = cKDTree(np.array(list(freq_gdf.geometry.apply(lambda p: (p.x, p.y)))))
    coords = np.array(list(stations_metric.geometry.apply(lambda p: (p.x, p.y))))
    
    idx = tree.query_ball_point(coords, r=200)
    
    departures = []
    lines_list = []
    for i in idx:
        if i:
            deps = freq_gdf.iloc[i]['daily_departures'].sum()
            departures.append(deps)
            if 'lines' in freq_gdf.columns:
                all_lines_str = ','.join([str(l) for l in freq_gdf.iloc[i]['lines'].dropna()])
                lines_list.append(filter_lines_for_category(all_lines_str, category))
            else:
                lines_list.append("")
        else:
            departures.append(0)
            lines_list.append("")
            
    stations["departures"] = departures
    stations["lines"] = lines_list
    return stations


def build_frequency_overlays(stations_list, all_walk_zones=None):
    all_stats = pd.concat([s for s in stations_list if not s.empty], ignore_index=True)
    if all_stats.empty or "departures" not in all_stats.columns:
        return {}
        
    all_stats = gpd.GeoDataFrame(all_stats, geometry="geometry", crs="EPSG:4326").to_crs(METRIC_CRS)
    overlays = {}
    
    if all_walk_zones is not None and not all_walk_zones.is_empty:
        if isinstance(all_walk_zones, gpd.GeoSeries):
            all_walk_zones_metric = all_walk_zones.to_crs(METRIC_CRS).unary_union
        else:
            all_walk_zones_metric = gpd.GeoSeries([all_walk_zones], crs="EPSG:4326").to_crs(METRIC_CRS).unary_union
    else:
        all_walk_zones_metric = None
    
    # Tier 1: High frequency (> 1200 departures/day)
    tier1 = all_stats[all_stats["departures"] > 1200]
    if not tier1.empty:
        geom = unary_union(tier1.geometry.buffer(250)).simplify(15)
        if all_walk_zones_metric is not None and not all_walk_zones_metric.is_empty:
            geom = geom.intersection(all_walk_zones_metric)
        if not geom.is_empty:
            overlays['high'] = gpd.GeoSeries([geom], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        
    # Tier 2: Medium frequency (400 - 1200 departures/day)
    tier2 = all_stats[(all_stats["departures"] >= 400) & (all_stats["departures"] <= 1200)]
    if not tier2.empty:
        geom = unary_union(tier2.geometry.buffer(175)).simplify(15)
        if all_walk_zones_metric is not None and not all_walk_zones_metric.is_empty:
            geom = geom.intersection(all_walk_zones_metric)
        if not geom.is_empty:
            overlays['medium'] = gpd.GeoSeries([geom], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
        
    return overlays


def add_map_panel(map_object, station_count, tram_stop_count, regional_count):
    panel_html = f"""
    <style>
      .leaflet-freq_pane-pane {{
        mix-blend-mode: multiply;
      }}
      .su-panel {{
        position: fixed;
        top: 28px;
        left: 28px;
        z-index: 1000;
        width: 340px;
        max-width: calc(100vw - 56px);
        background: rgba(246, 247, 241, 0.94);
        border: 1px solid rgba(47, 53, 47, 0.16);
        border-radius: 6px;
        box-shadow: 0 16px 45px rgba(47, 53, 47, 0.16);
        color: #262824;
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        padding: 16px 18px;
        line-height: 1.2;
      }}
      .su-kicker {{
        color: #667064;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0;
        margin-bottom: 6px;
        text-transform: uppercase;
      }}
      .su-title {{
        font-size: 20px;
        font-weight: 760;
        line-height: 1.18;
        margin-bottom: 10px;
      }}
      .su-meta {{
        color: #5f665d;
        font-size: 13px;
        line-height: 1.45;
      }}
      .su-legend {{
        display: grid;
        gap: 6px;
        margin-bottom: 10px;
      }}
      .su-legend-row {{
        align-items: center;
        display: flex;
      }}
      .su-swatch {{
        display: inline-block;
        width: 14px;
        height: 14px;
        margin-right: 7px;
        vertical-align: -2px;
        border: 1px solid rgba(47, 53, 47, 0.28);
      }}
      .station-marker-anchor {{
        background: transparent;
        border: 0;
      }}
      .station-symbol {{
        align-items: center;
        background: {STATION_FILL};
        border: 1px solid rgba(255, 255, 255, 0.9);
        box-sizing: border-box;
        color: #ffffff;
        display: flex;
        font-family: Arial, Helvetica, sans-serif;
        font-size: 8px;
        font-weight: 700;
        height: {STATION_ICON_SIZE}px;
        justify-content: center;
        line-height: 1;
        width: {STATION_ICON_SIZE}px;
      }}
      .station-symbol-s {{
        border-radius: 50%;
      }}
      .station-symbol-u {{
        border-radius: 2px;
      }}
      .station-symbol-t {{
        background: {TRAM_STOP_FILL};
        border-radius: 50%;
        font-size: 7px;
        height: {TRAM_STOP_ICON_SIZE}px;
        width: {TRAM_STOP_ICON_SIZE}px;
      }}
      .station-symbol-r {{
        background: {REGIONAL_STATION_FILL};
        border-radius: 2px;
        font-size: 7px;
        height: {STATION_ICON_SIZE}px;
        width: {STATION_ICON_SIZE}px;
      }}
      @media (max-width: 560px) {{
        .su-panel {{
          top: 14px;
          left: 14px;
          max-width: calc(100vw - 28px);
          padding: 14px 15px;
        }}
        .su-title {{
          font-size: 18px;
        }}
      }}
    </style>
    <div class="su-panel">
      <div class="su-kicker">Berlin transit walkability</div>
      <div class="su-title">Walk distance to S, U, Tram &amp; Regional</div>
      <div class="su-legend">
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{WALK_COLORS[5]}"></span>5 min walk · S/U-Bahn
        </div>
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{WALK_COLORS[10]}"></span>10 min walk · S/U-Bahn
        </div>
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{TRAM_WALK_COLOR}"></span>3 min walk · Tram
        </div>
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{REGIONAL_WALK_COLOR}"></span>20 min walk · Regionalbahn
        </div>
      </div>
      <div class="su-meta">
        {station_count} S/U-Bahn · {tram_stop_count} Tram · {regional_count} Regional stations.
        <br>Dark blue overlays show high transit connection frequency.
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(panel_html))


def render_map(place, stations, walk_zones, tram_stops, tram_walk_zones, regional_stations, regional_walk_zone, frequency_overlays):
    log("Rendering Folium map...")
    place_boundary = ox.geocode_to_gdf(place).to_crs("EPSG:4326")
    centre = place_boundary.geometry.unary_union.centroid
    map_object = folium.Map(
        location=[centre.y, centre.x],
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )
    
    # Create a custom pane specifically for frequency overlays to sit above walk zones (400) but below shadows/WMS (500)
    folium.map.CustomPane("freq_pane", z_index=450, pointer_events=False).add_to(map_object)
    for i, (tiles, label, attribution) in enumerate(BASEMAPS):
        tile_options = {
            "tiles": tiles,
            "name": label,
            "show": (i == 0),
        }
        if attribution:
            tile_options["attr"] = attribution
            tile_options["max_zoom"] = 19
        folium.TileLayer(
            **tile_options,
        ).add_to(map_object)
    folium.raster_layers.WmsTileLayer(
        url=ENVIRONMENTAL_JUSTICE_WMS_URL,
        layers=ENVIRONMENTAL_JUSTICE_LAYER,
        name=ENVIRONMENTAL_JUSTICE_LAYER_NAME,
        fmt="image/png",
        transparent=True,
        version="1.3.0",
        attr="Umweltatlas Berlin / Geoportal Berlin, Datenlizenz Deutschland Zero 2.0",
        overlay=True,
        control=True,
        show=False,
        opacity=ENVIRONMENTAL_JUSTICE_OPACITY,
        pane="shadowPane",
    ).add_to(map_object)

    folium.raster_layers.WmsTileLayer(
        url=WOHNLAGEN_WMS_URL,
        layers=WOHNLAGEN_LAYER,
        name=WOHNLAGEN_LAYER_NAME,
        fmt="image/png",
        transparent=True,
        version="1.3.0",
        attr="Geoportal Berlin / Mietspiegel 2024",
        overlay=True,
        control=True,
        show=False,
        opacity=WOHNLAGEN_OPACITY,
        pane="shadowPane",
    ).add_to(map_object)

    for level in sorted(walk_zones, reverse=True):
        color = WALK_COLORS[level]
        zone_layer = folium.FeatureGroup(
            name=f"{level} min walk from S-Bahn/U-Bahn",
            show=True,
        )
        feathers = build_edge_feathers(walk_zones[level], EDGE_DIFFUSION_METERS[level])
        for feather_geometry, feather_opacity in reversed(
            list(zip(feathers, EDGE_FEATHER_OPACITIES))
        ):
            folium.GeoJson(
                gpd.GeoDataFrame(
                    {"minutes": [level], "mode": ["edge feather"]},
                    geometry=[feather_geometry],
                    crs="EPSG:4326",
                ).__geo_interface__,
                name=f"{level} min soft edge",
                control=False,
                style_function=lambda _, c=color, o=feather_opacity: {
                    "fillColor": c,
                    "color": c,
                    "weight": 0,
                    "fillOpacity": o,
                    "opacity": 0,
                },
            ).add_to(zone_layer)

        folium.GeoJson(
            gpd.GeoDataFrame(
                {"minutes": [level], "mode": ["walk"]},
                geometry=[walk_zones[level]],
                crs="EPSG:4326",
            ).__geo_interface__,
            name=f"{level} min walk area",
            control=False,
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 0,
                "fillOpacity": WALK_OPACITY,
                "opacity": 0,
            },
            tooltip=f"{level} min walk to S-Bahn or U-Bahn",
        ).add_to(zone_layer)
        zone_layer.add_to(map_object)

    tram_zone_layer = folium.FeatureGroup(
        name=f"{TRAM_WALK_MINUTES} min walk from Tram",
        show=True,
    )
    tram_zone = tram_walk_zones[TRAM_WALK_MINUTES]
    for feather_geometry, feather_opacity in reversed(
        list(zip(build_edge_feathers(tram_zone, TRAM_EDGE_DIFFUSION_METERS), EDGE_FEATHER_OPACITIES))
    ):
        folium.GeoJson(
            gpd.GeoDataFrame(
                {"minutes": [TRAM_WALK_MINUTES], "mode": ["edge feather"]},
                geometry=[feather_geometry],
                crs="EPSG:4326",
            ).__geo_interface__,
            name=f"{TRAM_WALK_MINUTES} min tram soft edge",
            control=False,
            style_function=lambda _, c=TRAM_WALK_COLOR, o=feather_opacity: {
                "fillColor": c,
                "color": c,
                "weight": 0,
                "fillOpacity": o,
                "opacity": 0,
            },
        ).add_to(tram_zone_layer)
    folium.GeoJson(
        gpd.GeoDataFrame(
            {"minutes": [TRAM_WALK_MINUTES], "mode": ["walk"]},
            geometry=[tram_zone],
            crs="EPSG:4326",
        ).__geo_interface__,
        name=f"{TRAM_WALK_MINUTES} min tram walk area",
        control=False,
        style_function=lambda _: {
            "fillColor": TRAM_WALK_COLOR,
            "color": TRAM_WALK_COLOR,
            "weight": 0,
            "fillOpacity": TRAM_WALK_OPACITY,
            "opacity": 0,
        },
        tooltip=f"{TRAM_WALK_MINUTES} min walk to Tram",
    ).add_to(tram_zone_layer)
    tram_zone_layer.add_to(map_object)

    if regional_walk_zone is not None:
        regional_zone_layer = folium.FeatureGroup(
            name=f"{REGIONAL_WALK_MINUTES} min walk from Regionalbahn",
            show=True,
        )
        for feather_geometry, feather_opacity in reversed(
            list(zip(build_edge_feathers(regional_walk_zone, REGIONAL_EDGE_DIFFUSION_METERS), EDGE_FEATHER_OPACITIES))
        ):
            folium.GeoJson(
                gpd.GeoDataFrame(
                    {"minutes": [REGIONAL_WALK_MINUTES], "mode": ["edge feather"]},
                    geometry=[feather_geometry],
                    crs="EPSG:4326",
                ).__geo_interface__,
                name=f"{REGIONAL_WALK_MINUTES} min regional soft edge",
                control=False,
                style_function=lambda _, c=REGIONAL_WALK_COLOR, o=feather_opacity: {
                    "fillColor": c,
                    "color": c,
                    "weight": 0,
                    "fillOpacity": o,
                    "opacity": 0,
                },
            ).add_to(regional_zone_layer)
        folium.GeoJson(
            gpd.GeoDataFrame(
                {"minutes": [REGIONAL_WALK_MINUTES], "mode": ["walk"]},
                geometry=[regional_walk_zone],
                crs="EPSG:4326",
            ).__geo_interface__,
            name=f"{REGIONAL_WALK_MINUTES} min regional walk area",
            control=False,
            style_function=lambda _: {
                "fillColor": REGIONAL_WALK_COLOR,
                "color": REGIONAL_WALK_COLOR,
                "weight": 0,
                "fillOpacity": REGIONAL_WALK_OPACITY,
                "opacity": 0,
            },
            tooltip=f"{REGIONAL_WALK_MINUTES} min walk to Regionalbahn station",
        ).add_to(regional_zone_layer)
        regional_zone_layer.add_to(map_object)

    # Add frequency overlays BEFORE the points, so they sit under the station icons
    if frequency_overlays:
        freq_layer = folium.FeatureGroup(name="Connection Frequency Overlay", show=True)
        
        def render_freq_tier(name, geom, base_opacity, feather_spread, tooltip):
            # Render feathers
            feathers = build_edge_feathers(geom, feather_spread)
            num_feathers = len(feathers)
            for i, feather_geometry in enumerate(reversed(feathers)):
                f_opacity = base_opacity * (0.1 + 0.8 * (i / max(1, num_feathers - 1)))
                folium.GeoJson(
                    gpd.GeoDataFrame({"tier": [name]}, geometry=[feather_geometry], crs="EPSG:4326").__geo_interface__,
                    name=f"{name} feather",
                    control=False,
                    style_function=lambda _, o=f_opacity: {
                        "fillColor": FREQUENCY_COLOR,
                        "color": "transparent",
                        "weight": 0,
                        "fillOpacity": o,
                    },
                    pane="freq_pane"
                ).add_to(freq_layer)
                
            # Render core
            folium.GeoJson(
                gpd.GeoDataFrame({"tier": [name]}, geometry=[geom], crs="EPSG:4326").__geo_interface__,
                name=name,
                control=False,
                style_function=lambda _: {
                    "fillColor": FREQUENCY_COLOR,
                    "color": "transparent",
                    "weight": 0,
                    "fillOpacity": base_opacity,
                },
                tooltip=tooltip,
                pane="freq_pane"
            ).add_to(freq_layer)

        if 'medium' in frequency_overlays:
            render_freq_tier("Medium Frequency", frequency_overlays['medium'], 0.16, 175, "Medium Frequency (400-1200 departures/day)")
            
        if 'high' in frequency_overlays:
            render_freq_tier("High Frequency", frequency_overlays['high'], 0.28, 250, "High Frequency (>1200 departures/day)")
            
        freq_layer.add_to(map_object)

    tram_stop_layer = folium.FeatureGroup(name="Tram stops", show=False)
    for stop in tram_stops.itertuples():
        lines_text = f" ({stop.lines})" if getattr(stop, 'lines', '') else ""
        folium.Marker(
            location=[stop.geometry.y, stop.geometry.x],
            icon=folium.DivIcon(
                class_name="station-marker-anchor",
                html=f'<div class="station-symbol station-symbol-t">T</div>',
                icon_size=(TRAM_STOP_ICON_SIZE, TRAM_STOP_ICON_SIZE),
                icon_anchor=(TRAM_STOP_ICON_SIZE // 2, TRAM_STOP_ICON_SIZE // 2),
            ),
            tooltip=f"Tram: {stop.name}{lines_text}",
        ).add_to(tram_stop_layer)
    tram_stop_layer.add_to(map_object)

    station_layer = folium.FeatureGroup(name="S-Bahn and U-Bahn stations", show=False)
    for station in stations.itertuples():
        symbol = station_symbol(station.kind)
        symbol_class = "station-symbol-s" if symbol == "S" else "station-symbol-u"
        lines_text = f" ({station.lines})" if getattr(station, 'lines', '') else ""
        folium.Marker(
            location=[station.geometry.y, station.geometry.x],
            icon=folium.DivIcon(
                class_name="station-marker-anchor",
                html=f'<div class="station-symbol {symbol_class}">{symbol}</div>',
                icon_size=(STATION_ICON_SIZE, STATION_ICON_SIZE),
                icon_anchor=(STATION_ICON_SIZE // 2, STATION_ICON_SIZE // 2),
            ),
            tooltip=f"{station.kind}: {station.name}{lines_text}",
        ).add_to(station_layer)
    station_layer.add_to(map_object)

    regional_layer = folium.FeatureGroup(name="Regional stations", show=False)
    for station in regional_stations.itertuples():
        lines_text = f" ({station.lines})" if getattr(station, 'lines', '') else ""
        folium.Marker(
            location=[station.geometry.y, station.geometry.x],
            icon=folium.DivIcon(
                class_name="station-marker-anchor",
                html=f'<div class="station-symbol station-symbol-r">R</div>',
                icon_size=(STATION_ICON_SIZE, STATION_ICON_SIZE),
                icon_anchor=(STATION_ICON_SIZE // 2, STATION_ICON_SIZE // 2),
            ),
            tooltip=f"Regionalbahn: {station.name}{lines_text}",
        ).add_to(regional_layer)
    regional_layer.add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)
    add_map_panel(map_object, len(stations), len(tram_stops), len(regional_stations))
    map_object.save(str(OUTPUT_HTML))
    log(f"Saved map to: {OUTPUT_HTML}")


def run_analysis(place=PLACE):
    log(f"Starting Berlin S/U-Bahn walkability map for: {place}")
    CACHE_DIR.mkdir(exist_ok=True)

    # 1. Fetch all station data
    stations = fetch_su_stations(place)
    tram_stops = load_tram_stops()
    
    # Use VBB CSV for outside stations
    all_outside_terms = list(set(OUTSIDE_S_SEARCH_TERMS + REGIONAL_SEARCH_TERMS))
    all_outside_df = fetch_stations_from_csv(all_outside_terms)

    # 2. Check for cached zones
    walk_zones = {
        5: load_cached_polygon(WALK_ZONE_5_CACHE),
        10: load_cached_polygon(WALK_ZONE_10_CACHE)
    }
    tram_walk_zone = load_cached_polygon(TRAM_ZONE_3_CACHE)
    regional_walk_zone = load_cached_polygon(REGIONAL_ZONE_20_CACHE)

    # 3. Compute missing zones
    needs_graph = any(z is None for z in walk_zones.values()) or tram_walk_zone is None
    graph = None
    if needs_graph:
        graph = load_walk_graph(place)
        graph = add_walk_travel_times(graph, WALK_SPEED_KMH)

    # 5/10 min S/U-Bahn zones
    missing_mins = [m for m, z in walk_zones.items() if z is None]
    if missing_mins:
        log(f"Computing {missing_mins} minute walking reachability...")
        inside_zones = build_walk_zones(graph, stations, missing_mins)
        
        # Outside S-Bahn
        outside_s_df = all_outside_df[all_outside_df["term"].isin(OUTSIDE_S_SEARCH_TERMS)]
        minutes_map = {s.name: missing_mins for s in outside_s_df.itertuples()}
        outside_zones_dict = build_outside_walk_zones(outside_s_df, minutes_map)
        
        for mins in missing_mins:
            merged = inside_zones[mins]
            if mins in outside_zones_dict:
                merged = robust_union([merged] + outside_zones_dict[mins])
            walk_zones[mins] = merged
            save_cached_polygon(walk_zones[mins], WALK_ZONE_5_CACHE if mins == 5 else WALK_ZONE_10_CACHE)

    # 3 min Tram zone
    if tram_walk_zone is None:
        log("Computing 3-minute Tram walking reachability...")
        tram_zones_dict = build_walk_zones(graph, tram_stops, [TRAM_WALK_MINUTES])
        tram_walk_zone = tram_zones_dict[TRAM_WALK_MINUTES]
        save_cached_polygon(tram_walk_zone, TRAM_ZONE_3_CACHE)

    # 20 min Regional zone
    if regional_walk_zone is None:
        regional_stations = all_outside_df[all_outside_df["term"].isin(REGIONAL_SEARCH_TERMS)]
        if not regional_stations.empty:
            log(f"Computing {REGIONAL_WALK_MINUTES}-minute Regional walking reachability...")
            minutes_map = {s.name: [REGIONAL_WALK_MINUTES] for s in regional_stations.itertuples()}
            outside_zones_dict = build_outside_walk_zones(regional_stations, minutes_map)
            if REGIONAL_WALK_MINUTES in outside_zones_dict:
                # Merge all regional zones
                regional_walk_zone = robust_union(outside_zones_dict[REGIONAL_WALK_MINUTES])
                save_cached_polygon(regional_walk_zone, REGIONAL_ZONE_20_CACHE)

    # 4. Process connection frequencies
    freq_df = None
    if FREQUENCY_CACHE.exists():
        freq_df = pd.read_csv(FREQUENCY_CACHE)
    else:
        log("No frequency cache found. Run compute_frequencies.py to add intensity layers.")
        
    regional_stations_df = all_outside_df[all_outside_df["term"].isin(REGIONAL_SEARCH_TERMS)].copy()
    stations = assign_frequencies(stations, freq_df, "su")
    tram_stops = assign_frequencies(tram_stops, freq_df, "tram")
    regional_stations_df = assign_frequencies(regional_stations_df, freq_df, "regional")
    # Combine all walk zones to clip the frequency overlays
    all_walk_zones_list = [z for z in walk_zones.values() if z is not None]
    if tram_walk_zone is not None:
        all_walk_zones_list.append(tram_walk_zone)
    if regional_walk_zone is not None:
        all_walk_zones_list.append(regional_walk_zone)
        
    all_walk_zones_combined = robust_union(all_walk_zones_list)
    
    frequency_overlays = build_frequency_overlays([stations, tram_stops, regional_stations_df], all_walk_zones_combined)

    # 5. Render
    tram_walk_zones_dict = {TRAM_WALK_MINUTES: tram_walk_zone}
    
    render_map(place, stations, walk_zones, tram_stops, tram_walk_zones_dict, regional_stations_df, regional_walk_zone, frequency_overlays)


if __name__ == "__main__":
    run_analysis()
