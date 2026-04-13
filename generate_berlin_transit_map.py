import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely.ops import unary_union


warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.cache_folder = "cache/osmnx"

# -- SETTINGS -----------------------------------------------------------------
PLACE = "Berlin, Germany"
OUTPUT_HTML = Path("index.html")
GRAPH_CACHE = Path("cache/berlin_walk_graph.graphml")

WALK_LEVELS = [5, 10]
WALK_SPEED_KMH = 4.5

# Berlin is in UTM zone 33N. Buffering in a projected CRS keeps distances in m.
METRIC_CRS = "EPSG:25833"
NODE_BUFFER_METERS = 70
SIMPLIFY_TOLERANCE_METERS = 18

STATION_TAGS = {
    "railway": ["station", "halt"],
    "station": ["subway", "light_rail"],
    "subway": "yes",
}

WALK_COLORS = {
    5: "#7fa77c",
    10: "#dce7db",
}
STATION_FILL = "#2f352f"
# -----------------------------------------------------------------------------


def log(message):
    print(message, flush=True)


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

    zones = {}
    for level in sorted(walk_levels):
        selected = reached_buffers[reached_buffers["travel_time"] <= level * 60]
        if selected.empty:
            raise RuntimeError(f"No reachable walking-network nodes were found for {level} minutes.")

        log(f"Building {level}-minute walking polygon from {len(selected)} network nodes")
        zone = unary_union(selected.geometry)

        if not zone.is_valid:
            zone = zone.buffer(0)

        zone = zone.simplify(SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)
        zones[level] = gpd.GeoSeries([zone], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]

    return zones


def add_map_panel(map_object, station_count):
    panel_html = f"""
    <style>
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
      <div class="su-kicker">Berlin rapid transit</div>
      <div class="su-title">5 and 10 min walk to S-Bahn or U-Bahn</div>
      <div class="su-legend">
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{WALK_COLORS[5]}"></span>5-minute walk
        </div>
        <div class="su-legend-row">
          <span class="su-swatch" style="background:{WALK_COLORS[10]}"></span>10-minute walk
        </div>
      </div>
      <div class="su-meta">
        {station_count} station origins. Trams excluded. Walking speed: {WALK_SPEED_KMH:g} km/h.
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(panel_html))


def render_map(place, stations, walk_zones):
    log("Rendering Folium map...")
    place_boundary = ox.geocode_to_gdf(place).to_crs("EPSG:4326")
    centre = place_boundary.geometry.unary_union.centroid
    map_object = folium.Map(
        location=[centre.y, centre.x],
        zoom_start=11,
        tiles="CartoDB positron",
        control_scale=True,
    )

    for level in sorted(walk_zones, reverse=True):
        color = WALK_COLORS[level]
        folium.GeoJson(
            gpd.GeoDataFrame(
                {"minutes": [level], "mode": ["walk"]},
                geometry=[walk_zones[level]],
                crs="EPSG:4326",
            ).__geo_interface__,
            name=f"{level} min walk from S-Bahn/U-Bahn",
            style_function=lambda _, c=color: {
                "fillColor": c,
                "color": c,
                "weight": 2,
                "fillOpacity": 0.62,
                "opacity": 0.95,
            },
            tooltip=f"{level} min walk to S-Bahn or U-Bahn",
        ).add_to(map_object)

    station_layer = folium.FeatureGroup(name="S-Bahn and U-Bahn stations", show=True)
    for station in stations.itertuples():
        folium.CircleMarker(
            location=[station.geometry.y, station.geometry.x],
            radius=3.8,
            color="#FFFFFF",
            weight=1.2,
            fill=True,
            fill_color=STATION_FILL,
            fill_opacity=0.95,
            tooltip=f"{station.kind}: {station.name}",
        ).add_to(station_layer)
    station_layer.add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)
    add_map_panel(map_object, len(stations))
    map_object.save(str(OUTPUT_HTML))
    log(f"Saved map to: {OUTPUT_HTML}")


def run_analysis(place=PLACE):
    log(f"Starting Berlin S/U-Bahn walkability map for: {place}")
    stations = fetch_su_stations(place)
    if stations.empty:
        raise RuntimeError("No S-Bahn or U-Bahn stations found.")

    graph = load_walk_graph(place)
    graph = add_walk_travel_times(graph, WALK_SPEED_KMH)

    walk_zones = build_walk_zones(graph, stations, WALK_LEVELS)
    render_map(place, stations, walk_zones)


if __name__ == "__main__":
    run_analysis()
