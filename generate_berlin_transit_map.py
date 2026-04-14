import io
import warnings
from pathlib import Path

import folium
import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd
import requests
from shapely.geometry import LineString
from shapely.ops import unary_union


warnings.filterwarnings("ignore")
ox.settings.use_cache = True
ox.settings.cache_folder = "cache/osmnx"

# -- SETTINGS -----------------------------------------------------------------
PLACE = "Berlin, Germany"
OUTPUT_HTML = Path("index.html")
GRAPH_CACHE = Path("cache/berlin_walk_graph.graphml")
TRAM_STOPS_CACHE = Path("cache/tram_stops.geojson")

WALK_LEVELS = [5, 10]
WALK_SPEED_KMH = 4.5

BASEMAPS = [
    ("OpenStreetMap",     "OpenStreetMap"),
    ("CartoDB positron",  "CartoDB Positron"),
    ("CartoDB dark_matter", "CartoDB Dark Matter"),
]
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

TRAM_WALK_COLOR = "#f0a830"
TRAM_WALK_OPACITY = 0.35
TRAM_EDGE_DIFFUSION_METERS = 40
TRAM_STOP_FILL = "#b85c0a"
TRAM_STOP_ICON_SIZE = 9
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
        zone = unary_union(
            list(selected.geometry)
            + list(edge_buffers)
            + list(station_buffers)
        )

        if not zone.is_valid:
            zone = zone.buffer(0)

        zone = zone.simplify(SIMPLIFY_TOLERANCE_METERS, preserve_topology=True)
        zone = unary_union([zone] + list(station_buffers))
        zones[level] = gpd.GeoSeries([zone], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]

    return zones


def build_edge_feathers(geometry, spread_meters):
    metric_geometry = gpd.GeoSeries([geometry], crs="EPSG:4326").to_crs(METRIC_CRS).iloc[0]
    previous_geometry = metric_geometry
    feather_geometries = []

    for step in range(1, len(EDGE_FEATHER_OPACITIES) + 1):
        distance = spread_meters * step / len(EDGE_FEATHER_OPACITIES)
        expanded_geometry = metric_geometry.buffer(distance)
        feather_geometry = expanded_geometry.difference(previous_geometry)

        if not feather_geometry.is_empty:
            if not feather_geometry.is_valid:
                feather_geometry = feather_geometry.buffer(0)
            feather_geometry = feather_geometry.simplify(
                SIMPLIFY_TOLERANCE_METERS,
                preserve_topology=True,
            )
            feather_geometries.append(
                gpd.GeoSeries([feather_geometry], crs=METRIC_CRS).to_crs("EPSG:4326").iloc[0]
            )

        previous_geometry = expanded_geometry

    return feather_geometries


def add_map_panel(map_object, station_count, tram_stop_count):
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
      <div class="su-title">Walk distance to S-Bahn, U-Bahn &amp; Tram</div>
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
      </div>
      <div class="su-meta">
        {station_count} S/U-Bahn stations · {tram_stop_count} tram stops. Walking speed: {WALK_SPEED_KMH:g} km/h.
      </div>
    </div>
    """
    map_object.get_root().html.add_child(folium.Element(panel_html))


def render_map(place, stations, walk_zones, tram_stops, tram_walk_zones):
    log("Rendering Folium map...")
    place_boundary = ox.geocode_to_gdf(place).to_crs("EPSG:4326")
    centre = place_boundary.geometry.unary_union.centroid
    map_object = folium.Map(
        location=[centre.y, centre.x],
        zoom_start=11,
        tiles=None,
        control_scale=True,
    )
    for i, (tiles, label) in enumerate(BASEMAPS):
        folium.TileLayer(
            tiles=tiles,
            name=label,
            show=(i == 0),
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

    tram_stop_layer = folium.FeatureGroup(name="Tram stops", show=True)
    for stop in tram_stops.itertuples():
        folium.Marker(
            location=[stop.geometry.y, stop.geometry.x],
            icon=folium.DivIcon(
                class_name="station-marker-anchor",
                html=f'<div class="station-symbol station-symbol-t">T</div>',
                icon_size=(TRAM_STOP_ICON_SIZE, TRAM_STOP_ICON_SIZE),
                icon_anchor=(TRAM_STOP_ICON_SIZE // 2, TRAM_STOP_ICON_SIZE // 2),
            ),
            tooltip=f"Tram: {stop.name}",
        ).add_to(tram_stop_layer)
    tram_stop_layer.add_to(map_object)

    station_layer = folium.FeatureGroup(name="S-Bahn and U-Bahn stations", show=True)
    for station in stations.itertuples():
        symbol = station_symbol(station.kind)
        symbol_class = "station-symbol-s" if symbol == "S" else "station-symbol-u"
        folium.Marker(
            location=[station.geometry.y, station.geometry.x],
            icon=folium.DivIcon(
                class_name="station-marker-anchor",
                html=f'<div class="station-symbol {symbol_class}">{symbol}</div>',
                icon_size=(STATION_ICON_SIZE, STATION_ICON_SIZE),
                icon_anchor=(STATION_ICON_SIZE // 2, STATION_ICON_SIZE // 2),
            ),
            tooltip=f"{station.kind}: {station.name}",
        ).add_to(station_layer)
    station_layer.add_to(map_object)

    folium.LayerControl(collapsed=False).add_to(map_object)
    add_map_panel(map_object, len(stations), len(tram_stops))
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
    tram_stops = load_tram_stops()
    if tram_stops.empty:
        raise RuntimeError("No tram stops found.")
    tram_walk_zones = build_walk_zones(graph, tram_stops, [TRAM_WALK_MINUTES])
    render_map(place, stations, walk_zones, tram_stops, tram_walk_zones)


if __name__ == "__main__":
    run_analysis()
