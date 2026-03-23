import osmnx as ox
import geopandas as gpd
import pandas as pd
import networkx as nx
import folium
from shapely.geometry import Point, MultiPolygon
from shapely.ops import unary_union
import city2graph.graph as c2g_graph
import torch
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# ── SETTINGS ──────────────────────────────────────────────────────────────────
PLACE          = "Mitte, Berlin, Germany"   # default area
WALK_SPEED_KMH = 4.5                        # average walking speed km/h
WALK_MINUTES   = 5                          # isochrone size in minutes
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(place=PLACE, walk_min=WALK_MINUTES):
    print(f"🚀 Starting analysis for: {place}")
    walk_metres = (WALK_SPEED_KMH * 1000 / 60) * walk_min
    
    # 1. Download walking network
    print(f"📥 Downloading walking network...")
    try:
        G = ox.graph_from_place(place, network_type="walk", simplify=True)
    except Exception as e:
        print(f"❌ Error downloading network: {e}")
        return
        
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    nodes, edges = ox.graph_to_gdfs(G)
    
    # 2. Download transit stops
    print(f"📥 Downloading transit stops...")
    transit_tags = {
        "station": ["subway", "light_rail"],
        "railway": ["tram_stop", "station"],
        "public_transport": "stop_position"
    }
    try:
        transit_gdf = ox.features_from_place(place, tags=transit_tags)
    except Exception as e:
        print(f"⚠️ No transit stops found or error: {e}")
        transit_gdf = gpd.GeoDataFrame(columns=['geometry'], crs="EPSG:4326")

    # Clean transit stops
    transit_stops = transit_gdf[transit_gdf.geometry.geom_type == "Point"].copy()
    transit_polys = transit_gdf[transit_gdf.geometry.geom_type != "Point"].copy()
    if len(transit_polys) > 0:
        transit_polys['geometry'] = transit_polys.geometry.centroid
        transit_stops = pd.concat([transit_stops, transit_polys])
    
    transit_stops = transit_stops.reset_index(drop=True)
    transit_stops = transit_stops.to_crs(epsg=4326)
    print(f"✅ Found {len(transit_stops)} transit stops")

    # 3. Compute isochrones
    print(f"⏱️ Computing {walk_min}-minute isochrones...")
    trip_time_seconds = walk_min * 60
    isochrone_polys = []
    
    for _, stop in transit_stops.iterrows():
        try:
            nearest_node = ox.distance.nearest_nodes(G, stop.geometry.x, stop.geometry.y)
            subgraph = nx.ego_graph(G, nearest_node, radius=trip_time_seconds, distance="travel_time")
            node_points = [Point(data["x"], data["y"]) for _, data in subgraph.nodes(data=True)]
            if len(node_points) >= 3:
                poly = gpd.GeoSeries(node_points).unary_union.convex_hull
                isochrone_polys.append(poly)
        except:
            continue
            
    if not isochrone_polys:
        print("❌ No isochrones could be computed.")
        return

    walkable_zone = unary_union(isochrone_polys)
    walkable_gdf = gpd.GeoDataFrame(geometry=[walkable_zone], crs="EPSG:4326")

    # 4. Download buildings
    print(f"📥 Downloading building footprints...")
    try:
        buildings = ox.features_from_place(place, tags={"building": True})
        buildings = buildings[buildings.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()
        buildings = buildings.to_crs(epsg=4326).reset_index(drop=True)
    except Exception as e:
        print(f"❌ Error downloading buildings: {e}")
        return
    
    print(f"✅ Found {len(buildings)} buildings")

    # 5. Classify buildings
    print("🏷️ Classifying buildings...")
    buildings["within_walk"] = buildings.geometry.centroid.within(walkable_zone)
    n_within = buildings["within_walk"].sum()
    print(f"✅ {n_within} buildings are within walking distance.")

    # 6. Generate city2graph graph
    print("📊 Generating city2graph graph file...")
    # Add some features for the graph
    buildings['node_type'] = 'building'
    # city2graph expects certain columns or properties if we want a rich graph, 
    # but at its simplest, we can convert the GDF.
    try:
        # Convert buildings GDF to PyTorch Geometric Data object
        # We'll include the 'within_walk' as a feature (converted to float)
        graph_gdf = buildings.copy()
        graph_gdf['within_walk_feature'] = graph_gdf['within_walk'].astype(float)
        
        # city2graph.graph.gdf_to_pyg typically takes a GeoDataFrame
        pyg_data = c2g_graph.gdf_to_pyg(graph_gdf)
        
        # Save the graph
        graph_path = "berlin_walkability_graph.pt"
        torch.save(pyg_data, graph_path)
        print(f"✅ Graph saved to: {graph_path}")
    except Exception as e:
        print(f"⚠️ Error creating city2graph file: {e}")

    # 7. Generate Folium map
    print("🗺️ Generating Folium map...")
    centre = nodes.geometry.unary_union.centroid
    m = folium.Map(location=[centre.y, centre.x], zoom_start=14, tiles="CartoDB dark_matter")

    # Walkable zone
    folium.GeoJson(
        walkable_gdf.__geo_interface__,
        name="5-min walkable zone",
        style_function=lambda _: {"fillColor": "#4a90d9", "color": "#4a90d9", "weight": 1, "fillOpacity": 0.15}
    ).add_to(m)

    # Buildings
    near_buildings = buildings[buildings["within_walk"]]
    far_buildings = buildings[~buildings["within_walk"]]

    folium.GeoJson(
        far_buildings.__geo_interface__,
        name="Buildings > 5min walk",
        style_function=lambda _: {"fillColor": "#555555", "color": "#333333", "weight": 0.3, "fillOpacity": 0.6}
    ).add_to(m)

    folium.GeoJson(
        near_buildings.__geo_interface__,
        name="Buildings ≤ 5min walk",
        style_function=lambda _: {"fillColor": "#7ecf6e", "color": "#4aa83a", "weight": 0.5, "fillOpacity": 0.8}
    ).add_to(m)

    # Transit stops
    for _, stop in transit_stops.iterrows():
        name = stop.get('name', 'Transit Stop')
        folium.CircleMarker(
            location=[stop.geometry.y, stop.geometry.x],
            radius=4, color="#ff4444", fill=True, fill_color="#ff4444", fill_opacity=0.9,
            tooltip=name
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    title_html = f'''
    <div style="position:fixed; top:12px; left:60px; z-index:1000;
         background:rgba(0,0,0,0.8); color:white; padding:10px 16px;
         border-radius:8px; font-family:sans-serif; font-size:14px; border: 1px solid #444;">
      <b style="font-size:16px;">🚇 Berlin Transit Walkability</b><br>
      <small>{place}</small><br><br>
      <span style="color:#7ecf6e">■</span> Within 5 min walk &nbsp;
      <span style="color:#888">■</span> Further away<br>
      <span style="color:#ff4444">●</span> S/U/Tram Stop
    </div>'''
    m.get_root().html.add_child(folium.Element(title_html))

    map_path = "berlin_walkability.html"
    m.save(map_path)
    print(f"✅ Folium map saved to: {map_path}")

if __name__ == "__main__":
    run_analysis()
