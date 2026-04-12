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
WALK_LEVELS    = [5, 10, 15, 20]            # isochrone thresholds in minutes
VINTAGE_COLORS = {
    5:  "#fff1b5", # Light Yellow
    10: "#f7b267", # Orange
    15: "#f27059", # Red-Orange
    20: "#9e2a2b", # Crimson
    "outer": "#335c67" # Deep Blue-Green (for buildings outside 20m)
}
# ──────────────────────────────────────────────────────────────────────────────

def run_analysis(place=PLACE, walk_levels=WALK_LEVELS):
    print(f"🚀 Starting Vienna 1912 Style Analysis for: {place}")
    
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

    transit_stops = transit_gdf[transit_gdf.geometry.geom_type == "Point"].copy()
    transit_polys = transit_gdf[transit_gdf.geometry.geom_type != "Point"].copy()
    if len(transit_polys) > 0:
        transit_polys['geometry'] = transit_polys.geometry.centroid
        transit_stops = pd.concat([transit_stops, transit_polys])
    
    transit_stops = transit_stops.reset_index(drop=True)
    transit_stops = transit_stops.to_crs(epsg=4326)
    print(f"✅ Found {len(transit_stops)} transit stops")

    # 3. Compute Isochrones for each level
    isochrone_zones = {}
    for level in sorted(walk_levels):
        print(f"⏱️ Computing {level}-minute isochrones...")
        trip_time_seconds = level * 60
        polys = []
        for _, stop in transit_stops.iterrows():
            try:
                nearest_node = ox.distance.nearest_nodes(G, stop.geometry.x, stop.geometry.y)
                subgraph = nx.ego_graph(G, nearest_node, radius=trip_time_seconds, distance="travel_time")
                node_points = [Point(data["x"], data["y"]) for _, data in subgraph.nodes(data=True)]
                if len(node_points) >= 3:
                    polys.append(gpd.GeoSeries(node_points).unary_union.convex_hull)
            except:
                continue
        if polys:
            isochrone_zones[level] = unary_union(polys)

    if not isochrone_zones:
        print("❌ No isochrones could be computed.")
        return

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

    # 5. Classify buildings into levels
    print("🏷️ Classifying buildings into time bands...")
    buildings["walk_time"] = 999  # Default to "far"
    for level in sorted(walk_levels, reverse=True):
        zone = isochrone_zones[level]
        # Assign level if centroid is within zone (smaller levels will overwrite larger ones due to order)
        buildings.loc[buildings.geometry.centroid.within(zone), "walk_time"] = level

    # 6. Generate city2graph graph
    print("📊 Generating city2graph graph file...")
    try:
        graph_gdf = buildings.copy()
        graph_gdf['walk_time_feature'] = graph_gdf['walk_time'].astype(float)
        pyg_data = c2g_graph.gdf_to_pyg(graph_gdf)
        torch.save(pyg_data, "berlin_walkability_graph.pt")
        print(f"✅ Graph saved to: berlin_walkability_graph.pt")
    except Exception as e:
        print(f"⚠️ Error creating city2graph file: {e}")

    # 7. Generate Folium map
    print("🗺️ Generating Folium map (Vienna 1912 Style)...")
    centre = nodes.geometry.unary_union.centroid
    # Use CartoDB Positron for a cleaner, vintage-friendly look
    m = folium.Map(location=[centre.y, centre.x], zoom_start=14, tiles="CartoDB positron")

    # Add Isochrone Bands (Rings)
    # We add larger zones first, then smaller ones on top (or subtract them)
    # Actually, adding them with high opacity and smaller ones on top works well.
    for level in sorted(walk_levels, reverse=True):
        zone = isochrone_zones[level]
        color = VINTAGE_COLORS[level]
        folium.GeoJson(
            gpd.GeoDataFrame(geometry=[zone], crs="EPSG:4326").__geo_interface__,
            name=f"{level} min walk",
            style_function=lambda _, c=color: {
                "fillColor": c, "color": c, "weight": 1, "fillOpacity": 0.4
            }
        ).add_to(m)

    # Buildings
    def building_style(feature):
        time = feature['properties']['walk_time']
        color = VINTAGE_COLORS.get(time, VINTAGE_COLORS["outer"])
        return {
            "fillColor": color,
            "color": "#333333",
            "weight": 0.3,
            "fillOpacity": 0.7 if time <= 20 else 0.4
        }

    folium.GeoJson(
        buildings[['geometry', 'walk_time']].__geo_interface__,
        name="Buildings",
        style_function=building_style
    ).add_to(m)

    # Transit stops
    for _, stop in transit_stops.iterrows():
        name = stop.get('name', 'Transit Stop')
        folium.CircleMarker(
            location=[stop.geometry.y, stop.geometry.x],
            radius=3, color="#000000", fill=True, fill_color="#000000", fill_opacity=1.0,
            tooltip=name, weight=1
        ).add_to(m)

    folium.LayerControl().add_to(m)
    
    # Stylized Legend
    legend_html = f'''
    <div style="position:fixed; bottom:30px; left:30px; z-index:1000;
         background:rgba(255,255,255,0.9); padding:15px; border-radius:10px;
         font-family:'Palatino Linotype', 'Book Antiqua', Palatino, serif; 
         font-size:14px; border:2px solid #8b4513; box-shadow: 2px 2px 10px rgba(0,0,0,0.2);">
      <b style="font-size:18px; color:#5d4037;">Vienna 1912 Style</b><br>
      <small style="color:#8d6e63;">Berlin Transit Travel Times</small><br><hr style="border:0.5px solid #d7ccc8;">
      <div style="margin-bottom:5px;"><span style="background:{VINTAGE_COLORS[5]}; width:20px; height:15px; display:inline-block; border:1px solid #444; margin-right:8px;"></span> 0-5 Minutes</div>
      <div style="margin-bottom:5px;"><span style="background:{VINTAGE_COLORS[10]}; width:20px; height:15px; display:inline-block; border:1px solid #444; margin-right:8px;"></span> 5-10 Minutes</div>
      <div style="margin-bottom:5px;"><span style="background:{VINTAGE_COLORS[15]}; width:20px; height:15px; display:inline-block; border:1px solid #444; margin-right:8px;"></span> 10-15 Minutes</div>
      <div style="margin-bottom:5px;"><span style="background:{VINTAGE_COLORS[20]}; width:20px; height:15px; display:inline-block; border:1px solid #444; margin-right:8px;"></span> 15-20 Minutes</div>
      <div style="margin-bottom:5px;"><span style="background:{VINTAGE_COLORS['outer']}; width:20px; height:15px; display:inline-block; border:1px solid #444; margin-right:8px;"></span> Over 20 Minutes</div>
      <hr style="border:0.5px solid #d7ccc8;">
      <span style="color:#000;">●</span> Transit Stop (S/U/Tram)
    </div>'''
    m.get_root().html.add_child(folium.Element(legend_html))

    m.save("index.html")
    print(f"✅ Folium map saved to: index.html")

if __name__ == "__main__":
    run_analysis()
