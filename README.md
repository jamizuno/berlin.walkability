# Berlin Transit Walkability Analysis

Interactive Map and Graph analysis of transit walkability in Berlin (Mitte).

## Features
- **5-minute walking distance** isochrones for S-Bahn, U-Bahn, and Tram stations.
- **8,661 buildings** analyzed in Berlin Mitte.
- **Interactive Folium map** with color-coded building proximity.
- **City2Graph** compatible graph representation (PyTorch Geometric).

## Project Structure
- `generate_berlin_transit_map.py`: The main analysis script.
- `index.html`: Interactive Folium map (open in browser).
- `berlin_walkability_graph.pt`: Graph representation for GeoAI analysis.

## Usage
1. Install dependencies:
   ```bash
   pip install osmnx folium torch torch-geometric city2graph
   ```
2. Run the script:
   ```bash
   python generate_berlin_transit_map.py
   ```

## Results
In Berlin Mitte, approximately 99.8% of buildings are within a 5-minute walk of a transit stop.
