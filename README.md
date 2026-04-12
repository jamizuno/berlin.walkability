# Berlin Rapid Transit Walkability

Interactive Folium map showing the area within a 5-minute walk of Berlin
S-Bahn and U-Bahn stations. Tram stops are intentionally excluded.

## Features

- 5-minute walking reachability area for S-Bahn and U-Bahn stations across Berlin.
- No Geoapify credits or API key required.
- Uses OpenStreetMap data through OSMnx.
- Generates a clean Leaflet/Folium map at `index.html`.

## Project Structure

- `generate_berlin_transit_map.py`: Main analysis and map generation script.
- `index.html`: Generated interactive Folium map.
- `berlin_walkability_graph.pt`: Earlier graph output kept in the repository.

## Usage

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the script:

   ```bash
   python generate_berlin_transit_map.py
   ```

3. Open `index.html` in a browser.

## Notes

The current script does not call Geoapify. It computes the walking area locally
from OpenStreetMap data, then styles the result in a Geoapify-like way.
