import pandas as pd
from pathlib import Path

def run():
    stops = pd.read_csv("Fahrplan_VBB-2021/stops.txt", dtype=str)
    
    # Names of U5 stations
    u5_names = [
        "Hauptbahnhof", "Bundestag", "Brandenburger Tor", "Museumsinsel", 
        "Unter den Linden", "Rotes Rathaus", "Alexanderplatz", "Schillingstr", 
        "Strausberger Platz", "Weberwiese", "Frankfurter Tor", "Samariterstr", 
        "Frankfurter Allee", "Magdalenenstr", "Lichtenberg", "Friedrichsfelde", 
        "Tierpark", "Biesdorf", "Elsterwerdaer", "Wuhletal", "Kaulsdorf", 
        "Kienberg", "Cottbusser", "Hellersdorf", "Louis-Lewin", "Hönow"
    ]
    
    # We want to find the exact stop_names in the GTFS that represent these U-Bahn stations
    # Usually "U Weberwiese (Berlin)"
    matches = stops[stops['stop_name'].str.contains('|'.join(u5_names), case=False, na=False)]
    
    with open("cache/u5_stops.txt", "w", encoding="utf-8") as f:
        for name in matches['stop_name'].unique():
            f.write(name + "\n")
            
if __name__ == "__main__":
    run()
