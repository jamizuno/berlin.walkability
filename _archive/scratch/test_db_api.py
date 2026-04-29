import requests
import json

CLIENT_ID = "7e07cbbaff23595617e43ffbd652c136"
API_KEY = "418b04164337f15c67bf73846bffc154"

def get_station_coords(name):
    url = f"https://apis.deutschebahn.com/db-api-marketplace/apis/ris-stations/v1/stop-places/by-name/{name}"
    headers = {
        "DB-Client-Id": CLIENT_ID,
        "DB-Api-Key": API_KEY,
        "accept": "application/vnd.de.db.ris+json"
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for stop in data.get("stopPlaces", []):
            s_name = stop.get("names", {}).get("DE", {}).get("nameLong", "Unknown")
            pos = stop.get("position", {})
            results.append({
                "name": s_name,
                "lat": pos.get("latitude"),
                "lon": pos.get("longitude")
            })
        return results
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    stations = ["Birkenstein", "Hoppegarten", "Potsdam Hbf", "Bernau"]
    for s in stations:
        print(f"Results for {s}:")
        print(json.dumps(get_station_coords(s), indent=2))
        print("-" * 20)
