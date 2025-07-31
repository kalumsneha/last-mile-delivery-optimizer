import os
import requests

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

def get_travel_time(origin, destination):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin['lat']},{origin['lon']}",
        "destination": f"{destination['lat']},{destination['lon']}",
        "departure_time": "now",
        "key": GOOGLE_API_KEY
    }
    res = requests.get(url, params=params).json()
    try:
        return res["routes"][0]["legs"][0]["duration"]["value"]
    except:
        return 999999

def get_traffic_penalty(origin, destination):
    url = "https://maps.googleapis.com/maps/api/directions/json"
    params = {
        "origin": f"{origin['lat']},{origin['lon']}",
        "destination": f"{destination['lat']},{destination['lon']}",
        "departure_time": "now",
        "traffic_model": "best_guess",
        "key": GOOGLE_API_KEY
    }
    res = requests.get(url, params=params).json()
    try:
        normal = res["routes"][0]["legs"][0]["duration"]["value"]
        traffic = res["routes"][0]["legs"][0]["duration_in_traffic"]["value"]
        return max(traffic - normal, 0)
    except:
        return 0
