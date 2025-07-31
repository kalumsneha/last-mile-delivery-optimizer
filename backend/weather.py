import os
import requests

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather_condition(lat, lon):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY}
    res = requests.get(url, params=params).json()
    try:
        return res["weather"][0]["main"].lower()
    except:
        return "unknown"

def get_weather_delay_multiplier(condition):
    return {
        "rain": 1.2,
        "thunderstorm": 1.5,
        "snow": 1.3,
        "fog": 1.15,
        "clear": 1.0,
        "clouds": 1.0
    }.get(condition, 1.0)
