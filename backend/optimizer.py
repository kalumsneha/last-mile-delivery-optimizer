import random
import requests
from datetime import datetime

from deap import base, creator, tools, algorithms

import geopandas as gpd
from shapely.geometry import Point

# Load congestion zones once
try:
    congestion_gdf = gpd.read_file("backend/congestion_zones.geojson")  # Adjust path as needed
except Exception as e:
    print(f"⚠️ Failed to load congestion zones: {e}")
    congestion_gdf = None

def get_travel_time(origin, dest):
    base_url = "http://localhost:5000"
    coords = f"{origin['lon']},{origin['lat']};{dest['lon']},{dest['lat']}"
    url = f"{base_url}/route/v1/driving/{coords}?overview=false"

    try:
        response = requests.get(url, timeout=3)
        data = response.json()

        if data.get("code") == "Ok":
            duration_sec = data["routes"][0]["duration"]
            distance_km = data["routes"][0]["distance"] / 1000  # meters → km
            return duration_sec, distance_km
        else:
            print(f"⚠️ OSRM error: {data.get('message', 'Unknown error')}")
            raise Exception("OSRM failed")
    except Exception as e:
        print(f"⚠️ OSRM fallback: {e}")
        lat1, lon1 = origin["lat"], origin["lon"]
        lat2, lon2 = dest["lat"], dest["lon"]
        dist = ((lat1 - lat2)**2 + (lon1 - lon2)**2)**0.5
        speed_m_per_s = 15
        time_sec = (dist * 1000) / speed_m_per_s
        return time_sec, dist

def get_traffic_penalty(origin, dest, hour=None):
    if hour is None:
        hour = datetime.now().hour

    penalty = 0
    if 8 <= hour <= 10 or 17 <= hour <= 19:
        penalty += random.uniform(120, 600)
    elif 11 <= hour <= 16:
        penalty += random.uniform(60, 180)
    else:
        penalty += random.uniform(10, 60)

    if congestion_gdf is not None:
        mid_lat = (origin["lat"] + dest["lat"]) / 2
        mid_lon = (origin["lon"] + dest["lon"]) / 2
        mid_point = Point(mid_lon, mid_lat)
        for _, row in congestion_gdf.iterrows():
            if row.geometry.contains(mid_point):
                multiplier = row.get("multiplier", 1.0)
                penalty *= multiplier
                break

    return penalty

def get_weather_condition(lat, lon):
    return random.choice(["clear", "rain", "fog", "storm", "snow"])

def get_weather_delay_multiplier(weather):
    return {
        "clear": 1.0,
        "rain": 1.2,
        "fog": 1.3,
        "storm": 1.5,
        "snow": 1.4
    }.get(weather, 1.0)

def get_time_window_penalty(arrival_hour, time_window):
    if not time_window:
        return 0
    start, end = time_window
    if arrival_hour < start:
        return (start - arrival_hour) * 60
    elif arrival_hour > end:
        return (arrival_hour - end) * 120
    return 0

def evaluate_with_details(
    individual,
    point_map,
    alpha=1.0, beta=1.0, gamma=1.0, start_hour=8,
    w_time=1.0, w_distance=1.0, w_cost=1.0
):
    total_time_sec = 0
    total_distance = 0
    arrival_time = start_hour * 3600
    breakdown = []

    for i in range(len(individual) - 1):
        origin = point_map[individual[i]]
        dest = point_map[individual[i + 1]]

        base_time, dist_km = get_travel_time(origin, dest)
        total_distance += dist_km

        traffic_penalty = get_traffic_penalty(origin, dest)
        weather = get_weather_condition(origin["lat"], origin["lon"])
        weather_multiplier = get_weather_delay_multiplier(weather)
        weather_penalty = base_time * (weather_multiplier - 1)
        arrival_hour = start_hour + (arrival_time // 3600)
        time_window_penalty = get_time_window_penalty(arrival_hour, dest.get("time_window"))

        segment_time = base_time + alpha * traffic_penalty + beta * weather_penalty + gamma * time_window_penalty
        arrival_time += segment_time
        total_time_sec += segment_time

        cost_dollars = segment_time / 60

        breakdown.append({
            "from": individual[i],
            "to": individual[i + 1],
            "base_time_sec": base_time,
            "traffic_penalty_sec": traffic_penalty,
            "weather_penalty_sec": weather_penalty,
            "time_window_penalty_sec": time_window_penalty,
            "total_cost_sec": segment_time,
            "weather": weather,
            "distance_km": round(dist_km, 2),
            "cost_dollars": round(cost_dollars, 2)
        })

    metrics = {
        "total_time": round(total_time_sec),
        "total_cost": round(total_time_sec / 60, 2),
        "total_distance_km": round(total_distance, 2)
    }

    fitness_score = (
        w_time * total_time_sec +
        w_distance * total_distance +
        w_cost * (total_time_sec / 60)
    )

    return fitness_score, breakdown, metrics

def evaluate(individual, point_map):
    return (evaluate_with_details(individual, point_map)[0],)

def optimize_routes(points, base_point):
    # Ensure both points and base_point are dicts
    if hasattr(base_point, "__dict__"):
        base_point = vars(base_point)
    points = [vars(p) if hasattr(p, "__dict__") else p for p in points]

    all_points = [base_point] + points
    point_ids = [p["id"] for p in all_points]
    point_map = {p["id"]: p for p in all_points}

    delivery_ids = [p["id"] for p in points]
    if not delivery_ids:
        raise ValueError("No delivery points provided.")

    index_map = {i: pid for i, pid in enumerate(delivery_ids)}

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(delivery_ids)), len(delivery_ids))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate_indexed(individual):
        try:
            route_ids = [base_point["id"]] + [index_map[i] for i in individual] + [base_point["id"]]
            cost, _, metrics = evaluate_with_details(
                route_ids, point_map,
                w_time=1.0, w_distance=1.0, w_cost=1.0
            )
            individual.metrics = metrics
            return (cost,)
        except Exception as e:
            print(f"⚠️ Evaluation error: {e} → {individual}")
            return (float("inf"),)

    toolbox.register("evaluate", evaluate_indexed)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=30)
    hof = tools.HallOfFame(1)

    best_routes_over_time = []
    fitness_scores = []
    metrics_over_time = []

    def record_best_route(population, generation):
        hof.update(population)
        best_ind = hof[0]
        best_route = [base_point["id"]] + [index_map[i] for i in best_ind] + [base_point["id"]]
        best_routes_over_time.append(best_route)
        fitness_scores.append(best_ind.fitness.values[0])
        metrics_over_time.append(best_ind.metrics)

    NGEN = 30
    for gen in range(NGEN):
        offspring = algorithms.varAnd(pop, toolbox, cxpb=0.7, mutpb=0.2)
        fits = list(toolbox.map(toolbox.evaluate, offspring))

        valid_fits = [fit for fit in fits if fit is not None and fit[0] != float("inf")]
        if not valid_fits:
            raise ValueError("GA failed: all individuals had invalid fitness.")

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        pop = toolbox.select(offspring, k=len(pop))
        record_best_route(pop, gen)

    best_route = best_routes_over_time[-1]
    _, breakdown, _ = evaluate_with_details(best_route, point_map)
    route_coordinates = [
        {"lat": point_map[pid]["lat"], "lon": point_map[pid]["lon"]}
        for pid in best_route
    ]

    return {
        "best_route": best_route,
        "route_coordinates": route_coordinates,
        "route_breakdown": breakdown,
        "routes_over_time": best_routes_over_time,
        "fitness_over_time": fitness_scores,
        "metrics_over_time": metrics_over_time
    }

