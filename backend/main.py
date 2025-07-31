from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Tuple

from backend.optimizer import optimize_routes, evaluate_with_details
from backend.aco_optimizer import aco_optimize_routes
from backend.hybrid_optimizer import hybrid_optimize_routes

app = FastAPI()

class Point(BaseModel):
    id: int
    lat: float
    lon: float
    label: str
    time_window: Optional[Tuple[int, int]] = None

class OptimizationRequest(BaseModel):
    points: List[Point]
    algorithm: str

@app.post("/optimize")
def optimize(req: OptimizationRequest):
    base_point = Point(id=0, lat=43.651070, lon=-79.347015, label="Depot")
    point_map = {p.id: p.dict() for p in req.points}
    point_map[base_point.id] = base_point.dict()

    # Initialize all outputs
    best_route = []
    breakdown = []
    routes_over_time = []
    fitness_over_time = []
    metrics_over_time = []
    pheromone_history = []

    if req.algorithm == "GA":
        result = optimize_routes(req.points, base_point)
        best_route = result["best_route"]
        breakdown = result["route_breakdown"]
        routes_over_time = result.get("routes_over_time", [])
        fitness_over_time = result.get("fitness_over_time", [])
        metrics_over_time = result.get("metrics_over_time", [])
        print(f"[GA] Generations captured: {len(routes_over_time)}")

    elif req.algorithm == "ACO":
        best_route, total_cost, history, fitness_over_time, pheromone_history, metrics_over_time = aco_optimize_routes(
            req.points, base_point, point_map, return_history=True
        )

        _, breakdown, metrics = evaluate_with_details(best_route, point_map)
        metrics_over_time.append(metrics)
        routes_over_time = history if history else [best_route]
        print(f"[ACO] Generations captured: {len(routes_over_time)}")


    elif req.algorithm == "Hybrid":
        best_route, total_cost, history, fitness_over_time, metrics_over_time = hybrid_optimize_routes(
            req.points, base_point, point_map, return_history=True
        )
        _, breakdown, _ = evaluate_with_details(best_route, point_map)
        routes_over_time = history if history else [best_route]
        print(f"[Hybrid] Generations captured: {len(routes_over_time)}")


    else:
        return {"error": "Unsupported algorithm"}

    total_cost = sum(seg["total_cost_sec"] for seg in breakdown)

    response = {
        "best_route": best_route,
        "route_breakdown": breakdown,
        "routes_over_time": routes_over_time,
        "fitness_over_time": fitness_over_time,
        "metrics_over_time": metrics_over_time,
        "route_coordinates": [
            {"lat": point_map[pid]["lat"], "lon": point_map[pid]["lon"]}
            for pid in best_route
        ],
        "total_cost": total_cost
    }

    if req.algorithm == "ACO":
        response["pheromone_history"] = pheromone_history

    return response
