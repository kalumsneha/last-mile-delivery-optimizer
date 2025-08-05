import random
import json
import os
from collections import defaultdict
from datetime import datetime
from backend.optimizer import evaluate_with_details

def aco_optimize_routes(points, base_point, point_map,
                        num_ants=5, num_iterations=30,
                        alpha=1.0, beta=2.0, evaporation_rate=0.5,
                        pheromone_min=0.01, pheromone_max=5.0,
                        return_history=False,
                        log_dir="benchmark_results/logs"):

    cost_matrix = compute_cost_matrix(points, base_point, point_map)
    delivery_ids = [p.id for p in points]
    all_ids = [base_point.id] + delivery_ids + [base_point.id]
    pheromones = defaultdict(lambda: 1.0)

    best_route = None
    best_cost = float("inf")
    fitness_over_time = []
    metrics_over_time = []
    pheromone_history = []
    history = []
    ant_logs_all_iters = []

    for iteration in range(num_iterations):
        all_routes = []
        all_costs = []
        ant_logs = []

        for ant_idx in range(num_ants):
            route = [base_point.id]
            unvisited = set(delivery_ids)

            while unvisited:
                current = route[-1]
                choices = list(unvisited)
                probabilities = []

                for c in choices:
                    pher = pheromones[(current, c)]
                    cost = cost_matrix[current][c]
                    heuristic = 1.0 / (cost + 1e-6)
                    probabilities.append((pher ** alpha) * (heuristic ** beta))

                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]
                next_node = random.choices(choices, weights=probabilities)[0]
                route.append(next_node)
                unvisited.remove(next_node)

            route.append(base_point.id)
            cost, breakdown, _ = evaluate_with_details(route, point_map)

            all_routes.append(route)
            all_costs.append(cost)

            ant_logs.append({
                "iteration": iteration,
                "ant_id": ant_idx,
                "route": route,
                "breakdown": breakdown,
                "total_cost": cost
            })

            if cost < best_cost:
                best_cost = cost
                best_route = route

        # Adaptive evaporation based on improvement
        decay = evaporation_rate
        if iteration > 0 and fitness_over_time and best_cost >= fitness_over_time[-1]:
            decay = min(evaporation_rate * 1.1, 0.95)  # increase evaporation
        else:
            decay = max(evaporation_rate * 0.9, 0.2)   # reward improvement

        for i in all_ids:
            for j in all_ids:
                if i != j:
                    pheromones[(i, j)] *= (1 - decay)
                    pheromones[(i, j)] = max(pheromone_min, pheromones[(i, j)])

        # Reinforce best paths
        for route, cost in zip(all_routes, all_costs):
            for i in range(len(route) - 1):
                pheromones[(route[i], route[i + 1])] += 1.0 / (cost + 1e-6)
                pheromones[(route[i], route[i + 1])] = min(
                    pheromone_max, pheromones[(route[i], route[i + 1])]
                )

        if return_history:
            history.append(best_route)
            fitness_over_time.append(best_cost)
            snapshot = {f"{i}->{j}": pheromones[(i, j)] for i in all_ids for j in all_ids if i != j}
            pheromone_history.append(snapshot)
            _, _, metrics = evaluate_with_details(best_route, point_map)
            metrics_over_time.append(metrics)

        ant_logs_all_iters.extend(ant_logs)

    # Save logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"aco_ant_logs_{timestamp}.json")
    with open(log_path, "w") as f:
        json.dump(ant_logs_all_iters, f, indent=2)

    if return_history:
        return best_route, best_cost, history, fitness_over_time, pheromone_history, metrics_over_time
    else:
        return best_route, best_cost


def compute_cost_matrix(points, base_point, point_map):
    ids = [base_point.id] + [p.id for p in points]
    matrix = {}
    for i in ids:
        matrix[i] = {}
        for j in ids:
            matrix[i][j] = float('inf') if i == j else evaluate_with_details([i, j], point_map)[0]
    return matrix
