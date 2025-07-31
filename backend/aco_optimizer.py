import random
from collections import defaultdict
from backend.optimizer import evaluate_with_details


def aco_optimize_routes(points, base_point, point_map,
                        num_ants=5, num_iterations=20,
                        alpha=1.0, beta=2.0, evaporation_rate=0.5,
                        return_history=False):
    cost_matrix = compute_cost_matrix(points, base_point, point_map)

    delivery_ids = [p.id for p in points]
    all_ids = [base_point.id] + delivery_ids + [base_point.id]

    pheromones = defaultdict(lambda: 1.0)
    fitness_over_time = []
    metrics_over_time = []
    history = []
    pheromone_history = []

    def evaluate_route(route):
        cost, _, _ = evaluate_with_details(route, point_map)
        return cost

    best_route = None
    best_cost = float("inf")

    for _ in range(num_iterations):
        all_routes = []
        all_costs = []

        for _ in range(num_ants):
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

                    probabilities.append((pher**alpha) * (heuristic**beta))

                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]
                next_node = random.choices(choices, weights=probabilities)[0]
                route.append(next_node)
                unvisited.remove(next_node)

            route.append(base_point.id)
            cost = evaluate_route(route)
            all_routes.append(route)
            all_costs.append(cost)

            if cost < best_cost:
                best_cost = cost
                best_route = route

        # Pheromone evaporation
        for i in all_ids:
            for j in all_ids:
                if i != j:
                    pheromones[(i, j)] *= (1 - evaporation_rate)

        # Pheromone reinforcement
        for route, cost in zip(all_routes, all_costs):
            for i in range(len(route) - 1):
                pheromones[(route[i], route[i + 1])] += 1.0 / (cost + 1e-6)

        if return_history:
            history.append(best_route)
            fitness_over_time.append(best_cost)
            snapshot = {f"{i}->{j}": pheromones[(i, j)] for i in all_ids for j in all_ids if i != j}
            pheromone_history.append(snapshot)

            _, _, metrics = evaluate_with_details(best_route, point_map)
            metrics_over_time.append(metrics)

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
            if i == j:
                matrix[i][j] = float('inf')
            else:
                matrix[i][j], _, _ = evaluate_with_details([i, j], point_map)

    return matrix
