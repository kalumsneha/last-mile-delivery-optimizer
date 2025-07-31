import random
from deap import base, creator, tools
from backend.aco_optimizer import aco_optimize_routes, compute_cost_matrix
from backend.optimizer import evaluate_with_details

def hybrid_optimize_routes(points, base_point, point_map,
                           population_size=30, generations=40,
                           aco_top_k=2, aco_iterations=5,
                           aco_inject_every=5,
                           return_history=False):
    delivery_ids = [p.id for p in points]
    index_map = {i: pid for i, pid in enumerate(delivery_ids)}
    reverse_map = {pid: i for i, pid in index_map.items()}

    # Precompute ACO cost matrix once for reuse
    cost_matrix = compute_cost_matrix(points, base_point, point_map)

    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(len(delivery_ids)), len(delivery_ids))
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    def evaluate_indexed(individual):
        route_ids = [base_point.id] + [index_map[i] for i in individual] + [base_point.id]
        cost, _, metrics = evaluate_with_details(route_ids, point_map)
        individual.metrics = metrics
        return (cost,)

    toolbox.register("evaluate", evaluate_indexed)

    population = toolbox.population(n=population_size)
    routes_over_time = []
    fitness_over_time = []
    metrics_over_time = []

    for gen in range(generations):
        offspring = toolbox.select(population, len(population))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.7:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalid_ind:
            ind.fitness.values = toolbox.evaluate(ind)

        # ✅ Only inject ACO every N generations
        if gen % aco_inject_every == 0:
            offspring.sort(key=lambda ind: ind.fitness.values[0])
            top_k = offspring[:aco_top_k]

            for elite in top_k:
                best_aco_route, _ = aco_optimize_routes(
                    points=points,
                    base_point=base_point,
                    point_map=point_map,
                    num_ants=5,
                    num_iterations=aco_iterations,
                    return_history=False
                )

                new_gene = [reverse_map[pid] for pid in best_aco_route[1:-1] if pid in reverse_map]
                if len(new_gene) == len(elite):
                    elite[:] = new_gene
                    elite.fitness.values = toolbox.evaluate(elite)

        population[:] = offspring

        best_ind = tools.selBest(population, 1)[0]
        best_route = [base_point.id] + [index_map[i] for i in best_ind] + [base_point.id]
        best_score = best_ind.fitness.values[0]
        best_metrics = best_ind.metrics

        fitness_over_time.append(best_score)
        metrics_over_time.append(best_metrics)

        if return_history:
            routes_over_time.append(best_route)

        print(f"[Hybrid] Generation {gen+1}/{generations}, Best Score: {best_score:.2f}")

    return best_route, best_score, routes_over_time if return_history else [], fitness_over_time, metrics_over_time
