from typing import Any, Dict, List, Tuple

import numpy as np
from progress.bar import ChargingBar
from sympy.core.symbol import Symbol

from datastructures.fbinarytree import BT
from genetic_operators import get_next_population, new_population
from individual import Individual


def evosym(
    data: Tuple[np.ndarray, np.ndarray],
    population_size: int,
    generations: int,
    vars: List[Symbol],
    operators: Dict[str, int],
    *,
    data_prime: Tuple | None = None,
    crossover_yields_twot_children: bool = False,
    crossover_method: str = "normal",
    meancrossover_rate: float = 0.5,
    mutation_rate: float = 0.1,
    delete_mutation_rate: float = 0.0,
    divine_mutation_rate: float = 0.1,
    mutation_method: str = "nodal",
    fitness_method: str = "mde",
    constants_range: List[float | int] | Tuple[float | int, float | int] = [-1, 1],
    selection_method: str = "tournament",
    size_penalty: str = "linear",
) -> Tuple[
    Individual, float, float, List[float], List[float], List[float], Dict[str, Any]
]:
    population = new_population(
        operators,
        vars,
        3,
        population_size,
        data,
        method=fitness_method,
        size_penalty=size_penalty,
        constants_range=constants_range,
        data_prime=data_prime,
    )

    initial_fitnesses = [ind.fitness for ind in population]
    best_ind = Individual(BT(), float("inf"))
    best_fitness = float("inf")
    worst_fitness = float("-inf")
    generations_mean_fitness: List[float] = [sum(initial_fitnesses) / population_size]
    generations_best_fitness: List[float] = [min(initial_fitnesses)]
    generations_worst_fitness: List[float] = [max(initial_fitnesses)]
    geenrations_best_size: List[int] = [min(ind.tree.size() for ind in population)]
    geenrations_mean_size: List[float] = [
        sum(ind.tree.size() for ind in population) / population_size
    ]

    bar = ChargingBar("Run progress:", max=generations)
    bar.next()

    for _ in range(1, generations):
        population = get_next_population(
            population,
            operators,
            vars,
            data,
            fitness_method=fitness_method,
            size_penalty=size_penalty,
            crossover_method=crossover_method,
            yield_two_children=crossover_yields_twot_children,
            meancrossover_rate=meancrossover_rate,
            mutation_rate=mutation_rate,
            divine_mutation_rate=divine_mutation_rate,
            mutation_method=mutation_method,
            delete_mutation_rate=delete_mutation_rate,
            constants_range=constants_range,
            selection_method=selection_method,
            data_prime=data_prime,
        )

        best_gen_ind = min(population, key=lambda i: i.fitness)
        worst_gen_ind = max(population, key=lambda i: i.fitness)
        if best_gen_ind.fitness < best_fitness:
            best_fitness = best_gen_ind.fitness
            best_ind = best_gen_ind
        if worst_gen_ind.fitness > worst_fitness:
            worst_fitness = worst_gen_ind.fitness

        generations_best_fitness.append(best_gen_ind.fitness)
        generations_worst_fitness.append(worst_gen_ind.fitness)
        generations_mean_fitness.append(
            sum(ind.fitness for ind in population) / population_size
        )
        geenrations_best_size.append(best_gen_ind.tree.size())
        geenrations_mean_size.append(
            sum(ind.tree.size() for ind in population) / population_size
        )

        bar.next()

    bar.finish()

    extra_data = {
        "gens_best_size": geenrations_best_size,
        "gens_mean_size": geenrations_mean_size,
    }

    return (
        best_ind,
        best_fitness,
        worst_fitness,
        generations_best_fitness,
        generations_worst_fitness,
        generations_mean_fitness,
        extra_data,
    )
