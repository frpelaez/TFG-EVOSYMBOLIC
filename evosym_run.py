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
    alpha: float = 1.0,
    comp_depth: int = 2,
    crossover_yields_two_children: bool = False,
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
    """
    Executes the genetic algorithm to fit the given data using the modifications
    indicated by keyword parameters.

    Args:
        data (Tuple[np.ndarray, np.ndarray]): data to fit
        population_size (int): size of the population (remains constant)
        generations (int): number of generations
        vars (List[Symbol]): list of symbolic variables to be used
        operators (Dict[str, int]): dictionary with keys representing the operators and functions to be used and values being the arity of the keys
        data_prime (Tuple | None, optional): derivative-related data. Must match the shape of 'data'. Defaults to None.
        alpha (float, optional): represents 1 - the weight of the derivative in the fitness function. Defaults to 1.0.
        comp_depth (int, optional): controls the composition depth of the first generation. Defaults to 2.
        crossover_yields_two_children (bool, optional): Defaults to False.
        crossover_method (str, optional): choose crossover method (currently supports 'normal' and 'sizefair'). Defaults to 'normal'.
        meancrossover_rate (float, optional): propapility that a mean crossover happens. Defaults to 0.5.
        mutation_rate (float, optional): mutation probability. Defaults to 0.1.
        delete_mutation_rate (float, optional): 'deleting' variant of mutation probability. Defaults to 0.0.
        divine_mutation_rate (float, optional): 'divine' variant of mutation probability. Defaults to 0.1.
        mutation_method (str, optional): choose mutation method (currently supports 'nodal', 'complete' and 'shrinking'). Defaults to "nodal".
        fitness_method (str, optional): fitness function to be used (currently supports 'ade', 'mde', 'ase' and 'mse'). Defaults to "mde".
        constants_range (List[float  |  int] | Tuple[float  |  int, float  |  int], optional): range for constants. Defaults to [-1, 1].
        selection_method (str, optional): choose the selection method (currently supports 'elitism', 'roulette' and 'tournament'). Defaults to "tournament".
        size_penalty (str, optional): penalty functions to be applied (currently supports 'linear', 'cuad', 'log', 'sqrt' and 'none'). Defaults to "linear".

    Returns:
        Tuple[ Individual, float, float, List[float], List[float], List[float], Dict[str, Any] ]: first return value is the actual best solution found. Followed by its fitness score and other metrics
    """
    population = new_population(
        operators,
        vars,
        4,
        population_size,
        data,
        comp_depth=comp_depth,
        method=fitness_method,
        size_penalty=size_penalty,
        constants_range=constants_range,
        data_prime=data_prime,
    )

    initial_fitnesses = [ind.fitness for ind in population]
    best_gen = 0
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

    for i in range(1, generations):
        population = get_next_population(
            population,
            operators,
            vars,
            data,
            fitness_method=fitness_method,
            size_penalty=size_penalty,
            crossover_method=crossover_method,
            yield_two_children=crossover_yields_two_children,
            meancrossover_rate=meancrossover_rate,
            mutation_rate=mutation_rate,
            divine_mutation_rate=divine_mutation_rate,
            mutation_method=mutation_method,
            delete_mutation_rate=delete_mutation_rate,
            constants_range=constants_range,
            selection_method=selection_method,
            data_prime=data_prime,
            alpha=alpha,
        )

        best_gen_ind = min(population, key=lambda i: i.fitness)
        worst_gen_ind = max(population, key=lambda i: i.fitness)
        if best_gen_ind.fitness < best_fitness:
            best_fitness = best_gen_ind.fitness
            best_ind = best_gen_ind
            best_gen = i
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
        "best_gen": best_gen,
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
