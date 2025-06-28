from typing import Any, Dict, List, Tuple

from progress.bar import ChargingBar
from sympy.core.symbol import Symbol

from datastructures import BT
from genetic_operators import new_tree
from individual import Individual
from numerify import fitness


def random_search(
    data: Tuple[Any, Any],
    vars: List[Symbol],
    operators: Dict[str, int],
    iterations: int,
    terminal_rate: float,
    *,
    fitness_method: str = "mse",
    derivative_data: Tuple | None = None,
    alpha: float = 0.0,
    size_penalty: str = "none",
) -> Tuple[Individual, float, List[float], int]:
    best_ind = Individual(BT(), float("inf"))
    best_fitness = float("inf")
    fitnesses = [0.0] * iterations
    best_iter = 0

    bar = ChargingBar("Run progress:", max=iterations)
    for i in range(iterations):
        tree = new_tree(operators, vars, 6, terminal_rate=terminal_rate)
        fit = fitness(
            tree,
            data,
            vars,
            operators,
            method=fitness_method,
            size_penalty=size_penalty,
            data_prime=derivative_data,
            alpha=alpha,
        )
        fitnesses[i] = fit
        if fit < best_fitness:
            best_fitness = fit
            best_ind = Individual(tree, best_fitness)
            best_iter = i
        bar.next()

    return best_ind, best_fitness, fitnesses, best_iter
