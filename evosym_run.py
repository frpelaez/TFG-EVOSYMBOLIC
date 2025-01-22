import numpy as np
from typing import List, Dict, Tuple
from datastructures.fbinarytree import BT
from genetic_operatons import *
from numerify import *

def evosym_run(data: Tuple[np.ndarray, np.ndarray], population_size: int, generations: int,
               vars: List[sp.Symbol], operators: Dict[str, int],
               *,
               crossover_rate: float = 0.75, mutation_rate: float = 0.1) -> Tuple[BT, float]:
    
    population: List[BT] = generate_population(operators, vars, 5, population_size)
    best_tree = BT()
    best_fitness = float('inf')
    
    for _ in range(generations):
        fitnesses: List[float] = evaluate_population(population, data, vars, operators, method="mde")
        population = evolve_population(population, fitnesses, operators, vars,
                                       crossover_rate=crossover_rate, mutation_rate=mutation_rate)
        
        best_gen_tree: BT = selection(population, fitnesses, method="elitism")
        best_gen_fitness: float = fitness(best_gen_tree, data, vars, operators, method="mde")
        if best_gen_fitness < best_fitness:
            best_tree: BT = best_gen_tree
            best_fitness: float = best_gen_fitness
    
    return best_tree, best_fitness


def main() -> None:
    
    x = np.linspace(0, 10, 100)
    # y = 2 * x + 3 + np.random.normal(0, 1, 100)
    y = x **2 - x - 1
    data = (x, y)
    
    vars = [sp.Symbol('x')]
    operators = {'+': 2, '-': 2, '*': 2}
    
    population_size = 20
    generations = 100
    
    best_tree, best_fitness = evosym_run(data, population_size, generations, vars, operators)
    
    print("Goal expression: x^2 - x - 1")
    
    print("Best tree: ")
    best_tree.show()
    best_pst = best_tree.post_order()
    print(expr_from_postfix(best_pst, operators))
    print("Best fitness", best_fitness)
   
   
if __name__ == "__main__":
    main() 