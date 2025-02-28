import os
import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict, Tuple
from pprint import pprint
from progress.bar import Bar

from sympy.core.symbol import Symbol
from datastructures.fbinarytree import BT
from genetic_operatons import *
from numerify import *

def evosym_run(data: Tuple[np.ndarray, np.ndarray], population_size: int, generations: int,
               vars: List[Symbol], operators: Dict[str, int],
               *,
               crossover_rate: float = 0.75, mutation_rate: float = 0.1, constants_range = [-1, 1],
               selection_method: str = "tournament", tournament_size: int = 2,
               mutation_method: str = "nodal", fitness_method:str = "mde", size_penalty: str = "linear") -> Tuple[BT, float, BT, float, 
                                                                                                                  int,
                                                                                                                  List[float], List[float]]:
    
    population: List[BT] = generate_population(operators, vars, 5, population_size)
    best_tree = BT()
    best_fitness = float('inf')
    best_gen_tree = BT()
    best_gen_fitness = float('inf')
    best_gen: int = 0
    generations_mean_fitness: List[float] = []
    generations_best_fitness: List[float] = []
    
    bar = Bar("Run progress:", max=generations)
    for i in range(generations):
        fitnesses: List[float] = evaluate_population(population, data, vars, operators,
                                                     method=fitness_method,
                                                     size_penalty=size_penalty)
        
        generations_mean_fitness.append(sum(fitnesses) / population_size)
        generations_best_fitness.append(min(fitnesses))
        population = evolve_population(population, fitnesses, operators, vars,
                                       crossover_rate=crossover_rate,
                                       selection_method=selection_method,
                                       tournament_size=tournament_size,
                                       mutation_rate=mutation_rate,
                                       constants_range=constants_range,
                                       mutation_method=mutation_method)
        
        best_gen_tree: BT = selection(population, fitnesses, method="elitism")
        best_gen_fitness: float = fitness(best_gen_tree, data, vars, operators, method="mde")
        if best_gen_fitness <= best_fitness:
            best_tree: BT = best_gen_tree
            best_fitness: float = best_gen_fitness 
            best_gen = i
            
        bar.next()
    
    bar.finish()
    return best_tree, best_fitness, best_gen_tree, best_gen_fitness, best_gen, generations_mean_fitness, generations_best_fitness


def Vevosym_run(data: Tuple[np.ndarray, np.ndarray], population_size: int, generations: int,
                vars: List[Symbol], operators: Dict[str, int],
                *,
                crossover_rate: float = 0.75, mutation_rate: float = 0.1, constants_range = [-1, 1],
                selection_method: str = "tournament", tournament_size: int = 2,
                mutation_method: str = "nodal", fitness_method: str = "mde", size_penalty: str = "linear") -> Tuple[List[BT], np.floating,
                                                                                                                    List[BT], np.floating, 
                                                                                                                    int,
                                                                                                                    List[np.floating], List[np.floating]]:

    dim = data[1].shape[1]
    population: List[List[BT]] = Vgenerate_population(dim, operators, vars, 5, population_size)
    best_tree = [BT() for _ in range(len(vars))]
    best_fitness = float('inf') # type: ignore
    best_gen_tree = [BT() for _ in range(len(vars))]
    best_gen_fitness = float('inf') # type: ignore
    best_gen: int = 0
    generations_mean_fitness: List[np.floating] = []
    generations_best_fitness: List[np.floating] = []
    
    bar = Bar("Run progress:", max=generations)
    for i in range(generations):
        fitnesses: List[np.floating] = Vevaluate_population(population, data, vars, operators, # type: ignore
                                                            method=fitness_method,
                                                            size_penalty=size_penalty)
        
        generations_mean_fitness.append(sum(fitnesses) / population_size) # type: ignore
        generations_best_fitness.append(min(fitnesses))
        population = Vevolve_population(population, fitnesses, operators, vars,
                                        crossover_rate=crossover_rate,
                                        selection_method=selection_method,
                                        tournament_size=tournament_size,
                                        mutation_rate=mutation_rate,
                                        constants_range=constants_range,
                                        mutation_method=mutation_method)
        
        best_gen_tree: List[BT] = Vselection(population, fitnesses, method="elitism")
        best_gen_fitness: np.floating = Vfitness2(best_gen_tree, data, vars, operators, method="mde") # type: ignore
        if best_gen_fitness <= best_fitness:
            best_tree = best_gen_tree
            best_fitness = best_gen_fitness 
            best_gen = i
            
        bar.next()
    
    bar.finish()
    return best_tree, best_fitness, best_gen_tree, best_gen_fitness, best_gen, generations_mean_fitness, generations_best_fitness # type: ignore

def main() -> None:
    
    x = np.linspace(-2.0, 2.0, 100)
    y = 2 * np.sin(2*x) - 0.5*x + 1 + np.random.normal(0, 0.2, 100)
    data = (x, y)
    
    vars: List[Symbol] = [sp.Symbol('x')]
    operators: Dict[str, int] = {'+': 2, '-': 2, '*': 2}
    constants_range = [-3, 3]
    
    population_size = 30
    generations = 100
    crosspver_rate = 0.5
    mutation_rate = 0.02
    
    mutation_method = "nodal"
    slct_method = "weighted-tournament"
    trnmt_size = 4
    size_pnlty = "logarithmic"
    fitness_method = "mde"
    
    best_tree, best_fitness, best_lastgen_tree, best_last_fitness, \
    best_gen, mean_fitnesses, best_fitnesses= evosym_run(data, population_size, generations, vars, operators,
                                                                                        crossover_rate=crosspver_rate,
                                                                                        mutation_rate=mutation_rate,
                                                                                        mutation_method=mutation_method,
                                                                                        constants_range=constants_range,
                                                                                        selection_method=slct_method,
                                                                                        tournament_size=trnmt_size,
                                                                                        fitness_method=fitness_method,
                                                                                        size_penalty=size_pnlty)
    
    best_pst = best_tree.post_order()
    best_lastgen_pst = best_lastgen_tree.post_order()
    
    parameters = {"pop_size":population_size, "gen_number":generations, "crossover_rate":crosspver_rate, "mutation_rate":mutation_rate,
                                  "sel_method":slct_method, "tournament_size":trnmt_size, "mutation_method":mutation_method, "fitness_method":fitness_method,
                                  "size_penalty":size_pnlty}
    
    print("\n---------------------------------------", end="\n\n")
    
    print("Run parameters:")
    pprint(parameters, sort_dicts=False)
    print("\nRun symbolic variables:")
    pprint(vars)
    print("\nRun operators:")
    pprint(operators, sort_dicts=False)
    
    print("\nTarget expression:")
    pprint("2sin(2x) - 0.5x + 1 + N(0, 0.2)")
    
    best_expr = expr_from_postfix(best_pst, operators)
    simpl_best = best_expr.simplify() if not isinstance(best_expr, float | int) else best_expr
    print("\nExpression from best tree:", simpl_best)
    print("Depth of best tree:", best_tree.depth())
    print("Number of nodes of best tree:", best_tree.size())
    print("Found in generation", best_gen)
    print("Best fitness:", best_fitness, end="\n\n")
    
    last_best_expr = expr_from_postfix(best_lastgen_pst, operators)
    simpl_last_best = last_best_expr.simplify() if not isinstance(last_best_expr, float | int) else last_best_expr
    print("Expression from best last generation tree:", simpl_last_best)
    print("Depth of last best tree:", best_lastgen_tree.depth())
    print("Number of nodes of last best tree:", best_lastgen_tree.size())
    print("Best last generation fitness:", best_last_fitness)
    
    print("\n---------------------------------------")
    
    figure, axes = plt.subplots(3)
    axes[0].title.set_text("Target data")
    axes[0].plot(x, y, label="Target expression")
    best_eval = numerify_tree(best_tree, vars, operators)(x)
    best_last_eval = numerify_tree(best_lastgen_tree, vars, operators)(x)
    evaluation =  best_eval.reshape(-1) if not isinstance(best_eval, float | int) else [best_eval] * len(x)
    last_evaluation = best_last_eval.reshape(-1) if not isinstance(best_last_eval, float | int) else [best_last_eval] * len(x)
    axes[1].title.set_text("Best fitting curve")
    axes[1].plot(x, evaluation, label="Best obtained", color="red")
    axes[2].title.set_text("Best fitting curve from last generation")
    axes[2].plot(x, last_evaluation, label="Best last gen obtained", color="green")
    plt.tight_layout()
    
    stats, st_axes = plt.subplots(2)
    st_axes[0].title.set_text("Mean generation fitness")
    st_axes[0].plot([gen for gen in range(generations)], mean_fitnesses, label="Mean fitness")
    st_axes[1].title.set_text("Best generation fitness")
    st_axes[1].plot([gen for gen in range(generations)], best_fitnesses, label="Best fitness", color="red")
    plt.tight_layout()
    plt.show()
   
   
if __name__ == "__main__":
    main()
