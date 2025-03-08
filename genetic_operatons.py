from random import random, randint, choice, choices, uniform
from typing import List, Dict, Tuple, Any

import sympy as sp
import numpy as np

from datastructures import BT
from individual import Individual
from symbolic_expression import tree_from_postfix, expr_from_postfix
from numerify import numerify_tree

def generate_tree(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int,
                  *,
                  current_depth: int = 1, constants_range: List[float] = [-1, 1]) -> BT:
    """
    This function generates a random syntactic binary tree

    Args:
        operators (Dict[str, int])
        vars (List[sp.Symbol]): list of variables
        max_depth (int): maximum depth for the generated tree
        constants_range (List[float]): interval for constants, default [-1, 1]

    Returns:
        BT
    """
    if current_depth < max_depth:
        node_type: str = choices(["operator", "terminal"], [0.9, 0.1])[0]
    else:
        node_type = "terminal"
        
    if node_type == "operator":
        node: sp.Symbol | str | float = choice(list(operators.keys()))
        arity: int = operators[node]
        
        if arity == 1:
            left_child: BT = generate_tree(operators, vars, max_depth - 1) 
            return BT(node, left_child)
        
        if arity == 2:
            left_child: BT = generate_tree(operators, vars, max_depth - 1)
            right_child: BT = generate_tree(operators, vars, max_depth - 1)
            return BT(node, left_child, right_child)
        
    if node_type == "terminal":
        if random() < 0.75:
            node: sp.Symbol | str | float = choice(vars)
        else:
            start, end = constants_range
            cte: float = round(uniform(start, end), 4)
            node: sp.Symbol | str | float = cte
        return BT(node)
    
    return BT()


def generate_individual(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int,
                        *,
                        constants_range: List[float] = [-1, 1]) -> Individual:
    
    return Individual(generate_tree(operators, vars, max_depth, constants_range=constants_range), float("inf"))
  

def generate_population(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, population_size: int,
                        *,
                        constants_range: List[float] = [-1, 1]) -> List[BT]:
    """
    This function generates a population of random syntactic binary trees

    Args:
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity
        vars (List[sp.Symbol]): list of variables
        max_depth (int): maximum depth for the generated trees
        population_size (int): number of trees in the population
        constants_range (List[float]): interval for constants, default [-1, 1]

    Returns:
        List[BT]
    """
    return [generate_tree(operators, vars, max_depth, constants_range=constants_range) for _ in range(population_size)]


def Igenerate_population(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, population_size: int,
                         *,
                         constants_range: List[float] = [-1, 1]) -> List[Individual]:
    
    return [generate_individual(operators, vars, max_depth, constants_range=constants_range) for _ in range(population_size)]


def Vgenerate_population(dimension: int, operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, population_size: int,
                         *,
                         constants_range: List[float] = [-1, 1]) -> List[List[BT]]:
    """
    This function generates a population of random syntactic binary trees

    Args:
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity
        vars (List[sp.Symbol]): list of variables
        max_depth (int): maximum depth for the generated trees
        population_size (int): number of trees in the population
        constants_range (List[float]): interval for constants, default [-1, 1]

    Returns:
        List[BT]
    """
    return [[generate_tree(operators, vars, max_depth, constants_range=constants_range) for _ in range(dimension)]
            for _ in range(population_size)]


def IVgenerate_population(dimension: int, operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, population_size: int,
                          *,
                          constants_range: List[float] = [-1, 1]) -> List[List[Individual]]:
    
    return [[generate_individual(operators, vars, max_depth, constants_range=constants_range) for _ in range(dimension)]
            for _ in range(population_size)]


def crossover(tree1: BT, tree2: BT, operators: Dict[str, int]) -> BT:
    """
    This function applies the binary operation of 'crossover' to two syntactic trees and produces a new one

    Args:
        tree1 (BT): tree from which genetic material will be extracted (donor tree)
        tree2 (BT): tree in which genetic material will be inserted (receiver tree)
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        BT: offspring syntactic tree
    """
    pst1: List[Any] = tree1.post_order()
    pst2: List[Any] = tree2.post_order()
    
    cross_point1: int = randint(0, len(pst1) - 1)
    cross_point2: int = randint(0, len(pst2) - 1)
    
    selected_node1 = pst1[cross_point1]
    if isinstance(selected_node1, (sp.Symbol, float)):
        extracted: List[Any] = pst1[cross_point1 : cross_point1 + 1]
    else:
        assert isinstance(selected_node1, str)
        nodes_left: int = operators[selected_node1]
        counter = 0
        while nodes_left > 0 and counter < cross_point1:
            next_node = pst1[cross_point1 - counter - 1]
            if isinstance(next_node, (sp.Symbol, float)):
                nodes_left -= 1

            if isinstance(next_node, str):
                nodes_left += operators[next_node] - 1
            
            counter += 1
        
        extracted = pst1[cross_point1 - counter : cross_point1 + 1]
    
    selected_node2 = pst2[cross_point2]
    if isinstance(selected_node2, (sp.Symbol, float)):
        extractedIndex = cross_point2
    else:
        assert isinstance(selected_node2, str)
        nodes_left: int = operators[selected_node2]
        counter = 0
        while nodes_left > 0 and counter < cross_point2:
            next_node = pst2[cross_point2 - counter - 1]
            if isinstance(next_node, (sp.Symbol, float)):
                nodes_left -= 1
                
            if isinstance(next_node, str):
                nodes_left += operators[next_node] - 1
            
            counter += 1
            
        extractedIndex: int = cross_point2 - counter
        
    offspring_pst: List[Any] = pst2[: extractedIndex] + extracted + pst2[cross_point2 + 1 :]
    
    return tree_from_postfix(offspring_pst, operators)


def Icrossover(ind1: Individual, ind2: Individual, operators: Dict[str, int]) -> Individual:
    
    return Individual(crossover(ind1.tree, ind2.tree, operators), min(ind1.parent_fitness, ind2.parent_fitness))


def mean_crossover(tree1: BT, tree2: BT, operators: Dict[str, int]) -> BT:
    
    if not "+" in operators and not "*" in operators:
        raise ValueError("The mean crossover operation requieres both '+' and '*' operators")
    
    sum_tree = BT("+", tree1, tree2)
    mean_tree = BT("*", BT(0.5), sum_tree)
    
    return mean_tree


def Imean_crossover(ind1: Individual, ind2: Individual, operators: Dict[str, int]) -> Individual:
    
    return Individual(mean_crossover(ind1.tree, ind2.tree, operators), min(ind1.parent_fitness, ind2.parent_fitness))


def mutation(tree: BT, vars: List[sp.Symbol], operators: Dict[str, int],
             *,
             variant: str = "nodal", constants_range: List[float] | Tuple[float, float] = [-1, 1]) -> BT:
    """
    This function applies the unary 'mutation' to a syntactic binary tree and procudes a new one

    Args:
        tree (BT)
        vars (List[sp.Symbol]): list of variables
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity
        variant (str, optional): 'mutation' operation variants. Defaults to "nodal". The other supported variants are "complete" and "shrinking"
        constants_range (int, optional): magnitude bound for the (float) constants that might appear. Defaults to 1.

    Returns:
        BT
    """
    pst = tree.post_order()
    mutation_point: int = randint(0, len(pst) - 1)
    selected_node = pst[mutation_point]
    
    match variant:
        case "nodal":
            if isinstance(selected_node, (sp.Symbol, float)):
                if random() < 0.75:
                    pst[mutation_point] = choice(vars)
                else:
                    start, end = constants_range
                    cte: float = round(uniform(start, end), 4)
                    if isinstance(pst[mutation_point], float):
                        pst[mutation_point] += cte
                    else:
                        pst[mutation_point] = cte
            else:
                assert isinstance(selected_node, str)
                arity: int = operators[selected_node]
                pst: List[Any] = pst[: mutation_point] + \
                                 [choice([valid_op for valid_op in list(operators.keys()) if operators[valid_op] == arity])] + \
                                 pst[mutation_point + 1 :]
                                 
            return tree_from_postfix(pst, operators)
            
        case "complete":
            if isinstance(selected_node, (sp.Symbol, float)):
                extractedIndex = mutation_point
                depth = 1
            else:
                assert isinstance(selected_node, str)
                nodes_left: int = operators[selected_node]
                counter = 0
                while nodes_left > 0 and counter <= mutation_point:
                    next_node = pst[mutation_point - counter - 1]
                    if isinstance(next_node, (sp.Symbol, float)):
                        nodes_left -= 1
                        
                    if isinstance(next_node, str):
                        nodes_left += operators[next_node] - 1
                    
                    counter += 1
                
                extractedIndex: int = mutation_point - counter
                extracted_subtree_pst: List[Any] = pst[extractedIndex : mutation_point + 1]
                depth: int = tree_from_postfix(extracted_subtree_pst, operators).depth()
            
            new_subtree: BT = generate_tree(operators, vars, depth)
            new_subpst: List[Any] = new_subtree.post_order()
            
            pst = pst[: extractedIndex] + new_subpst + pst[mutation_point + 1 :]
            
            return tree_from_postfix(pst, operators)
            
        case "shrinking":
            if isinstance(selected_node, (sp.Symbol, float)):
                extractedIndex = mutation_point
                depth = 1
            else:
                assert isinstance(selected_node, str)
                nodes_left: int = operators[selected_node]
                counter = 0
                while nodes_left > 0 and counter <= mutation_point:
                    next_node = pst[mutation_point - counter - 1]
                    if isinstance(next_node, (sp.Symbol, float)):
                        nodes_left -= 1
                        
                    if isinstance(next_node, str):
                        nodes_left += operators[next_node] - 1
                    
                    counter += 1
                
                extractedIndex: int = mutation_point - counter
                extracted_subtree_pst: List[Any] = pst[extractedIndex : mutation_point + 1]
                depth: int = tree_from_postfix(extracted_subtree_pst, operators).depth()
            
            new_subtree: BT = generate_tree(operators, vars, depth - 1)
            new_subpst: List[Any] = new_subtree.post_order()
            
            pst = pst[: extractedIndex] + new_subpst + pst[mutation_point + 1 :]
            
            return tree_from_postfix(pst, operators)
        
        case _:
            raise Exception("Avalible mutation variants are 'nodal' (by default), 'complete' and 'shrinking'")


def Imutation(ind: Individual, vars: List[sp.Symbol], operators: Dict[str, int],
              *,
              variant: str = "nodal", constants_range: List[float] | Tuple[float, float] = [-1, 1]) -> Individual:
    
    return Individual(mutation(ind.tree, vars, operators, variant=variant, constants_range=constants_range), ind.parent_fitness)


def divine_mutation(tree: BT, vars: List[sp.Symbol], operators: Dict[str, int], data: Tuple[np.ndarray, np.ndarray]) -> BT:
    
    if not "+" in operators:
        raise ValueError("The divine mutation operation requires the '+' operator")
    
    t, x = data
    func = numerify_tree(tree, vars, operators)
    x_pred = func(t)
    difference = x - x_pred
    mean_difference = float(np.mean(difference))
    new_tree = BT("+", tree, BT(mean_difference))
    
    return new_tree


def Idivine_mutation(ind: Individual, vars: List[sp.Symbol], operators: Dict[str, int], data: Tuple[np.ndarray, np.ndarray]) -> Individual:

    return Individual(divine_mutation(ind.tree, vars, operators, data), ind.parent_fitness)


def selection(population: List[BT], fitness: List[float],
              *,
              method: str = "tournament", tournament_size: int = 2) -> BT:
    
    if method == "tournament":
        selected_indices: List[int] = choices(list(range(len(population))), k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "weighted-tournament":
        max_fitness: float = max(fitness)
        min_fitness: float = min(fitness)
        weights: List[float] = [min_fitness + max_fitness - f for f in fitness]
        selected_indices: List[int] = choices(list(range(len(population))), weights, k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "roulette":
        max_fitness: float = max(fitness)
        min_fitness: float = min(fitness)
        weights: List[float] = [min_fitness + max_fitness - f for f in fitness]
        selected_index: int = choices(list(range(len(population))), weights)[0]
        return population[selected_index]
    
    if method == "elitism":
        winner_index: int = min(range(len(population)), key=lambda i: fitness[i])
        return population[winner_index]
    
    raise Exception("Available selection methods are 'tournament' (by default), 'weighted-tournament', 'roulette' and 'elitism'")


def Iselection(population: List[Individual], fitness: List[float],
               *,
               method: str = "tournament", tournament_size: int = 2) -> Individual:
    
    if method == "tournament":
        selected_indices: List[int] = choices(list(range(len(population))), k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "weighted-tournament":
        max_fitness: float = max(fitness)
        min_fitness: float = min(fitness)
        weights: List[float] = [min_fitness + max_fitness - f for f in fitness]
        selected_indices: List[int] = choices(list(range(len(population))), weights, k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "roulette":
        max_fitness: float = max(fitness)
        min_fitness: float = min(fitness)
        weights: List[float] = [min_fitness + max_fitness - f for f in fitness]
        selected_index: int = choices(list(range(len(population))), weights)[0]
        return population[selected_index]
    
    if method == "elitism":
        winner_index: int = min(range(len(population)), key=lambda i: fitness[i])
        return population[winner_index]
    
    raise Exception("Available selection methods are 'tournament' (by default), 'weighted-tournament', 'roulette' and 'elitism'")


def Vselection(population: List[List[BT]], fitness: List[np.floating],
               *,
               method: str = "tournament", tournament_size: int = 2) -> List[BT]:
    
    if method == "tournament":
        selected_indices: List[int] = choices(list(range(len(population))), k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "weighted-tournament":
        max_fitness: np.floating = max(fitness)
        min_fitness: np.floating = min(fitness)
        weights: List[np.floating] = [min_fitness + max_fitness - f for f in fitness]
        selected_indices: List[int] = choices(list(range(len(population))), weights, k=tournament_size) # type: ignore
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "roulette":
        max_fitness: np.floating = max(fitness)
        min_fitness: np.floating = min(fitness)
        probabilities: List[np.floating] = [min_fitness + max_fitness - f for f in fitness]
        selected_index: int = choices(list(range(len(population))), probabilities)[0] # type: ignore
        return population[selected_index]
    
    if method == "elitism":
        winner_index: int = min(range(len(population)), key=lambda i: fitness[i])
        return population[winner_index]
    
    raise Exception("Available selection methods are 'tournament' (by default), 'weighted-tournament', 'roulette' and 'elitism'")


def IVselection(population: List[List[Individual]], fitness: List[np.floating],
               *,
               method: str = "tournament", tournament_size: int = 2) -> List[Individual]:
    
    if method == "tournament":
        selected_indices: List[int] = choices(list(range(len(population))), k=tournament_size)
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "weighted-tournament":
        max_fitness: np.floating = max(fitness)
        min_fitness: np.floating = min(fitness)
        weights: List[np.floating] = [min_fitness + max_fitness - f for f in fitness]
        selected_indices: List[int] = choices(list(range(len(population))), weights, k=tournament_size) # type: ignore
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "roulette":
        max_fitness: np.floating = max(fitness)
        min_fitness: np.floating = min(fitness)
        probabilities: List[np.floating] = [min_fitness + max_fitness - f for f in fitness]
        selected_index: int = choices(list(range(len(population))), probabilities)[0] # type: ignore
        return population[selected_index]
    
    if method == "elitism":
        winner_index: int = min(range(len(population)), key=lambda i: fitness[i])
        return population[winner_index]
    
    raise Exception("Available selection methods are 'tournament' (by default), 'weighted-tournament', 'roulette' and 'elitism'")


def evolve_population(population: List[BT], fitness: List[float],
                      operators: Dict[str, int], vars: List[sp.Symbol],
                      data: Tuple[np.ndarray, np.ndarray],
                      *,
                      crossover_rate: float = 0.75, meancrossover_rate: float = 0.2,
                      mutation_rate: float = 0.1, divine_mutation_rate = 0.1,
                      mutation_method = "nodal", constants_range: List[float] | Tuple[float, float] = [-1, 1],
                      selection_method: str = "tournament", tournament_size: int = 2) -> List[BT]:
    
    new_population: List[BT] = []
    while len(new_population) < len(population):
        parent1: BT = selection(population, fitness,
                                method=selection_method, tournament_size=tournament_size)
        parent2: BT = selection(population, fitness,
                                method=selection_method, tournament_size=tournament_size)
        
        if random() < crossover_rate:
            if random() < meancrossover_rate:
                offspring: BT = mean_crossover(parent1, parent2, operators)
            else:
                offspring: BT = crossover(parent1, parent2, operators)
        else:
            offspring: BT = parent1.copy()
        
        if random() < mutation_rate:
            if random() < divine_mutation_rate:
                offspring: BT = divine_mutation(offspring, vars, operators, data)
            else:
                offspring: BT = mutation(offspring, vars, operators, variant=mutation_method, constants_range=constants_range)
        
        new_population.append(offspring)

    return new_population


def Ievolve_population(population: List[Individual], fitness: List[float],
                      operators: Dict[str, int], vars: List[sp.Symbol],
                      data: Tuple[np.ndarray, np.ndarray],
                      *,
                      crossover_rate: float = 0.75, meancrossover_rate: float = 0.2,
                      mutation_rate: float = 0.1, divine_mutation_rate: float = 0.1,
                      mutation_method = "nodal", constants_range: List[float] | Tuple[float, float] = [-1, 1],
                      selection_method: str = "tournament", tournament_size: int = 2) -> List[Individual]:
    
    new_population: List[Individual] = []
    while len(new_population) < len(population):
        parent1: Individual = Iselection(population, fitness,
                                method=selection_method, tournament_size=tournament_size)
        parent2: Individual = Iselection(population, fitness,
                                method=selection_method, tournament_size=tournament_size)
        
        if random() < crossover_rate:
            if random() < meancrossover_rate:
                offspring = Imean_crossover(parent1, parent2, operators)
            else:
                offspring = Icrossover(parent1, parent2, operators)
        else:
            offspring = parent1.copy()
        
        if random() < mutation_rate:
            if random() < divine_mutation_rate:
                offspring = Idivine_mutation(offspring, vars, operators, data)
            else:
                offspring = Imutation(offspring, vars, operators, variant=mutation_method, constants_range=constants_range)
        
        new_population.append(offspring)

    return new_population


def Vevolve_population(population: List[List[BT]], fitness: List[np.floating],
                       operators: Dict[str, int], vars: List[sp.Symbol],
                       data: Tuple[np.ndarray, np.ndarray] | None = None,
                       *,
                       crossover_rate: float = 0.75, meancrossover_rate: float = 0.2,
                       mutation_rate: float = 0.1, divine_mutation_rate: float = 0.0,
                       constants_range: List[float] | Tuple[float, float] = [-1, 1],
                       selection_method: str = "tournament", tournament_size: int = 2,
                       mutation_method = "nodal") -> List[List[BT]]:
    
    new_population: List[List[BT]] = []
    while len(new_population) < len(population):
        parent1: List[BT] = Vselection(population, fitness,
                                       method=selection_method, tournament_size=tournament_size)
        parent2: List[BT] = Vselection(population, fitness,
                                       method=selection_method, tournament_size=tournament_size)
        offspring: List[BT] = []
        for i in range(len(parent1)):
            if random() < crossover_rate:
                if random() < meancrossover_rate:
                    offspring_component = mean_crossover(parent1[i], parent2[i], operators)
                else:
                    offspring_component = crossover(parent1[i], parent2[i], operators)
            else:
                offspring_component = parent1[i].copy()
                
            if random() < mutation_rate:
                if data and random() < divine_mutation_rate:
                    offspring_component = divine_mutation(offspring_component, vars, operators, data)
                else:
                    offspring_component = mutation(offspring_component, vars, operators, variant=mutation_method, constants_range=constants_range)
                
            offspring.append(offspring_component)
            
        new_population.append(offspring)
        
    return new_population


def IVevolve_population(population: List[List[Individual]], fitness: List[np.floating],
                       operators: Dict[str, int], vars: List[sp.Symbol],
                       data: Tuple[np.ndarray, np.ndarray],
                       *,
                       crossover_rate: float = 0.75, meancrossover_rate: float = 0.2,
                       mutation_rate: float = 0.1, divine_mutation_rate: float = 0.0,
                       constants_range: List[float] | Tuple[float, float] = [-1, 1],
                       selection_method: str = "tournament", tournament_size: int = 2,
                       mutation_method = "nodal") -> List[List[Individual]]:
    
    new_population: List[List[Individual]] = []
    while len(new_population) < len(population):
        parent1: List[Individual] = IVselection(population, fitness,
                                                method=selection_method, tournament_size=tournament_size)
        parent2: List[Individual] = IVselection(population, fitness,
                                                method=selection_method, tournament_size=tournament_size)
        offspring: List[Individual] = []
        for i in range(len(parent1)):
            if random() < crossover_rate:
                if random() < meancrossover_rate:
                    offspring_component = Imean_crossover(parent1[i], parent2[i], operators)
                else:
                    offspring_component = Icrossover(parent1[i], parent2[i], operators)
            else:
                offspring_component = parent1[i].copy()
                
            if random() < mutation_rate:
                if data and random() < divine_mutation_rate:
                    offspring_component = Idivine_mutation(offspring_component, vars, operators, data)
                else:
                    offspring_component = Imutation(offspring_component, vars, operators, variant=mutation_method, constants_range=constants_range)
                
            offspring.append(offspring_component)
            
        new_population.append(offspring)
        
    return new_population


def main() -> None:
    
    ops: Dict[str, int] = {"+":2, 
                           "-": 2,
                           "*": 2,
                           "/": 2,
                           "pow": 2,
                           "exp": 1,
                           "log": 1,
                           "sin": 1,
                           "cos": 1}
    
    x = sp.Symbol('x')
    vlist: List[sp.Symbol] = [x]
    
    print("Random generated tree:")
    t: BT = generate_tree(ops, vlist, 5)
    t.show()
    print("\nAnd now a copy of it")
    t2c: BT = t.copy()
    t2c.show()
    print("\nIts postorder traversal is:")
    t_pst = t.post_order()
    print(t_pst)
    print("\nAnd its corresponding Sympy expression is:")
    print(expr_from_postfix(t_pst, ops))
    
    print("\nA (nodal) mutation of the tree might be:")
    tm: BT = mutation(t, vlist, ops, variant="nodal")
    tm.show()
    print("\nIts postorder traversal is:")
    tm_pst = tm.post_order()
    print(tm_pst)
    print("\nAnd its corresponding Sympy expression is:")
    print(expr_from_postfix(tm_pst, ops))
    
    print("\nA (complete) mutation of the tree might be:")
    tmc: BT = mutation(t, vlist, ops, variant="complete")
    tmc.show()
    print("\nIts postorder traversal is:")
    tmc_pst = tmc.post_order()
    print(tmc_pst)
    print("\nAnd its corresponding Sympy expression is:")
    print(expr_from_postfix(tmc_pst, ops))
    
    print("\nA (shrinking) mutation of the tree might be:")
    tms: BT = mutation(t, vlist, ops, variant="complete")
    tms.show()
    print("\nIts postorder traversal is:")
    tms_pst = tms.post_order()
    print(tms_pst)
    print("\nAnd its corresponding Sympy expression is:")
    print(expr_from_postfix(tms_pst, ops))
    
    print("\nNow we will test the crossover operation:")
    t1: BT = generate_tree(ops, vlist, 4)
    t2: BT = generate_tree(ops, vlist, 4)
    print("Tree 1:")
    t1.show()
    t1_pst = t1.post_order()
    print(expr_from_postfix(t1_pst, ops))
    print("Tree 2:")
    t2.show()
    t2_pst = t2.post_order()
    print(expr_from_postfix(t2_pst, ops))
    offspring: BT = crossover(t1, t2, ops)
    print("Offspring:")
    offspring.show()
    offspring_pst = offspring.post_order()
    print(expr_from_postfix(offspring_pst, ops))
    
if __name__ == "__main__":
    # main()
    t1 = generate_tree({"+":2, "*":2}, [sp.Symbol("x")], 4)
    t2 = generate_tree({"+":2, "*":2}, [sp.Symbol("x")], 4)
    print("Tree 1:")
    t1.show()
    print("Tree 2")
    t2.show()
    
    mean_tree = mean_crossover(t1, t2, {"+":2, "*":2})
    print("Mean tree")
    mean_tree.show()
    
    t3 = generate_tree({"+":2, "*":2}, [sp.Symbol("x")], 4)
    print("Tree 3:")
    t3.show()
    
    data = (np.array([.0, .5, 1., 1.5, 2.]), 2 + np.array([.0, .5, 1., 1.5, 2.]))
    t3m = divine_mutation(t3, [sp.Symbol("x")], {"+":2, "*":2}, data)
    print("Tree after divine mutation:")
    t3m.show()