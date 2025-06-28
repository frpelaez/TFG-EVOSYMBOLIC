from random import choice, choices, randint, random, uniform
from typing import Any, Dict, List, Tuple

import numpy as np
import sympy as sp

from datastructures import BT
from individual import Individual
from numerify import fitness, numerify_tree
from symbolic_expression import tree_from_postfix
from utils import k_largest, random_pairs


def new_tree(
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    depth: int,
    *,
    comp_depth: int = -1,
    force_depth: bool = False,
    force_binary: bool = False,
    terminal_rate: float = 0.3,
    variable_rate: float = 0.75,
    constants_range: List[float | int] | Tuple[float | int, float | int] = [-1, 1],
) -> BT:
    if depth == 0 or comp_depth == 0 or random() < terminal_rate:
        if random() < variable_rate:
            return BT(choice(vars))
        else:
            return BT(round(uniform(*constants_range), 4))
    else:
        if random() < terminal_rate:
            if random() < variable_rate:
                return BT(choice(vars))
            else:
                return BT(round(uniform(*constants_range), 4))
        else:
            if force_binary:
                op = choice(list(filter(lambda s: operators[s] == 2, operators.keys())))
            else:
                op = choice(list(operators.keys()))
            arity = operators[op]
            if arity == 1:
                child = new_tree(
                    operators,
                    vars,
                    depth - 1,
                    comp_depth=comp_depth - 1,
                    force_depth=True,
                    force_binary=True,
                    terminal_rate=terminal_rate,
                )
                return BT(op, child)
            else:
                if force_depth:
                    child1 = new_tree(
                        operators,
                        vars,
                        depth - 1,
                        comp_depth=comp_depth - 1,
                        terminal_rate=terminal_rate,
                    )
                    child2 = new_tree(
                        operators,
                        vars,
                        depth - 1,
                        comp_depth=comp_depth - 1,
                        terminal_rate=terminal_rate,
                    )
                else:
                    child1 = new_tree(
                        operators,
                        vars,
                        depth - 1,
                        comp_depth=comp_depth,
                        terminal_rate=terminal_rate,
                    )
                    child2 = new_tree(
                        operators,
                        vars,
                        depth - 1,
                        comp_depth=comp_depth,
                        terminal_rate=terminal_rate,
                    )
                return BT(op, child1, child2)


def generate_tree(
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    max_depth: int,
    *,
    current_depth: int = 1,
    constants_range: List[float] = [-1.0, 1.0],
) -> BT:
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


def generate_individual(
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    max_depth: int,
    *,
    constants_range: List[float] = [-1, 1],
) -> Individual:
    return Individual(
        generate_tree(operators, vars, max_depth, constants_range=constants_range),
        float("inf"),
    )


def generate_tree_population(
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    max_depth: int,
    population_size: int,
    *,
    constants_range: List[float | int] | Tuple[float | int, float | int] = [-1.0, 1.0],
    comp_depth: int = 2,
) -> List[BT]:
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
    return [
        new_tree(
            operators,
            vars,
            max_depth,
            constants_range=constants_range,
            comp_depth=comp_depth,
        )
        for _ in range(population_size)
    ]


def new_population(
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    max_depth: int,
    population_size: int,
    data: Tuple[np.ndarray, np.ndarray],
    *,
    data_prime: Tuple | None = None,
    comp_depth: int = 2,
    method: str = "mse",
    size_penalty: str = "none",
    constants_range: List[float | int] | Tuple[float | int, float | int] = [-1, 1],
) -> List[Individual]:
    trees = generate_tree_population(
        operators,
        vars,
        max_depth,
        population_size,
        constants_range=constants_range,
        comp_depth=comp_depth,
    )
    return [
        Individual(
            tree,
            fitness(
                tree,
                data,
                vars,
                operators,
                method=method,
                size_penalty=size_penalty,
                data_prime=data_prime,
            ),
        )
        for tree in trees
    ]


def crossover(
    tree1: BT, tree2: BT, operators: Dict[str, int], *, yield_two_children: bool = False
) -> BT | Tuple[BT, BT]:
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
        remanent_left = pst1[:cross_point1]
        remanent_right = pst1[cross_point1 + 1 :]
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
        remanent_left = pst1[: cross_point1 - counter]
        remanent_right = pst1[cross_point1 + 1 :]

    selected_node2 = pst2[cross_point2]
    if isinstance(selected_node2, (sp.Symbol, float)):
        insertion_index = cross_point2
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

        insertion_index: int = cross_point2 - counter

    offspring_pst1: List[Any] = (
        pst2[:insertion_index] + extracted + pst2[cross_point2 + 1 :]
    )

    if yield_two_children:
        offspring_pst2 = (
            remanent_left + pst2[insertion_index : cross_point2 + 1] + remanent_right
        )
        return (
            tree_from_postfix(offspring_pst1, operators),
            tree_from_postfix(offspring_pst2, operators),
        )

    return tree_from_postfix(offspring_pst1, operators)


def sizefair_crossover(
    tree1: BT,
    tree2: BT,
    operators: Dict[str, int],
    *,
    yield_two_children: bool = False,
) -> BT | Tuple[BT, BT]:
    pst1 = tree1.post_order()
    pst2 = tree2.post_order()
    cross_point1 = randint(0, len(pst1) - 1)
    selected_node = pst1[cross_point1]
    if isinstance(selected_node, (sp.Symbol, float)):
        extracted1 = pst1[cross_point1 : cross_point1 + 1]
        remaining1_left = pst1[:cross_point1]
        remaining1_right = pst1[cross_point1 + 1 :]
    else:
        assert isinstance(selected_node, str)
        assert selected_node in operators
        nodes_left: int = operators[selected_node]
        counter = 0
        while nodes_left > 0 and counter < cross_point1:
            next_node = pst1[cross_point1 - counter - 1]
            if isinstance(next_node, (sp.Symbol, float)):
                nodes_left -= 1

            if isinstance(next_node, str):
                nodes_left += operators[next_node] - 1

            counter += 1

        extracted1 = pst1[cross_point1 - counter : cross_point1 + 1]
        remaining1_left = pst1[: cross_point1 - counter]
        remaining1_right = pst1[cross_point1 + 1 :]

    extracted_size = len(extracted1)

    remaining = set(range(len(pst2)))
    selected_index = -1
    counter = 0
    found = False
    while not found and len(remaining) > 0:
        candidate = choice(list(remaining))
        selected_node = pst2[candidate]
        if isinstance(selected_node, (sp.Symbol, float)):
            counter = 1
        else:
            assert isinstance(selected_node, str)
            assert selected_node in operators
            nodes_left: int = operators[selected_node]
            counter = 1
            while nodes_left > 0 and counter <= candidate:
                next_node = pst2[candidate - counter]
                if isinstance(next_node, (sp.Symbol, float)):
                    nodes_left -= 1

                if isinstance(next_node, str):
                    nodes_left += operators[next_node] - 1

                counter += 1

        if abs(extracted_size - counter) <= 1:
            found = True
            selected_index = candidate
            continue

        remaining.remove(candidate)

    if selected_index != -1:
        offspring_pst = (
            pst2[: selected_index - counter + 1]
            + extracted1
            + pst2[selected_index + 1 :]
        )
    else:
        offspring_pst = pst2

    if yield_two_children:
        if selected_index != -1:
            offspring_pst2 = (
                remaining1_left
                + pst2[selected_index - counter + 1 : selected_index + 1]
                + remaining1_right
            )
        else:
            offspring_pst2 = pst1

        return (
            tree_from_postfix(offspring_pst, operators),
            tree_from_postfix(offspring_pst2, operators),
        )

    return tree_from_postfix(offspring_pst, operators)


def mean_crossover(tree1: BT, tree2: BT, operators: Dict[str, int]) -> BT:
    if "+" not in operators and "*" not in operators:
        raise ValueError(
            "The mean crossover operation requieres both '+' and '*' operators"
        )

    sum_tree = BT("+", tree1, tree2)
    mean_tree = BT("*", BT(0.5), sum_tree)

    return mean_tree


def mutation(
    tree: BT,
    vars: List[sp.Symbol],
    operators: Dict[str, int],
    *,
    variant: str = "nodal",
    constants_range: List[float] | Tuple[float, float] = [-1, 1],
    delete_mutation_rate: float = 0.0,
) -> BT:
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
                if (
                    mutation_point != 0
                    and len(pst[: mutation_point + 1]) > 1
                    and random() < delete_mutation_rate
                ):
                    nodes_left = operators[selected_node]
                    counter = 0
                    while nodes_left > 0 and counter <= mutation_point:
                        next_node = pst[mutation_point - counter - 1]
                        if isinstance(next_node, (sp.Symbol, float)):
                            nodes_left -= 1
                        if isinstance(next_node, (str)):
                            nodes_left += operators[next_node] - 1
                        counter += 1
                    if random() < 0.5:
                        node = choice(vars)
                    else:
                        node = round(uniform(*constants_range), 4)
                    pst = (
                        pst[: mutation_point - counter]
                        + [node]
                        + pst[mutation_point + 1 :]
                    )
                else:
                    assert isinstance(selected_node, str)
                    arity: int = operators[selected_node]
                    pst: List[Any] = (
                        pst[:mutation_point]
                        + [
                            choice(
                                [
                                    valid_op
                                    for valid_op in list(operators.keys())
                                    if operators[valid_op] == arity
                                ]
                            )
                        ]
                        + pst[mutation_point + 1 :]
                    )

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
                extracted_subtree_pst: List[Any] = pst[
                    extractedIndex : mutation_point + 1
                ]
                depth: int = tree_from_postfix(extracted_subtree_pst, operators).depth()

            new_subtree: BT = new_tree(operators, vars, depth)
            new_subpst: List[Any] = new_subtree.post_order()

            pst = pst[:extractedIndex] + new_subpst + pst[mutation_point + 1 :]

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
                extracted_subtree_pst: List[Any] = pst[
                    extractedIndex : mutation_point + 1
                ]
                depth: int = tree_from_postfix(extracted_subtree_pst, operators).depth()

            new_subtree: BT = new_tree(operators, vars, depth - 1)
            new_subpst: List[Any] = new_subtree.post_order()

            pst = pst[:extractedIndex] + new_subpst + pst[mutation_point + 1 :]

            return tree_from_postfix(pst, operators)

        case _:
            raise Exception(
                "Avalible mutation variants are 'nodal' (by default), 'complete' and 'shrinking'"
            )


def divine_mutation(
    tree: BT,
    vars: List[sp.Symbol],
    operators: Dict[str, int],
    data: Tuple[np.ndarray, np.ndarray],
) -> BT:
    if "+" not in operators:
        raise ValueError("The divine mutation operation requires the '+' operator")

    t, x = data
    func = numerify_tree(tree, vars, operators)
    x_pred = func(t)
    difference = x - x_pred
    mean_difference = float(np.mean(difference))
    new_tree = BT("+", tree, BT(mean_difference))

    return new_tree


def selection(
    population: List[BT],
    fitness: List[float],
    *,
    method: str = "tournament",
    tournament_size: int = 2,
) -> BT:
    if method == "tournament":
        selected_indices: List[int] = choices(
            list(range(len(population))), k=tournament_size
        )
        winner_index: int = min(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]

    if method == "weighted-tournament":
        max_fitness: float = max(fitness)
        min_fitness: float = min(fitness)
        weights: List[float] = [min_fitness + max_fitness - f for f in fitness]
        selected_indices: List[int] = choices(
            list(range(len(population))), weights, k=tournament_size
        )
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

    raise Exception(
        "Available selection methods are 'tournament' (by default), 'weighted-tournament', 'roulette' and 'elitism'"
    )


def get_next_population(
    population: List[Individual],
    operators: Dict[str, int],
    vars: List[sp.Symbol],
    data: Tuple[np.ndarray, np.ndarray],
    *,
    data_prime: Tuple | None = None,
    alpha: float = 1.0,
    fitness_method: str = "mse",
    size_penalty: str = "none",
    crossover_method: str = "normal",
    yield_two_children: bool = False,
    meancrossover_rate: float = 0.2,
    mutation_rate: float = 0.1,
    delete_mutation_rate: float = 0.0,
    divine_mutation_rate=0.1,
    mutation_method="nodal",
    constants_range: List[float] | Tuple[float, float] = [-1, 1],
    selection_method: str = "tournament",
) -> List[Individual]:
    new_population = []
    for parent1, parent2 in random_pairs(population):
        if crossover_method != "sizefair":
            offspring = (
                crossover(
                    parent1.tree,
                    parent2.tree,
                    operators,
                    yield_two_children=yield_two_children,
                )
                if random() >= meancrossover_rate
                else mean_crossover(parent1.tree, parent2.tree, operators)
            )
        else:
            offspring = (
                sizefair_crossover(
                    parent1.tree,
                    parent2.tree,
                    operators,
                    yield_two_children=yield_two_children,
                )
                if random() >= meancrossover_rate
                else mean_crossover(parent1.tree, parent2.tree, operators)
            )

        if random() < mutation_rate:
            if random() < divine_mutation_rate:
                if isinstance(offspring, tuple):
                    offspring_a = divine_mutation(offspring[0], vars, operators, data)
                    offspring_b = divine_mutation(offspring[1], vars, operators, data)
                else:
                    offspring = divine_mutation(offspring, vars, operators, data)
            else:
                if isinstance(offspring, tuple):
                    offspring_a = mutation(
                        offspring[0],
                        vars,
                        operators,
                        variant=mutation_method,
                        delete_mutation_rate=delete_mutation_rate,
                        constants_range=constants_range,
                    )
                    offspring_b = mutation(
                        offspring[1],
                        vars,
                        operators,
                        variant=mutation_method,
                        delete_mutation_rate=delete_mutation_rate,
                        constants_range=constants_range,
                    )
                else:
                    offspring = mutation(
                        offspring,
                        vars,
                        operators,
                        variant=mutation_method,
                        delete_mutation_rate=delete_mutation_rate,
                        constants_range=constants_range,
                    )

        if isinstance(offspring, tuple):
            offspring_a, offspring_b = offspring
            new_population.extend(
                [
                    Individual(
                        offspring_a,
                        fitness(
                            offspring_a,
                            data,
                            vars,
                            operators,
                            fitness_method,
                            size_penalty,
                            data_prime=data_prime,
                            alpha=alpha,
                        ),
                    ),
                    Individual(
                        offspring_b,
                        fitness(
                            offspring_b,
                            data,
                            vars,
                            operators,
                            fitness_method,
                            size_penalty,
                            data_prime=data_prime,
                            alpha=alpha,
                        ),
                    ),
                ]
            )
        else:
            new_population.append(
                Individual(
                    offspring,
                    fitness(
                        offspring,
                        data,
                        vars,
                        operators,
                        fitness_method,
                        size_penalty,
                        data_prime=data_prime,
                        alpha=alpha,
                    ),
                )
            )

    res: List[Individual] = []
    candidates = (
        population + new_population
        if isinstance(offspring, tuple)  # type: ignore
        else population + new_population + new_population
    )
    match selection_method:
        case "elitism":
            res.extend(k_largest(candidates, len(population), lambda ind: -ind.fitness))

        case "roulette":
            min_f, max_f = (
                min(i.fitness for i in candidates),
                max(i.fitness for i in candidates),
            )
            weights = [min_f + max_f - i.fitness for i in candidates]
            res.extend(choices(candidates, weights, k=len(population)))

        case "tournament":
            res.extend(
                [
                    min(contendents, key=lambda c: c.fitness)
                    for contendents in random_pairs(candidates)
                ]
            )

    return res
