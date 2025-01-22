import sympy as sp
import numpy as np

from typing import List, Dict, Tuple, Callable

from datastructures import BT
from symbolic_expression import expr_from_tree


def numerify_expression(expr: sp.Expr, variables: List[sp.Symbol]) -> Callable:
    """
    Given a symbolic expression, converts it into a lambda-generated function for numpy.
    """
    func = sp.lambdify(variables, expr, 'numpy')
    return func


def numerify_tree(tree: BT, variables: List[sp.Symbol], operators: Dict[str, int]) -> Callable:
    """
    Given a binary tree, converts it into a lambda-generated function for numpy.
    """
    expr: sp.Expr = expr_from_tree(tree, operators)
    return numerify_expression(expr, variables)


def fitness(tree: BT, data: Tuple[np.ndarray, np.ndarray], variables: List[sp.Symbol], operators: Dict[str, int],
            *,
            method: str = "mde") -> float:
    """
    Given a binary tree, calculates the fitness of the tree with respect to the data.
    """
    func = numerify_tree(tree, variables, operators)
    x, y = data
    y_pred = func(x)
    
    if method == "ade":
        return np.sum(np.abs(y - y_pred))
    
    if method == "mde":
        return np.mean(np.abs(y - y_pred))
    
    if method == "ase":
        return np.sum((y - y_pred) ** 2)
    
    if method == "mse":
        return np.mean((y - y_pred) ** 2)
    
    raise ValueError("Avalible methods for fitness function are 'ade' and 'mse'")


def numerify_population(population: List[BT], variables: List[sp.Symbol], operators: Dict[str, int]) -> List[Callable]:
    """
    Given a population of binary trees, converts them into a list of lambda-generated functions for numpy.
    """
    return [numerify_tree(tree, variables, operators) for tree in population]


def evaluate_population(population: List[BT], data: Tuple[np.ndarray, np.ndarray],
                        variables: List[sp.Symbol], operators: Dict[str, int],
                        *,
                        method: str = "ade") -> List[float]:
    """
    Given a population of binary trees, evaluates each tree with respect to the data.
    """
    return [fitness(tree, data, variables, operators, method=method) for tree in population]