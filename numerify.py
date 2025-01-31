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
    if isinstance(func, float):
        return lambda x: np.full_like(x, func)
    
    return func


def numerify_tree(tree: BT, variables: List[sp.Symbol], operators: Dict[str, int]) -> Callable:
    """
    Given a binary tree, converts it into a lambda-generated function for numpy.
    """
    expr: sp.Expr = expr_from_tree(tree, operators)
    return numerify_expression(expr, variables)


def fitness(tree: BT, data: Tuple[np.ndarray, np.ndarray], variables: List[sp.Symbol], operators: Dict[str, int],
            *,
            method: str = "mde", size_penalty: str = "linear") -> float:
    """
    Given a binary tree, calculates the fitness of the tree with respect to the data.
    """
    func = numerify_tree(tree, variables, operators)
    x, y = data
    y_pred = func(x)
    res: float = 0.0
    
    match method:
        case "ade":
            res =  np.sum(np.abs(y - y_pred))
    
        case "mde":
            res = np.mean(np.abs(y - y_pred))
        
        case "ase":
            res = np.sum((y - y_pred) ** 2)
        
        case "mse":
            res = np.mean((y - y_pred) ** 2)
            
        case _:
            raise Exception()
    
    tree_size: int = tree.size()
    sample_size: int = x.size
    match size_penalty:
        case "lienar":
            res += tree_size / sample_size
            
        case "square":
            res += tree_size * tree_size / sample_size
            
    return res


def numerify_population(population: List[BT], variables: List[sp.Symbol], operators: Dict[str, int]) -> List[Callable]:
    """
    Given a population of binary trees, converts them into a list of lambda-generated functions for numpy.
    """
    return [numerify_tree(tree, variables, operators) for tree in population]


def evaluate_population(population: List[BT], data: Tuple[np.ndarray, np.ndarray],
                        variables: List[sp.Symbol], operators: Dict[str, int],
                        *,
                        method: str = "mde", size_penalty: str = "lienar") -> List[float]:
    """
    Given a population of binary trees, evaluates each tree with respect to the data.
    """
    return [fitness(tree, data, variables, operators, method=method, size_penalty=size_penalty) for tree in population]