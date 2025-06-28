from typing import Callable, Dict, List, Tuple

import numpy as np
import sympy as sp

from datastructures import BT
from symbolic_expression import expr_from_tree

MAX_FITNESS = 10_000.0


def numerify_expression(expr: sp.Expr, variables: List[sp.Symbol]) -> Callable:
    """
    Given a symbolic expression, converts it into a lambda-generated function for numpy.
    """
    func = sp.lambdify(variables, expr, "numpy")
    if isinstance(
        func(np.array([0.0 for _ in range(len(variables))])),
        (int, float, np.integer, np.floating),
    ):
        return lambda x: np.full_like(x, func(0.0))

    return func


def numerify_tree(
    tree: BT, variables: List[sp.Symbol], operators: Dict[str, int]
) -> Callable:
    """
    Given a binary tree, converts it into a lambda-generated function for numpy.
    """
    expr: sp.Expr = expr_from_tree(tree, operators)
    return numerify_expression(expr, variables)


def fitness(
    tree: BT,
    data: Tuple[np.ndarray, np.ndarray],
    variables: List[sp.Symbol],
    operators: Dict[str, int],
    method: str = "mde",
    size_penalty: str = "linear",
    *,
    data_prime: Tuple | None = None,
    alpha: float = 1.0,
) -> float:
    """
    Given a binary tree, calculates the fitness of the tree with respect to the data.
    """
    func = numerify_tree(tree, variables, operators)
    x, y = data
    y_pred = func(x)
    res: float = 0.0
    match method:
        case "ade":
            res = np.sum(np.abs(y - y_pred))
        case "mde":
            res = np.mean(np.abs(y - y_pred))
        case "ase":
            res = np.sum((y - y_pred) ** 2)
        case "mse":
            res = np.mean((y - y_pred) ** 2)
        case _:
            raise ValueError(f"Unknown fitness evaluation method: '{method}'")

    if data_prime is not None:
        deriv = numerify_expression(
            sp.diff(expr_from_tree(tree, operators), *variables), variables
        )
        x, y_prime = data_prime
        y_prime_predict = deriv(x)
        deriv_res: float = 0.0
        match method:
            case "ade":
                deriv_res = np.sum(np.abs(y_prime - y_prime_predict))
            case "mde":
                deriv_res = np.mean(np.abs(y_prime - y_prime_predict))
            case "ase":
                deriv_res = np.sum((y_prime - y_prime_predict) ** 2)
            case "mse":
                deriv_res = np.mean((y_prime - y_prime_predict) ** 2)
            case _:
                raise ValueError(f"Unknown fitness evaluation method: '{method}'")

        res = alpha * res + (1 - alpha) * deriv_res

    tree_size: int = tree.size()
    sample_size: int = x.size
    penalty = 0.0
    match size_penalty:
        case "linear":
            penalty = tree_size / sample_size
        case "cuad":
            penalty = tree_size * tree_size / sample_size
        case "sqrt":
            penalty = np.sqrt(tree_size) / sample_size
        case "log":
            penalty = np.log2(tree_size) / sample_size
        case "none":
            pass
        case _:
            raise ValueError(
                "Avalible size penalties are 'linear', 'cuad', 'sqrt' and 'log'"
            )

    return res * (1 + penalty)


def numerify_population(
    population: List[BT], variables: List[sp.Symbol], operators: Dict[str, int]
) -> List[Callable]:
    """
    Given a population of binary trees, converts them into a list of lambda-generated functions for numpy.
    """
    return [numerify_tree(tree, variables, operators) for tree in population]
