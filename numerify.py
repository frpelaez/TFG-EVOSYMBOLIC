import sympy as sp
import numpy as np

from typing import List, Dict, Tuple, Callable
from itertools import starmap

from datastructures import BT
from symbolic_expression import expr_from_tree
from integrator import rk4

MAX_FITNESS = 10_000.0

def numerify_expression(expr: sp.Expr, variables: List[sp.Symbol]) -> Callable:
    """
    Given a symbolic expression, converts it into a lambda-generated function for numpy.
    """
    func = sp.lambdify(variables, expr, 'numpy')
    if isinstance(func(np.array([0.0 for _ in range(len(variables))])), (int, float, np.integer, np.floating)):
        return lambda x: np.full_like(x, func(0.0))
    
    return func


def numerify_tree(tree: BT, variables: List[sp.Symbol], operators: Dict[str, int]) -> Callable:
    """
    Given a binary tree, converts it into a lambda-generated function for numpy.
    """
    expr: sp.Expr = expr_from_tree(tree, operators)
    return numerify_expression(expr, variables)


def fitness(tree: BT, data: Tuple[np.ndarray, np.ndarray],
            variables: List[sp.Symbol], operators: Dict[str, int],
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
            
        case "cuad":
            res += tree_size * tree_size / sample_size
            
        case "sqrt":
            res += np.sqrt(tree_size) / sample_size
            
        case "log":
            res += np.log2(tree_size) / sample_size
            
    return res


def Vfitness(trees: List[BT], data: Tuple[np.ndarray, np.ndarray],
             variables: List[sp.Symbol], operators: Dict[str, int],
             *,
             method: str = "mde", size_penalty: str = "linear") -> np.floating | float:
    """
    Vectorized version of the fitness evaluation function. Interprets the trees as homogeneous differential equations.
    Evaluates the fitness depending on how similar their solutions are to the given data.

    Args:
        trees (List[BT]):
            list of the binary trees that represent the differential equations
            
        data (Tuple[np.ndarray, np.ndarray]):
            data to compare individials with
            
        variables (List[sp.Symbol]):
            list of variables
            
        operators (Dict[str, int]):
            operators dictionary
            
        method (str, optional):
            fitness-evaluation method. Defaults to "mde".
            
        size_penalty (str, optional):
            size penalty scaling. Defaults to "linear".

    Returns:
        np.floating
    """
    components = list(starmap(numerify_tree, [(tree, variables, operators) for tree in trees]))
    #func = lambda x : np.array([f(*x) for f in components])
    func = lambda t : np.array([f(t) for f in components]).transpose()
    ts, x = data
    # x_pred = rk4(lambda t, x : func(x), (ts[0], ts[-1]), ts.size, x[0])
    x_pred = func(ts)
    res = 0.0
    
    match method:
        case "ade":
            res =  np.sum(np.abs(x - x_pred))
    
        case "mde":
            res = np.mean(np.abs(x - x_pred))
        
        case "ase":
            res = np.sum((x - x_pred) ** 2)
        
        case "mse":
            res = np.mean((x - x_pred) ** 2)
            
        case _:
            raise Exception()
    
    if res == np.inf or res == np.nan:
        return MAX_FITNESS
    
    tree_sizes = [tree.size() for tree in trees]
    n_trees = len(trees)
    sample_size = ts.size
    
    match size_penalty:
        case "none":
            res += 0.0
        
        case "linear":
            res += np.mean(tree_sizes) / (sample_size)
            
        case "cuad":
            res += np.mean(tree_sizes) ** 2 / (sample_size)
    
    return res # type: ignore


def Vfitness2(individual: List[BT], data: Tuple[np.ndarray, np.ndarray],
              variables: List[sp.Symbol], operators: Dict[str, int],
              *,
              method: str = "mde", size_penalty: str = "linear") -> np.floating | float:
    """
    Vectorized version of the fitness evaluation function. Interprets the trees as homogeneous differential equations.
    Evaluates the fitness depending on how similar their solutions are to the given data.

    Args:
        trees (List[BT]):
            list of the binary trees that represent the differential equations
            
        data (Tuple[np.ndarray, np.ndarray]):
            data to compare individials with
            
        variables (List[sp.Symbol]):
            list of variables
            
        operators (Dict[str, int]):
            operators dictionary
            
        method (str, optional):
            fitness-evaluation method. Defaults to "mde".
            
        size_penalty (str, optional):
            size penalty scaling. Defaults to "linear".

    Returns:
        np.floating
    """
    t, x = data
    x_rows, x_cols = x.shape
    if x_cols != len(individual):
        x = x.transpose()
    splited_data = [x[:, i] for i in range(x.shape[1])]
    components_fitnesses = np.array(list(starmap(fitness,
                                                               [(tree, (t, col), variables, operators, method, size_penalty)
                                                                for tree, col in zip(individual, splited_data)])))
    
    return components_fitnesses.sum()


def numerify_population(population: List[BT],
                        variables: List[sp.Symbol], operators: Dict[str, int]) -> List[Callable]:
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


def Vevaluate_population(population: List[List[BT]], data: Tuple[np.ndarray, np.ndarray],
                         variables: List[sp.Symbol], operators: Dict[str, int],
                         *,
                         method: str = "mde", size_penalty: str = "lienar") -> List[np.floating | float]:
    """
    Given a population of binary trees, evaluates each tree with respect to the data.
    """
    return [Vfitness2(tree, data, variables, operators, method=method, size_penalty=size_penalty) for tree in population]


def main() -> None:
    
    f = lambda t, x : np.array([-x[1], x[0]])
    N = 1_000
    t_span = (-0.77, 0.77)
    t = np.linspace(t_span[0], t_span[1], N)
    x0 = [1., 0.]
    data = t, rk4(f, t_span, N, x0)
    
    x, y = sp.symbols("x, y")
    vars = [x, y]
    ops = {"+": 2, "-": 2, "*": 2}
    
    t1 = BT("-", BT(y), BT("+", BT(y), BT(y)))
    t2 = BT(x)
    
    sys = [t1, t2]
    
    print(Vfitness(sys, data, vars, ops))

if __name__ == "__main__":
    main()