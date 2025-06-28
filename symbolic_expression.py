import sys
from typing import Callable, Dict, List

import sympy as sp

from datastructures import BT, Stack

sys.setrecursionlimit(10**6)

AVALIBLE_OPS: Dict[str, Callable] = {
    "+": sp.Add,
    "-": sp.Add,
    "*": sp.Mul,
    "/": sp.Mul,
    "pow": sp.Pow,
    "exp": sp.exp,
    "log": sp.log,
    "sin": sp.sin,
    "cos": sp.cos,
}


def tree_from_postfix(
    traversal: List[sp.Symbol | str], operators: Dict[str, int]
) -> BT:
    """
    This function reconstructs a syntactic binary tree from its postorder traversal

    Args:
        traversal (List[sp.Symbol  |  str]): postorder traversal of the syntactic binary tree
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        BT: reconstructed syntactic binary tree
    """
    s = Stack()
    subtrees = []

    if len(traversal) == 1:
        if not isinstance(traversal[0], (sp.Expr, float)):
            raise Exception(
                "Singleton post-fix notations must be of type 'sp.Expr' or 'float'"
            )
        return BT(traversal[0])

    for elm in traversal:
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)

        if isinstance(elm, str):
            if elm not in operators:
                raise Exception(f"Found an unexpected operator -> ({elm})")

            arity: int = operators[elm]
            if arity >= 3:
                raise Exception("Maximum supported arity is currently 2")
            if s.size() < arity:
                raise Exception(
                    f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it"
                )

            if arity == 1:
                arg = s.pop()
                if isinstance(arg, BT):
                    node = BT(elm, arg)
                else:
                    node = BT(elm, BT(arg))
                s.put(node)
                subtrees.append(node)

            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()

                if not isinstance(arg1, BT) and not isinstance(arg2, BT):
                    node = BT(elm, BT(arg1), BT(arg2))
                else:
                    if isinstance(arg1, BT) and not isinstance(arg2, BT):
                        node = BT(elm, arg1, BT(arg2))
                    else:
                        if not isinstance(arg1, BT) and isinstance(arg2, BT):
                            node = BT(elm, BT(arg1), arg2)
                        else:
                            node = BT(elm, arg1, arg2)
                s.put(node)
                subtrees.append(node)

    if len(subtrees) == 0:
        print("\n", traversal)

    return subtrees[-1]


def trees_from_postfix(
    lst: List[sp.Symbol | str], operators: Dict[str, int]
) -> List[BT]:
    """
    This function reconstructs all subtrees of a syntactic binary tree from its postorder traversal

    Args:
        traversal (List[sp.Symbol  |  str]): postorder traversal of the syntactic binary tree
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        List[BT]: reconstructed subtrees of the syntactic binary tree
    """
    s = Stack()
    subtrees = []

    if len(lst) == 1:
        if isinstance(lst[0], str):
            raise Exception(
                "Singleton post-fix notations must be of type 'sp.Expr' or 'float'"
            )
        return [BT(lst[0])]

    for elm in lst:
        print(elm)
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)

        if isinstance(elm, str):
            if elm not in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")

            arity: int = operators[elm]
            if arity >= 3:
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(
                    f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it"
                )

            if arity == 1:
                arg = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), BT):
                #         raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
                if isinstance(arg, BT):
                    node = BT(elm, arg)
                else:
                    node = BT(elm, BT(arg))
                s.put(node)
                subtrees.append(node)

            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()

                if not isinstance(arg1, BT) and not isinstance(arg2, BT):
                    node = BT(elm, BT(arg1), BT(arg2))
                else:
                    if isinstance(arg1, BT) and not isinstance(arg2, BT):
                        node = BT(elm, arg1, BT(arg2))
                    else:
                        if not isinstance(arg1, BT) and isinstance(arg2, BT):
                            node = BT(elm, BT(arg1), arg2)
                        else:
                            node = BT(elm, arg1, arg2)
                s.put(node)
                subtrees.append(node)

    return subtrees


def expr_from_postfix(
    lst: List[sp.Symbol | str | float | str], operators: Dict[str, int]
) -> sp.Expr:
    """
    This function reconstructs a syntactic expression from the postorder traversal of one of its representing binary trees

    Args:
        traversal (List[sp.Symbol  |  str]): postorder traversal of the representing syntactic binary tree
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        sp.Expr: reconstructed syntactic expression
    """
    s = Stack()
    subexpressions = []

    if len(lst) == 1:
        if isinstance(lst[0], str):
            raise Exception(
                "Singleton post-fix notations must be of type 'sp.Expr', 'int' or 'float"
            )
        return lst[0]  # type: ignore

    for elm in lst:
        if isinstance(elm, (sp.Expr, float, int)):
            s.put(elm)

        if isinstance(elm, str):
            if elm not in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")

            arity: int = operators[elm]
            if arity >= 3:
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(
                    f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it"
                )

            if arity == 1:
                arg = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), sp.Expr):
                #         raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
                if elm not in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                expr = AVALIBLE_OPS[elm](arg)
                s.put(expr)
                subexpressions.append(expr)

            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), sp.Expr):
                #         raise Exception(f"Expected 2 arguments for operator ({elm}), but 3 or more were given")
                if elm not in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                if elm == "-":
                    expr = sp.Add(arg1, -arg2)
                elif elm == "/":
                    expr = sp.Mul(arg1, sp.Pow(arg2, -1))
                else:
                    expr = AVALIBLE_OPS[elm](arg1, arg2)
                s.put(expr)
                subexpressions.append(expr)

    return subexpressions[-1]


def exprs_from_postfix(
    lst: List[sp.Symbol | str | float], operators: Dict[str, int]
) -> List[sp.Expr]:
    """
    This function reconstructs the syntactic subexpressions from the postorder traversal of one of its representing binary trees

    Args:
        traversal (List[sp.Symbol  |  str]): postorder traversal of the representing syntactic binary tree
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        List[sp.Expr]: reconstructed syntactic subexpressions
    """
    s = Stack()
    subexpressions = []

    for elm in lst:
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)

        if isinstance(elm, str):
            if elm not in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")

            arity: int = operators[elm]
            if arity >= 3:
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(
                    f"Operator({elm}) expected to have arity ({elm}), but {s.size()} argument(s) were passed to it"
                )

            if arity == 1:
                arg = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), sp.Expr):
                #         raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
                if elm not in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                expr = AVALIBLE_OPS[elm](arg)
                s.put(expr)
                subexpressions.append(expr)

            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()
                if not s.is_empty():
                    if not isinstance(s.top(), sp.Expr):
                        raise Exception(
                            f"Expected 2 arguments for operator ({elm}), but 3 or more were given"
                        )
                if elm not in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                if elm == "-":
                    expr = sp.Add(arg1, -arg2)
                elif elm == "/":
                    expr = sp.Mul(arg1, sp.Pow(arg2, -1))
                else:
                    expr = AVALIBLE_OPS[elm](arg1, arg2)
                s.put(expr)
                subexpressions.append(expr)

    return subexpressions


def expr_from_tree(tree: BT, operators: Dict[str, int]) -> sp.Expr:
    """
    This function returns the (shallowly simplified) expression associated with the given syntactic binary tree

    Args:
        tree (BT): syntactic binary tree
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity

    Returns:
        sp.Expr: reconstructed syntactic expression
    """
    return expr_from_postfix(tree.post_order(), operators)


def get_variables(tree: BT) -> set[sp.Symbol]:
    return set(filter(lambda t: isinstance(t, sp.Symbol), tree.post_order()))
