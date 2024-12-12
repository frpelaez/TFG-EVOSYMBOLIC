import sys
from typing import List, Dict, Callable, Any
from random import random, randint, choice, choices

import sympy as sp
import numpy as np

from datastructures import BT, Stack

sys.setrecursionlimit(10**6)

AVALIBLE_OPS: Dict[str, Callable] = {"+": sp.Add,
                                    "-": sp.Add,
                                    "*": sp.Mul,
                                    "/": sp.Mul,
                                    "pow": sp.Pow,
                                    "exp": sp.exp,
                                    "log": sp.log,
                                    "sin": sp.sin,
                                    "cos": sp.cos}


def tree_from_postfix(traversal: List[sp.Symbol | str], operators: Dict[str, int]) -> BT:
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
    
    if len(traversal) == 1 :
        if not isinstance(traversal[0], (sp.Expr, float)):
            raise Exception("Singleton post-fix notations must be of type 'sp.Expr' or 'float'")
        return BT(traversal[0])
    
    for elm in traversal:
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)
            
        if isinstance(elm, str):
            if not elm in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")
            
            arity: int = operators[elm]
            if arity >= 3: 
                raise Exception("Maximum supported arity is currently 2")
            if s.size() < arity:
                raise Exception(f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it")
            
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
            
    return subtrees[-1]


def trees_from_postfix(lst: List[sp.Symbol | str], operators: Dict[str, int]) -> List[BT]:
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
    
    if len(lst) == 1 :
        if isinstance(lst[0], str):
            raise Exception("Singleton post-fix notations must be of type 'sp.Expr' or 'float'")
        return [BT(lst[0])]
    
    for elm in lst:
        print(elm)
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)
            
        if isinstance(elm, str):
            if not elm in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")
            
            arity: int = operators[elm]
            if arity >= 3: 
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it")
            
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


def expr_from_postfix(lst: List[sp.Symbol | str], operators: Dict[str, int]) -> sp.Expr:
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
            raise Exception("Singleton post-fix notations must be of type 'sp.Expr' or 'float")
        return lst[0]
    
    for elm in lst:
        if isinstance(elm, (sp.Expr, float)):
            s.put(elm)
            
        if isinstance(elm, str):
            if not elm in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")
            
            arity: int = operators[elm]
            if arity >= 3: 
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(f"Operator ({elm}) expected to have arity {arity}, but {s.size()} argument(s) were passed to it")
            
            if arity == 1:
                arg = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), sp.Expr):
                #         raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
                if not elm in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                expr = AVALIBLE_OPS[elm](arg)
                s.put(expr)
                subexpressions.append(expr)
                
            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()
                if not s.is_empty():
                    if not isinstance(s.top(), sp.Expr):
                        raise Exception(f"Expected 2 arguments for operator ({elm}), but 3 or more were given")
                if not elm in AVALIBLE_OPS:
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


def exprs_from_postfix(lst: List[sp.Symbol | str], operators: Dict[str, int]) -> List[sp.Expr]:
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
            if not elm in operators.keys():
                raise Exception(f"Found an unexpected operator -> ({elm})")
            
            arity: int = operators[elm]
            if arity >= 3: 
                raise Exception("Maximum arity supported is currently 2")
            if s.size() < arity:
                raise Exception(f"Operator({elm}) expected to have arity ({elm}), but {s.size()} argument(s) were passed to it")
            
            if arity == 1:
                arg = s.pop()
                # if not s.is_empty():
                #     if not isinstance(s.top(), sp.Expr):
                #         raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
                if not elm in AVALIBLE_OPS:
                    raise Exception(f"({elm}) operator has not been implemented yet")
                expr = AVALIBLE_OPS[elm](arg)
                s.put(expr)
                subexpressions.append(expr)
                
            if arity == 2:
                arg2 = s.pop()
                arg1 = s.pop()
                if not s.is_empty():
                    if not isinstance(s.top(), sp.Expr):
                        raise Exception(f"Expected 2 arguments for operator ({elm}), but 3 or more were given")
                if not elm in AVALIBLE_OPS:
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


def generate_tree(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, *, current_depth: int = 1) -> BT:
    """
    This function generates a random syntactic binary tree

    Args:
        operators (Dict[str, int])
        vars (List[sp.Symbol]): list of variables
        max_depth (int): maximum depth for the generated tree

    Returns:
        BT
    """
    if current_depth < max_depth:
        node_type: str = choices(["operator", "terminal"], [0.7, 0.3])[0]
    else:
        node_type = "terminal"
        
    if node_type == "operator":
        node: sp.Symbol | str = choice(list(operators.keys()))
        arity: int = operators[node]
        
        if arity == 1:
            left_child: BT = generate_tree(operators, vars, max_depth - 1) 
            return BT(node, left_child)
        
        if arity == 2:
            left_child: BT = generate_tree(operators, vars, max_depth - 1)
            right_child: BT = generate_tree(operators, vars, max_depth - 1)
            return BT(node, left_child, right_child)
        
    if node_type == "terminal":
        node: sp.Symbol | str = choice(vars)
        return BT(node)
    
    return BT()
    

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
        while nodes_left > 0 and counter <= cross_point1:
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
        while nodes_left > 0 and counter <= cross_point2:
            next_node = pst1[cross_point2 - counter - 1]
            if isinstance(next_node, (sp.Symbol, float)):
                nodes_left -= 1
                
            if isinstance(next_node, str):
                nodes_left += operators[next_node] - 1
            
            counter += 1
            
        extractedIndex: int = cross_point2 - counter
        
    offspring_pst: List[Any] = pst2[: extractedIndex] + extracted + pst2[cross_point2 + 1 :]
    
    return tree_from_postfix(offspring_pst, operators)

    
def mutation(tree: BT, vars: List[sp.Symbol], operators: Dict[str, int], *, variant: str = "nodal", constants_range: int = 1) -> BT:
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
    print("Selected point:", mutation_point, selected_node)
    
    match variant:
        case "nodal":
            if isinstance(selected_node, (sp.Symbol, float)):
                if random() < 0.75:
                    pst[mutation_point] = choice(vars)
                    print("var")
                else:
                    print("cte")
                    cte: float = constants_range * round(2 * random() - 1, 4)
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
                depth: int = tree_from_postfix(extracted_subtree_pst, ops).depth()
            
            new_subtree: BT = generate_tree(ops, vars, depth)
            new_subpst: List[Any] = new_subtree.post_order()
            
            pst = pst[: extractedIndex] + new_subpst + pst[mutation_point + 1 :]
            
            return tree_from_postfix(pst, ops)
            
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
                depth: int = tree_from_postfix(extracted_subtree_pst, ops).depth()
            
            new_subtree: BT = generate_tree(ops, vars, depth - 1)
            new_subpst: List[Any] = new_subtree.post_order()
            
            pst = pst[: extractedIndex] + new_subpst + pst[mutation_point + 1 :]
            
            return tree_from_postfix(pst, ops)
        
        case _:
            raise Exception("Avalible mutation variants are 'nodal' (by default), 'complete' and 'shrinking' ")
    
    
def main() -> None:
    
    vars = sp.symbols("x0:5")
    vars_list = list(vars)
    x0, x1, x2, x3, x4 = vars
    C = sp.Symbol("C")
    vars_list.append(C)
    print("Using variables:", vars_list)
    
    ops: Dict[str, int] = {"+":2, 
                           "-": 2,
                           "*": 2,
                           "/": 2,
                           "pow": 2,
                           "exp": 1,
                           "log": 1,
                           "sin": 1,
                           "cos": 1}
    print("Operators dictionary (name, arity)", ops)
    
    t = BT("+",
          BT("-",
                BT("exp",
                      BT(x1)),
                BT("pow",
                      BT(x0),
                      BT("/",
                            BT(C),
                            BT(x3)))),
                      
          BT("*",
                BT(C),
                BT(x4)))
    print("Tree:")
    t.show()
    print("\nIt has a total of", t.size(), "nodes")

    print("\nPostfix of t:", t.post_order(), sep="\n")
    print("\nReconstruction of tree from postfix notation:")
    rt: BT = tree_from_postfix(t.post_order(), ops)
    rt.show()

    print("\nSympy expression from t:")
    print(expr_from_postfix(t.post_order(), ops))
    
    x = sp.Symbol('x')
    t2: BT = generate_tree(ops, [x], 3)
    print("\nRandom generated tree:")
    t2.show()
    print("\nAnd now a copy of it")
    t2c: BT = t2.copy()
    t2c.show()
    print("\nIts postorder traversal is:")
    t2_pst = t2.post_order()
    print(t2_pst)
    print("\nAnd its corresponding Sympy expression is:")
    print(expr_from_postfix(t2_pst, ops))
    
if __name__ == "__main__":
    main()
    ops: Dict[str, int] = {"+":2, 
                           "-": 2,
                           "*": 2,
                           "/": 2,
                           "pow": 2,
                           "log": 1,
                           "exp": 1,
                           "sin": 1,
                           "cos": 1}
    vars = sp.symbols("x0:5")
    C = sp.Symbol("C")
    vlist = list(vars) + [C]