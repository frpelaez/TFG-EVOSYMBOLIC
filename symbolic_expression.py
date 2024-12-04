import sys
from typing import List, Dict, Callable
from random import random, choice, choices

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
                                    "sin": sp.sin,
                                    "cos": sp.cos}


def tree_from_postfix(lst: List[sp.Symbol | str], operators: Dict[str, int]) -> BT:
    
    s = Stack()
    subtrees = []
    
    if len(lst) == 1 :
        if not isinstance(lst[0], sp.Expr):
            raise Exception("Singleton post-fix notations must be of type 'sp.Expr'")
        return BT(lst[0])
    
    for elm in lst:
        if isinstance(elm, sp.Expr):
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
                if not s.is_empty():
                    if not isinstance(s.top(), BT):
                        raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
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
    
    s = Stack()
    subtrees = []
    
    if len(lst) == 1 :
        if isinstance(lst[0], str):
            raise Exception("Singleton post-fix notations must be of type sp.Expr")
        return [BT(lst[0])]
    
    for elm in lst:
        print(elm)
        if isinstance(elm, sp.Expr):
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
                if not s.is_empty():
                    if not isinstance(s.top(), BT):
                        raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
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
    
    s = Stack()
    subexpressions = []
    
    if len(lst) == 1:
        if isinstance(lst[0], str):
            raise Exception("Singleton post-fix notations must be of type sp.Expr")
        return lst[0]
    
    for elm in lst:
        if isinstance(elm, sp.Expr):
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
                if not s.is_empty():
                    if not isinstance(s.top(), sp.Expr):
                        raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
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
    
    s = Stack()
    subexpressions = []
    
    for elm in lst:
        if isinstance(elm, sp.Expr):
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
                if not s.is_empty():
                    if not isinstance(s.top(), sp.Expr):
                        raise Exception(f"Expected 1 argument for operator ({elm}), but 2 or more were given")
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


def expr_from_tree(tree: BT, ops: Dict[str, int]) -> sp.Expr:
    
    return expr_from_postfix(tree.post_order(), ops)


def generate_tree(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, current_depth: int = 1) -> BT:
    
    if current_depth < max_depth:
        node_type: str = choices(["operator", "terminal"], [0.6, 0.4])[0]
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

    print("\nPostfix of t:", t.post_order(), sep="\n")
    print("\nReconstruction of tree from postfix notation:")
    rt: BT = tree_from_postfix(t.post_order(), ops)
    rt.show()

    print("\nSympy expression from t:")
    print(expr_from_postfix(t.post_order(), ops))
    
if __name__ == "__main__":
    main()
