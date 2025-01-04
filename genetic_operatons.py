from random import random, randint, choice, choices
from typing import List, Dict, Any

import sympy as sp

from datastructures import BT
from symbolic_expression import tree_from_postfix, expr_from_postfix

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
        node_type: str = choices(["operator", "terminal"], [0.9, 0.1])[0]
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
    

def generate_population(operators: Dict[str, int], vars: List[sp.Symbol], max_depth: int, population_size: int) -> List[BT]:
    """
    This function generates a population of random syntactic binary trees

    Args:
        operators (Dict[str, int]): dictionary of the operators present in the tree with their arity
        vars (List[sp.Symbol]): list of variables
        max_depth (int): maximum depth for the generated trees
        population_size (int): number of trees in the population

    Returns:
        List[BT]
    """
    return [generate_tree(operators, vars, max_depth) for _ in range(population_size)]


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
        

def selection(population: List[BT], fitness: List[float], *, method: str = "tournament", tournament_size: int = 2) -> BT:
    
    if method == "tournament":
        selected_indices: List[int] = choices(list(range(len(population))), fitness, k=tournament_size)
        winner_index: int = max(selected_indices, key=lambda i: fitness[i])
        return population[winner_index]
    
    if method == "fitness-proportional":
        total_fitness: float = sum(fitness)
        probabilities: List[float] = [f / total_fitness for f in fitness]
        selected_index: int = choices(list(range(len(population))), probabilities)[0]
        return population[selected_index]
    
    if method == "elitisim":
        winner_index: int = max(range(len(population)), key=lambda i: fitness[i])
        return population[winner_index]
    
    raise Exception("Available selection methods are 'tournament' (by default), 'fitness-proportional' and 'elitism'")

 
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
    
if __name__ == "__main__":
    main()