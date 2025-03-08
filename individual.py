from datastructures import BT


class Individual:
    
    def __init__(self, tree: BT, parent_fitness: float) -> None:
        
        self._tree: BT = tree
        self._parent_fitness: float = parent_fitness

    @property
    def tree(self) -> BT:
        
        return self._tree
    
    @tree.setter
    def x(self, tree: BT) -> None:
        
        self._tree = tree
        
    @property
    def parent_fitness(self) -> float:
        
        return self._parent_fitness
    
    @parent_fitness.setter
    def parent_fitness(self, value: float) -> None:
        
        self._parent_fitness = value
        
    def copy(self) -> "Individual":
        
        return Individual(self.tree.copy(), self.parent_fitness)