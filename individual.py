from datastructures import BT


class Individual:
    def __init__(self, tree: BT, fitness: float) -> None:
        self._tree: BT = tree
        self._fitness: float = fitness

    @property
    def tree(self) -> BT:
        return self._tree

    @tree.setter
    def tree(self, tree: BT) -> None:
        self._tree = tree

    @property
    def fitness(self) -> float:
        return self._fitness

    @fitness.setter
    def parent_fitness(self, value: float) -> None:
        self._parent_fitness = value

    def copy(self) -> "Individual":
        return Individual(self.tree.copy(), self._fitness)
