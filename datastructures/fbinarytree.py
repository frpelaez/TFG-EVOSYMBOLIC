from typing import Any, List

from datastructures.fqueue import Queue


class BT:
    def __init__(self, data: Any = None, left: "BT" = None, right: "BT" = None) -> None:  # type: ignore
        self._data = data
        self._left: BT = left
        self._right: BT = right

    def size(self) -> int:
        return len(self.pre_order())

    def root(self) -> Any:
        return self._data

    def left_child(self) -> "BT":
        return self._left

    def right_child(self) -> "BT":
        return self._right

    def depth(self) -> int:
        if self._data is None:
            return 0
        elif self.left_child() is None and self.right_child() is None:
            return 1
        elif self.left_child() is None:
            return 1 + self.right_child().depth()
        elif self.right_child() is None:
            return 1 + self.left_child().depth()
        else:
            return 1 + max(self.left_child().depth(), self.right_child().depth())

    def pre_order(self) -> List[Any]:
        if self is None:
            return []

        lst = []

        lst.append(self._data)
        if self.left_child() is not None:
            lst.extend(self._left.pre_order())
        if self.right_child() is not None:
            lst.extend(self._right.pre_order())

        return lst

    def in_order(self) -> List[Any]:
        if self is None:
            return []

        lst = []

        if self.left_child() is not None:
            lst.extend(self._left.in_order())
        lst.append(self._data)
        if self.right_child() is not None:
            lst.extend(self._right.in_order())

        return lst

    def post_order(self) -> List[Any]:
        if self is None:
            return []

        lst = []

        if self.left_child() is not None:
            lst.extend(self._left.post_order())
        if self.right_child() is not None:
            lst.extend(self._right.post_order())
        lst.append(self._data)

        return lst

    def by_level(self) -> List[Any]:
        if self is None:
            return []

        lst = []

        q = Queue()
        q.enqueue(self)
        while not q.is_empty():
            node = q.first()
            q.move()
            lst.append(node.root())
            if node.left_child() is not None:
                q.enqueue(node.left_child())
            if node.right_child() is not None:
                q.enqueue(node.right_child())

        return lst

    def get_level(self, n: int) -> List[Any]:
        if self is None:
            return []

        lst = []

        currentPair = (self, 0)
        q = Queue()
        q.enqueue(currentPair)
        while not q.is_empty():
            node = q.first()[0]
            currentDepth = q.first()[1]
            q.move()
            if currentDepth == n:
                lst.append(node.root())
            if node.left_child() is not None:
                nextPair = (node.left_child(), currentDepth + 1)
                q.enqueue(nextPair)
            if node.right_child() is not None:
                nextPair = (node.right_child(), currentDepth + 1)
                q.enqueue(nextPair)

        return lst

    def copy(self) -> "BT":
        return BT(self._data, self._left, self._right)

    def show(self) -> None:
        showBT(self)


def showBT(node: BT, level: int = 0) -> None:
    if node is not None:
        showBT(node.right_child(), level + 1)
        print(" " * 4 * level + "-> " + str(node._data))
        showBT(node.left_child(), level + 1)


# t = BT(1,
#        BT(2,
#           BT(4,
#              BT(7)),
#           BT(9)),
#        BT(3,
#           BT(5),
#           BT(6)))
# t.show()
# print("Dpeth of t =", t.depth())
# t.left_child().show()
# print("Depth of left child =", t.left_child().depth())
# t.right_child().show()
# print("Depth of right child =", t.right_child().depth())
# print(t.pre_order())
# print(t.in_order())
# print(t.post_order())
