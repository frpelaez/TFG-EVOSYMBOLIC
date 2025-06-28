import heapq
from random import shuffle
from typing import Any, Callable, List, Optional, Tuple

import numpy as np


def Transpose(matrix: List[List[Any]]) -> List[List[Any]]:
    return list(map(list, zip(*matrix)))


def Add(seq1: List, seq2: List) -> List:
    assert len(seq1) == len(seq2)
    return [e1 + e2 for e1, e2 in zip(seq1, seq2)]


def random_pairs[T](items: List[T]) -> List[Tuple[T, T]]:
    shuffled = items.copy()
    shuffle(shuffled)
    return [(shuffled[i], shuffled[i + 1]) for i in range(0, len(items) - 1, 2)]


def random_groups[T](items: List[T], group_size: int) -> List[Tuple[T, ...]]:
    if len(items) % group_size != 0:
        raise ValueError("The group size must divide the length of the item list")

    shuffled = items.copy()
    shuffle(shuffled)
    return [
        tuple(shuffled[i + k] for k in range(group_size))
        for i in range(0, len(items) - 1, group_size)
    ]


def k_largest[T](
    items: List[T], k: int, key: Optional[Callable[[T], Any]] = None
) -> List[T]:
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > len(items):
        raise ValueError(
            f"k ({k}) cannot be greater than the length of the input list ({len(items)})"
        )

    return heapq.nlargest(k, items, key=key)


def calculate_deriv(data):
    t, y = data
    n = len(y)
    y_prime = [None] * n
    y_prime[0] = (y[1] - y[0]) / (t[1] - t[0])
    for i in range(1, n - 1):
        y_prime[i] = (y[i + 1] - y[i - 1]) / (t[i + 1] - t[i - 1])
    y_prime[n - 1] = (y[n - 1] - y[n - 2]) / (t[n - 1] - t[n - 2])
    return np.array(y_prime)
