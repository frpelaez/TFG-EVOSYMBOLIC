import heapq
from random import shuffle
from typing import Any, Callable, List, Optional, Tuple


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
    """
    Returns the k largest elements from a list.

    Args:
        items: A list of elements to find the largest from.
        k: The number of largest elements to return.
        key: An optional function to extract a comparison value from each element.
             Same as the key function in functions like sorted(), min(), max(), etc.

    Returns:
        A list containing the k largest elements from the input list.
        The returned list is sorted in descending order.

    Raises:
        ValueError: If k is less than or equal to 0.
        ValueError: If k is greater than the length of the input list.

    Examples:
        >>> k_largest([1, 5, 3, 9, 2, 6], 3)
        [9, 6, 5]

        >>> k_largest(['apple', 'banana', 'cherry', 'date', 'elderberry'], 2)
        ['elderberry', 'date']

        >>> k_largest(['apple', 'banana', 'cherry', 'date'], 2, key=len)
        ['banana', 'cherry']
    """
    # Validate input parameters
    if k <= 0:
        raise ValueError("k must be greater than 0")

    if k > len(items):
        raise ValueError(
            f"k ({k}) cannot be greater than the length of the input list ({len(items)})"
        )

    # Use heapq.nlargest to efficiently find the k largest elements
    return heapq.nlargest(k, items, key=key)
