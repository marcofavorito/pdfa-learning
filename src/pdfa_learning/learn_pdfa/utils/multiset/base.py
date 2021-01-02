"""Base module."""
from abc import ABC, abstractmethod
from typing import Iterator, Sequence, Set, Tuple

from pdfa_learning.types import Word


class Multiset(ABC):
    """Abstract multiset."""

    @abstractmethod
    def get_counts(self, trace: Word) -> int:
        """Get counts."""

    @abstractmethod
    def add(self, t: Word, times: int = 1) -> None:
        """
        Add a trace in the multiset.

        :param t: the trace to add.
        :param times: how many times it should be added.
        :return: None
        """

    @property
    @abstractmethod
    def size(self) -> int:
        """Get the size."""

    @abstractmethod
    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""

    @abstractmethod
    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""

    @property
    @abstractmethod
    def traces(self) -> Set[Word]:
        """Get the traces."""

    def elements(self) -> Iterator[Word]:
        """Get the set of traces."""
        for trace, count in self.items():
            for _ in range(count):
                yield trace

    @abstractmethod
    def items(self) -> Iterator[Tuple[Word, int]]:
        """Get an iterator of tuples (trace, count)."""

    def update(self, sample: Sequence[Word]):
        """Add items."""
        for t in sample:
            self.add(t)

    def __len__(self) -> int:
        """Get the length."""
        return self.size

    def __iter__(self):
        """Get the traces."""
        return iter(self.traces)

    def __getitem__(self, item):
        """Get the count."""
        return self.get_counts(item)

    def values(self) -> Sequence[int]:
        """Get the values."""
        return [v for _, v in self.items()]
