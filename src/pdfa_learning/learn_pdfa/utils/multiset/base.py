# -*- coding: utf-8 -*-
#
# Copyright 2020 Marco Favorito
#
# ------------------------------
#
# This file is part of pdfa-learning.
#
# pdfa-learning is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pdfa-learning is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pdfa-learning.  If not, see <https://www.gnu.org/licenses/>.
#
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
