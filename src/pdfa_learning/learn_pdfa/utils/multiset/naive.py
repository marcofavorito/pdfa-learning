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
"""Vanilla implementation of a multiset."""
from collections import Counter
from typing import Iterator, Set, Tuple

from pdfa_learning.learn_pdfa.utils.multiset.base import Multiset
from pdfa_learning.types import Word


class NaiveMultiset(Multiset):
    """Implement a multiset in a naive way - using a counter."""

    def __init__(self):
        """Initialize the multiset."""
        self._counter = Counter()

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return self._counter[trace]

    def add(self, t: Word, times: int = 1) -> None:
        """Add an item to the multiset."""
        self._counter.update({t: times})

    @property
    def size(self) -> int:
        """Get the size of the multiset."""
        return sum(self._counter.values())

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        if self.size == 0:
            return 0
        return self._counter[t] / self.size

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        if self.size == 0:
            return 0
        p = 0.0
        for string in self._counter.keys():
            for i in range(len(string) + 1):
                prefix, suffix = string[:i], string[i:]
                if prefix != t:
                    continue
                p += self._counter[prefix + suffix]

        return p / self.size

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return set(map(tuple, self._counter.keys()))

    def items(self) -> Iterator[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return iter(self._counter.items())
