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
"""Test generator."""
from abc import abstractmethod

import pytest

from pdfa_learning.learn_pdfa.utils.generator import (
    Generator,
    MultiprocessedGenerator,
    SimpleGenerator,
)
from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.helpers import FINAL_SYMBOL
from tests.pdfas import make_pdfa_one_state


class BaseTestGenerator:
    """Base test class for generators."""

    def make_automaton(self) -> PDFA:
        """Make a PDFA to generate samples from."""
        automaton = make_pdfa_one_state()
        return automaton

    @abstractmethod
    def make_generator(self) -> Generator:
        """Make a sample generator. To be implemented."""

    def setup(self):
        """Set up the test."""
        self.generator = self.make_generator()

    @pytest.mark.parametrize("nb_samples", [5, 10, 20, 100])
    def test_generation(self, nb_samples):
        """Test generator 'sample' method."""
        sample = self.generator.sample(n=nb_samples)
        assert len(sample) == nb_samples
        assert all(character in {0, 1, FINAL_SYMBOL} for s in sample for character in s)


class TestSimpleGenerator(BaseTestGenerator):
    """Test simple generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return SimpleGenerator(self.make_automaton())


class TestMultiprocessedGenerator(BaseTestGenerator):
    """Test multiprocessed generator."""

    def make_generator(self) -> Generator:
        """Make a generator for testing."""
        return MultiprocessedGenerator(SimpleGenerator(self.make_automaton()))


def test_multiprocess_generator_helper_function():
    """Test multiprocess generator helper function."""
    automaton = make_pdfa_one_state()
    sample = MultiprocessedGenerator._job(10, SimpleGenerator(automaton).sample)
    assert len(sample) == 10
    assert all(character in {0, 1, FINAL_SYMBOL} for s in sample for character in s)
