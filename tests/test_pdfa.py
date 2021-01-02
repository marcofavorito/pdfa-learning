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
"""Main test module."""
from pathlib import Path

import numpy as np
import pytest

from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.base import FINAL_STATE
from pdfa_learning.pdfa.helpers import FINAL_SYMBOL
from pdfa_learning.pdfa.render import to_graphviz
from tests.conftest import tempdir


def test_pdfa_example():
    """Test the PDFA class with a simple instantiation."""
    automaton = PDFA(
        3,
        2,
        {
            0: {0: (2, 0.1), 1: (1, 0.9)},
            1: {0: (1, 0.1), 1: (2, 0.9)},
            2: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )

    with tempdir() as tmp:
        to_graphviz(automaton).render(Path(tmp, "output"))


def test_nb_states_zero():
    """Test empty set of states."""
    with pytest.raises(
        AssertionError, match="Number of states must be greater than zero."
    ):
        PDFA(0, 0, {})


def test_alphabet_size_zero():
    """Test empty alphabet.."""
    with pytest.raises(
        AssertionError, match="Alphabet size must be greater than zero."
    ):
        PDFA(1, 0, {})


def test_not_a_probability():
    """Test not a probability."""
    with pytest.raises(AssertionError, match="'42.0' is not a probability."):
        PDFA(1, 1, {0: {0: (FINAL_STATE, 42.0)}})


def test_sum_outgoing_transitions_probabilities_greater_than_one():
    """Test the case when the sum of out transition probabilities is greater than one."""
    with pytest.raises(
        AssertionError, match="Outgoing probability from state 0 do not sum to 1."
    ):
        PDFA(
            2,
            2,
            {
                0: {
                    0: (0, 0.999),
                    1: (1, 0.999),
                },
                1: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
            },
        )


def test_wrong_state():
    """Test the case when some destination state is wrong."""
    with pytest.raises(
        AssertionError,
        match="Provided state is not in the set of states, nor is a final state.",
    ):
        PDFA(
            1,
            2,
            {
                0: {
                    0: (0, 0.5),
                    1: (42, 0.5),
                }
            },
        )


def test_wrong_character():
    """Test the case when some character is wrong."""
    with pytest.raises(
        AssertionError, match="Provided character is not in the alphabet."
    ):
        PDFA(
            1,
            2,
            {
                0: {
                    42: (0, 0.5),
                    1: (FINAL_STATE, 0.5),
                }
            },
        )


class TestMethods:
    """Test PDFA's methods."""

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.automaton = PDFA(
            2,
            2,
            {
                0: {
                    0: (0, 0.5),
                    1: (1, 0.5),
                },
                1: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
            },
        )

    def test_successor(self):
        """Test successor."""
        assert self.automaton.get_successor(0, 0) == 0
        assert self.automaton.get_successor(0, 1) == 1
        assert self.automaton.get_successor(1, -1) == self.automaton.final_state

    def test_successors(self):
        """Test the 'get_successors' method."""
        assert self.automaton.get_successors(0) == {0, 1}
        assert self.automaton.get_successors(1) == {self.automaton.final_state}

    def test_transitions(self):
        """Test the transitions attribute."""
        expected_transitions = {
            (0, 0, 0.5, 0),
            (0, 1, 0.5, 1),
            (1, FINAL_SYMBOL, 1.0, self.automaton.final_state),
        }
        actual_transitions = self.automaton.transitions
        assert expected_transitions == actual_transitions

    def test_probability(self):
        """Test the probability of a string."""
        # final state never reached - probability is zero.
        assert self.automaton.get_probability([-1]) == 0.0
        assert self.automaton.get_probability([0, -1]) == 0.0
        assert self.automaton.get_probability([0, 0, -1]) == 0.0

        # final state reached
        assert self.automaton.get_probability([1, -1]) == 0.5
        assert self.automaton.get_probability([0, 1, -1]) == 0.25
        assert self.automaton.get_probability([0, 0, 1, -1]) == 0.125

        # read more symbols from final state gives probability zero.
        assert self.automaton.get_probability([1, 0, -1]) == 0.0
        assert self.automaton.get_probability([0, 1, 0, -1]) == 0.0
        assert self.automaton.get_probability([0, 0, 0, -1]) == 0.0

    def test_sample(self):
        """Test the sample method."""
        nb_samples = 5000
        samples = [self.automaton.sample() for _ in range(nb_samples)]
        # two characters, plus final symbol
        expected_average_length = 2 + 1
        actual_average_length = np.mean([len(w) for w in samples])
        assert np.isclose(expected_average_length, actual_average_length, rtol=0.05)
