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

from pdfa_learning.pdfa import PDFA
from tests.pdfas import (
    make_pdfa_one_state,
    make_pdfa_sequence_three_states,
    make_pdfa_two_state,
    make_reber_grammar,
)
from tests.test_learn_pdfa.base import BaseTestLearnPDFA


class TestOneState(BaseTestLearnPDFA):
    """Test PDFA learning of one state PDFA."""

    ALPHABET_LEN = 2
    OVERWRITE_CONFIG = dict(nb_samples=50000)

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_one_state()


class TestTwoState(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_two_state()


class TestSequenceThreeStates(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    PROBABILITIES = (0.4, 0.3, 0.2, 0.1)
    ALPHABET_LEN = 3
    OVERWRITE_CONFIG = dict(nb_samples=200000)

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_sequence_three_states(*cls.PROBABILITIES)


class TestReber(BaseTestLearnPDFA):
    """Test PDFA learning on Reber PDFA."""

    ALPHABET_LEN = 6

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_reber_grammar()
