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
"""Tests for Palmer & Goldberg PDFA learning algorithm."""
from pdfa_learning.learn_pdfa.base import Algorithm
from pdfa_learning.pdfa import PDFA
from tests.pdfas import make_pdfa_one_state, make_pdfa_two_state
from tests.test_learn_pdfa.base import PALMER_CONFIG, BaseTestLearnPDFA


class TestOneState(BaseTestLearnPDFA):
    """Test PDFA learning of one state PDFA."""

    ALGORITHM = Algorithm.PALMER
    CONFIG = PALMER_CONFIG
    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_one_state()


class TestTwoState(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    ALGORITHM = Algorithm.PALMER
    CONFIG = PALMER_CONFIG
    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_two_state()
