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
"""Base module for the learn pdfa implementation."""
from enum import Enum
from typing import Callable, Dict

from pdfa_learning.learn_pdfa.balle.core import learn_pdfa as balle_learn_pdfa
from pdfa_learning.learn_pdfa.palmer.core import learn_pdfa as palmer_learn_pdfa
from pdfa_learning.pdfa import PDFA


class Algorithm(Enum):
    """Enumeration of supported PAC learning algorithms for PDFAs."""

    PALMER = "palmer"
    BALLE = "balle"


_algorithm_to_function: Dict[Algorithm, Callable] = {
    Algorithm.PALMER: palmer_learn_pdfa,
    Algorithm.BALLE: balle_learn_pdfa,
}


def learn_pdfa(algorithm: Algorithm = Algorithm.BALLE, **kwargs) -> PDFA:
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm.
    :return: the learnt PDFA.
    """
    return _algorithm_to_function[algorithm](**kwargs)
