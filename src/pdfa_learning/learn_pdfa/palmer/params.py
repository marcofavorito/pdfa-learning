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
"""Params class for Palmer algorithm."""

from dataclasses import dataclass
from typing import Optional

from pdfa_learning.helpers.base import assert_
from pdfa_learning.learn_pdfa.utils.generator import Generator


@dataclass(frozen=True)
class PalmerParams:
    """
    Parameters for the (Palmer  and Goldberg, 2005) learning algorithm.

    sample_generator: the sample generator from the true PDFA.
    alphabet_size: the alphabet size.
    epsilon: the tolerance error.
    delta: the failure probability for the subgraph construction.
    delta: the failure probability for the probability estimation.
    mu: the distinguishability factor.
    n: the upper bound of the number of states.
    """

    sample_generator: Generator
    alphabet_size: int
    epsilon: float = 0.05
    delta_1: float = 0.1
    delta_2: float = 0.1
    mu: float = 0.4
    n: int = 3
    # debug parameters - force upper bounds
    m0_max_debug: Optional[int] = None
    n1_max_debug: Optional[int] = None
    n2_max_debug: Optional[int] = None

    def __post_init__(self):
        """Validate inputs."""
        assert_(
            self.delta_1 + self.delta_2 <= 1.0,
            "Sum of two probabilities cannot be greater than 1.",
        )
