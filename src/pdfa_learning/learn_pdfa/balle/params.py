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
"""Params class for Balle's algorithm."""
import pprint
from dataclasses import dataclass
from typing import Collection, Optional

from pdfa_learning.helpers.base import assert_
from pdfa_learning.learn_pdfa.utils.generator import Generator
from pdfa_learning.types import Word


@dataclass(frozen=True, repr=False)
class BalleParams:
    """
    Parameters for the (Balle et al., 2013) learning algorithm.

    sample_generator: the sample generator from the true PDFA.
    alphabet_size: the alphabet size.
    epsilon: the tolerance error.
    delta: the failure probability for the subgraph construction.
    delta: the failure probability for the probability estimation.
    mu: the prefix-distinguishability factor.
    n: the upper bound of the number of states.
    """

    sample_generator: Optional[Generator] = None
    dataset: Optional[Collection[Word]] = None
    nb_samples: int = 10000
    n: int = 10
    alphabet_size: int = 5
    delta: float = 0.1
    epsilon: float = 0.1
    with_smoothing: bool = False
    with_ground: bool = False
    with_infty_norm: bool = True

    def __post_init__(self):
        """Validate inputs."""
        assert_(
            0 < self.delta < 1.0,
            "Delta must be a non-zero probability.",
        )
        assert_(
            ((self.dataset is None) != (self.sample_generator is None)),
            "Only one between dataset and sample generator must be specified.",
        )

    @property
    def delta_0(self) -> float:
        """Get the error probability of test distinct."""
        d = self.delta
        s = self.alphabet_size
        n = self.n
        return d / (n * (n * s + s + 1))

    def get_gamma_min(self, expected_length: float) -> float:
        """
        Get the smoothing probability.

        :param expected_length: the expected length of traces.
        :return:
        """
        return self.epsilon / 4 / expected_length / (self.alphabet_size + 1)

    def __repr__(self):
        """Get the representation."""
        return pprint.pformat(
            {
                "sample_generator": self.sample_generator,
                "dataset_size": len(self.dataset) if self.dataset else None,
                "nb_samples": self.nb_samples,
                "n": self.n,
                "alphabet_size": self.alphabet_size,
                "delta": self.delta,
                "epsilon": self.epsilon,
                "with_smoothing": self.with_smoothing,
                "with_ground": self.with_ground,
                "with_infty_norm": self.with_infty_norm,
            }
        )
