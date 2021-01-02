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
"""Entrypoint for the algorithm."""

import pprint

from pdfa_learning.learn_pdfa import logger
from pdfa_learning.learn_pdfa.palmer.learn_probabilities import learn_probabilities
from pdfa_learning.learn_pdfa.palmer.learn_subgraph import learn_subgraph
from pdfa_learning.learn_pdfa.palmer.params import PalmerParams


def learn_pdfa(**kwargs):
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm (see the PalmerParams class).
    :return: the learnt PDFA.
    """
    params = PalmerParams(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    vertices, transitions = learn_subgraph(params)
    logger.info(f"Number of vertices: {len(vertices)}.")
    logger.info(f"Transitions: {pprint.pformat(transitions)}.")
    pdfa = learn_probabilities((vertices, transitions), params)
    return pdfa
