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
"""
Package that contains the implementation of [1].

- [1] Palmer N., Goldberg P.W. (2005)
      PAC-Learnability of Probabilistic Deterministic Finite State Automata
      in Terms of Variation Distance.
      In: Jain S., Simon H.U., Tomita E. (eds) Algorithmic Learning Theory. ALT 2005.
      Lecture Notes in Computer Science, vol 3734. Springer, Berlin, Heidelberg.
      https://doi.org/10.1007/11564089_14
"""
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
)
