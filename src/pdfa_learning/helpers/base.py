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
"""Base helper module."""
from pdfa_learning.types import TransitionFunctionDict


def assert_(condition: bool, message: str = ""):
    """User-defined assert."""
    if not condition:
        raise AssertionError(message)


def normalize(f: TransitionFunctionDict) -> TransitionFunctionDict:
    """Normalize a transition function."""
    result: TransitionFunctionDict = {}

    for start, outgoing in f.items():
        result[start] = {}
        total = sum([p for _, (_, p) in outgoing.items()])
        for char, (end, prob) in outgoing.items():
            result[start][char] = (end, prob / total)

    return result
