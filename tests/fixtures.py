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
"""Definition of PDFAs."""
import pytest

from tests.pdfas import (
    make_pdfa_one_state,
    make_pdfa_sequence_three_states,
    make_pdfa_two_state,
)


@pytest.fixture
def pdfa_one_state():
    """Get a PDFA with one state."""
    return make_pdfa_one_state()


@pytest.fixture
def pdfa_two_states():
    """Get a PDFA with two states."""
    return make_pdfa_two_state()


@pytest.fixture
def pdfa_sequence_three_states(request):
    """Get a PDFA with two states."""
    p1, p2, p3, stop_probability = request.param
    return make_pdfa_sequence_three_states(p1, p2, p3, stop_probability)
