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
"""Conftest module."""
import logging
import multiprocessing
import shutil
import sys
import tempfile
from contextlib import contextmanager

import pytest

# this is to import fixtures
from tests.fixtures import (  # noqa: E402, F401
    pdfa_one_state,
    pdfa_sequence_three_states,
    pdfa_two_states,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def nb_processes():
    """Get the number of processes available."""
    result = multiprocessing.cpu_count()
    logger.debug(f"Number of cpus: {result}")
    return result


@contextmanager
def tempdir():
    """Create a temporary directory as a context manager."""
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        try:
            shutil.rmtree(path)
        except IOError:
            sys.stderr.write("Failed to clean up temp dir {}".format(path))


@pytest.fixture(params=["pdfa_one_state"])
def pdfas(request):
    """Get a list of PDFAs."""
    return request.getfuncargvalue(request.param)
