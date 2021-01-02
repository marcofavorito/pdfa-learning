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
