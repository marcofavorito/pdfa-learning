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
