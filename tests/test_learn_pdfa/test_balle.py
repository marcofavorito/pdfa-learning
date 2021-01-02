"""Main test module."""

from pdfa_learning.pdfa import PDFA
from tests.pdfas import (
    make_pdfa_one_state,
    make_pdfa_sequence_three_states,
    make_pdfa_two_state,
    make_reber_grammar,
)
from tests.test_learn_pdfa.base import BaseTestLearnPDFA


class TestOneState(BaseTestLearnPDFA):
    """Test PDFA learning of one state PDFA."""

    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_one_state()


class TestTwoState(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    ALPHABET_LEN = 2

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_two_state()


class TestSequenceThreeStates(BaseTestLearnPDFA):
    """Test PDFA learning of two state PDFA."""

    PROBABILITIES = (0.4, 0.3, 0.2, 0.1)
    ALPHABET_LEN = 3
    OVERWRITE_CONFIG = dict(nb_samples=200000)

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_pdfa_sequence_three_states(*cls.PROBABILITIES)


class TestReber(BaseTestLearnPDFA):
    """Test PDFA learning on Reber PDFA."""

    ALPHABET_LEN = 6

    @classmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""
        return make_reber_grammar()
