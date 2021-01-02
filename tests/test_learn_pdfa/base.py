"""Main test module."""
from abc import abstractmethod
from copy import copy
from typing import Dict

import numpy as np
from hypothesis import assume, given, strategies

from pdfa_learning.learn_pdfa.base import Algorithm, learn_pdfa
from pdfa_learning.learn_pdfa.utils.generator import (
    MultiprocessedGenerator,
    SimpleGenerator,
)
from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.helpers import FINAL_SYMBOL

BALLE_CONFIG = dict(
    algorithm=Algorithm.BALLE,
    nb_samples=20000,
    delta=0.1,
    n=10,
)

PALMER_CONFIG = dict(
    algorithm=Algorithm.PALMER,
    epsilon=0.1,
    delta_1=0.05,
    delta_2=0.05,
    mu=0.1,
    n=5,
    n1_max_debug=100000,
    n2_max_debug=100000,
    m0_max_debug=100000 / 10,
)


class BaseTestLearnPDFA:
    """Base test class for PDFA learning."""

    NB_PROCESSES = 8
    ALGORITHM = Algorithm.BALLE
    CONFIG: Dict = BALLE_CONFIG
    ALPHABET_LEN = 3
    OVERWRITE_CONFIG: Dict = {}
    WITH_SMOOTHING = False
    RTOL = 0.1

    @classmethod
    @abstractmethod
    def _make_automaton(cls) -> PDFA:
        """Make automaton."""

    @classmethod
    def setup_class(cls):
        """Set up the test."""
        cls.expected = cls._make_automaton()
        generator = MultiprocessedGenerator(
            SimpleGenerator(cls.expected), nb_processes=cls.NB_PROCESSES
        )

        config = copy(cls.CONFIG)
        config.update(cls.OVERWRITE_CONFIG)
        cls.actual = learn_pdfa(
            sample_generator=generator,
            alphabet_size=cls.expected.alphabet_size,
            with_smoothing=cls.WITH_SMOOTHING,
            **config
        )

    def test_same_nb_states(self):
        """Test same number of states."""
        assert len(self.actual.states) == len(self.expected.states)

    @given(
        trace=strategies.lists(
            strategies.integers(min_value=0, max_value=ALPHABET_LEN - 1),
            min_size=0,
            max_size=20,
        )
    )
    def test_equivalence(self, trace):
        """Test equivalence between expected and learned PDFAs."""
        max_value = self.expected.alphabet_size - 1
        assume(all(x <= max_value for x in trace))
        actual_trace = tuple(trace) + (FINAL_SYMBOL,)
        actual_prob = self.actual.get_probability(actual_trace)
        expected_prob = self.expected.get_probability(actual_trace)
        assert np.isclose(expected_prob, actual_prob, rtol=self.RTOL)
