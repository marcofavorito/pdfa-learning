"""Definition of PDFAs."""

from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.base import FINAL_STATE
from pdfa_learning.pdfa.helpers import FINAL_SYMBOL


def make_pdfa_one_state(p: float = 0.3):
    """Make a PDFA with one state, for testing purposes."""
    automaton = PDFA(
        2,
        2,
        {
            0: {
                0: (0, p),
                1: (1, 1 - p),
            },
            1: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )
    return automaton


def make_pdfa_two_state(p1: float = 0.4, p2: float = 0.7):
    """Make a PDFA with two states, for testing purposes."""
    automaton = PDFA(
        3,
        2,
        {
            0: {
                0: (1, p1),
                1: (2, 1 - p1),
            },
            1: {
                0: (2, 1 - p2),
                1: (1, p2),
            },
            2: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )
    return automaton


def make_pdfa_sequence_three_states(
    p1: float, p2: float, p3: float, stop_probability: float
):
    """Make a PDFA with three states, for testing purposes."""
    automaton = PDFA(
        4,
        3,
        {
            0: {
                0: (1, p1),
                1: (0, p2),
                2: (0, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            1: {
                0: (0, p1),
                1: (2, p2),
                2: (0, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            2: {
                0: (0, p1),
                1: (0, p2),
                2: (3, p3),
                FINAL_SYMBOL: (FINAL_STATE, stop_probability),
            },
            3: {
                FINAL_SYMBOL: (FINAL_STATE, 1.0),
            },
        },
    )
    return automaton


def make_reber_grammar() -> PDFA:
    """
    Make PDFA for Reber grammar [1].

    Order of characters: "BTPSXV"

    - [1] R. C. Carrasco and J. Oncina. Learning deterministic regular
          grammars from stochastic samples in polynomial time.
          ITA, 33(1):1–20, 1999
    """
    nb_states = 7
    alphabet_size = 6

    automaton = PDFA(
        nb_states,
        alphabet_size,
        {
            0: {0: (1, 1.0)},
            1: {1: (2, 0.5), 2: (3, 0.5)},
            2: {3: (2, 0.6), 4: (4, 0.4)},
            3: {1: (3, 0.7), 5: (5, 0.3)},
            4: {4: (3, 0.5), 3: (6, 0.5)},
            5: {2: (4, 0.5), 5: (6, 0.5)},
            6: {FINAL_SYMBOL: (FINAL_STATE, 1.0)},
        },
    )
    return automaton
