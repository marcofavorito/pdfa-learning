"""Main test module."""
from pathlib import Path

import numpy as np
import pytest

from src.pdfa import PDFA
from src.pdfa.render import to_graphviz
from tests.conftest import tempdir


def test_pdfa_example():
    """Test the PDFA class with a simple instantiation."""
    automaton = PDFA(
        2, 2, {0: {0: (2, 0.1), 1: (1, 0.9)}, 1: {0: (1, 0.1), 1: (2, 0.9)}}
    )

    with tempdir() as tmp:
        to_graphviz(automaton).render(Path(tmp, "output"))


def test_nb_states_zero():
    """Test empty set of states."""
    with pytest.raises(
        AssertionError, match="Number of states must be greater than zero."
    ):
        PDFA(0, 0, {})


def test_alphabet_size_zero():
    """Test empty alphabet.."""
    with pytest.raises(
        AssertionError, match="Alphabet size must be greater than zero."
    ):
        PDFA(1, 0, {})


def test_not_a_probability():
    """Test not a probability."""
    with pytest.raises(AssertionError, match="'42.0' is not a probability."):
        PDFA(1, 1, {0: {0: (1, 42.0)}})


def test_sum_outgoing_transitions_probabilities_greater_than_one():
    """Test the case when the sum of out transition probabilities is greater than one."""
    with pytest.raises(
        AssertionError, match="Outgoing probability from state 0 do not sum to 1."
    ):
        PDFA(
            1,
            2,
            {
                0: {
                    0: (0, 0.999),
                    1: (1, 0.999),
                }
            },
        )


def test_wrong_state():
    """Test the case when some destination state is wrong."""
    with pytest.raises(
        AssertionError,
        match="Provided state is not in the set of states, nor is a final state.",
    ):
        PDFA(
            1,
            2,
            {
                0: {
                    0: (0, 0.5),
                    1: (42, 0.5),
                }
            },
        )


def test_wrong_character():
    """Test the case when some character is wrong."""
    with pytest.raises(
        AssertionError, match="Provided character is not in the alphabet."
    ):
        PDFA(
            1,
            2,
            {
                0: {
                    42: (0, 0.5),
                    1: (1, 0.5),
                }
            },
        )


def test_successor():
    """Test the 'get_successor' method."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (1, 0.5),
            }
        },
    )
    assert automaton.get_successor(0, 0) == 0
    assert automaton.get_successor(0, 1) == 1


def test_successors():
    """Test the 'get_successors' method."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (1, 0.5),
            }
        },
    )
    assert automaton.get_successors(0) == {0, 1}


def test_transitions():
    """Test the transitions attribute."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (1, 0.5),
            }
        },
    )

    expected_transitions = {
        (0, 0, 0.5, 0),
        (0, 1, 0.5, 1),
    }
    actual_transitions = automaton.transitions
    assert expected_transitions == actual_transitions


def test_probability():
    """Test the probability of a string."""
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, 0.5),
                1: (1, 0.5),
            }
        },
    )

    # final state never reached - probability is zero.
    assert automaton.get_probability([]) == 0.0
    assert automaton.get_probability([0]) == 0.0
    assert automaton.get_probability([0, 0]) == 0.0

    # final state reached
    assert automaton.get_probability([1]) == 0.5
    assert automaton.get_probability([0, 1]) == 0.25
    assert automaton.get_probability([0, 0, 1]) == 0.125

    # read more symbols from final state gives probability zero.
    assert automaton.get_probability([1, 0]) == 0.0
    assert automaton.get_probability([0, 1, 0]) == 0.0
    assert automaton.get_probability([0, 0, 0]) == 0.0


def test_sample():
    """Test the sample method."""
    p = 0.5
    automaton = PDFA(
        1,
        2,
        {
            0: {
                0: (0, p),
                1: (1, 1 - p),
            }
        },
    )

    nb_samples = 5000
    samples = [automaton.sample() for _ in range(nb_samples)]
    expected_average_length = 2
    actual_average_length = np.mean([len(w) for w in samples])
    assert np.isclose(expected_average_length, actual_average_length, rtol=0.05)
