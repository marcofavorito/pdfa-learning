"""Helpers module of the PDFA package."""
from typing import Set

from src.helpers.base import assert_
from src.pdfa.types import Character, State, TransitionFunctionDict, Word

ROUND_PRECISION = 10


def _check_transitions_are_legal(
    transitions: TransitionFunctionDict, nb_states: int, alphabet_size: int
):
    """Check states and characters are legal."""
    for state, char2state in transitions.items():
        _check_is_legal_state(state, nb_states)
        sum_outgoing_probabilities = 0.0
        for character, (next_state, probability) in char2state.items():
            _check_is_legal_character(character, alphabet_size)
            assert_(0.0 <= probability <= 1.0, f"'{probability}' is not a probability.")
            sum_outgoing_probabilities += probability
            _check_is_legal_character(character, alphabet_size)
            _check_is_legal_state_or_final(next_state, nb_states)
        rounded_sum = round(sum_outgoing_probabilities, ROUND_PRECISION)
        assert_(
            rounded_sum == 1.0,
            f"Outgoing probability from state {state} do not sum to 1: {rounded_sum}",
        )


def _check_ergodicity(
    transitions: TransitionFunctionDict, nb_states: int, final_state: int
):
    """Check ergodicity of a transition function."""
    # reachability
    current: Set[State] = set()
    next_ = {final_state}
    while current != next_:
        current = next_

        for start, out_transitions in transitions.items():
            for _char, (end, probability) in out_transitions.items():
                if end in current and probability > 0.0:
                    next_.add(start)

    nonreachability_set = set(range(nb_states)).difference(current)
    assert_(
        len(nonreachability_set) == 0,
        f"The following states cannot reach the final state: {nonreachability_set}",
    )


def _check_is_legal_state(state: State, nb_states: int) -> None:
    """Check that a state is legal."""
    assert_(0 <= state < nb_states, "Provided state is not in the set of states.")


def _check_is_legal_state_or_final(state: State, nb_states: int) -> None:
    """Check that a state is legal, including final states."""
    assert_(
        0 <= state <= nb_states,
        "Provided state is not in the set of states, nor is a final state.",
    )


def _check_is_legal_character(character: Character, alphabet_size) -> None:
    """Check that a character is in the alphabet."""
    assert_(
        0 <= character < alphabet_size, "Provided character is not in the alphabet."
    )


def _check_is_legal_word(w: Word, alphabet_size) -> None:
    """Check that a word is consistent with the alphabet."""
    assert_(
        all(0 <= c < alphabet_size for c in w), "Provided word is not in the alphabet."
    )
