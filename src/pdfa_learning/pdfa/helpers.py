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
"""Helpers module of the PDFA package."""
from collections import deque
from copy import copy
from typing import Deque, Set, Tuple

from pdfa_learning.helpers.base import assert_
from pdfa_learning.types import Character, State, TransitionFunctionDict, Word

FINAL_STATE = -1
FINAL_SYMBOL = -1

ROUND_PRECISION = 5
PROB_LOWER_BOUND = 0.01


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
            _check_final_symbol_and_final_state(character, next_state)
        rounded_sum = round(sum_outgoing_probabilities, ROUND_PRECISION)
        assert_(
            rounded_sum == 1.0,
            f"Outgoing probability from state {state} do not sum to 1: {rounded_sum}",
        )


def _check_final_symbol_and_final_state(character: Character, next_state: State):
    """Check that all and only the transitions with final symbol ends to the final state."""
    is_final_symbol = character == FINAL_SYMBOL
    is_final_state = next_state == FINAL_STATE
    final_symbol_implies_final_state = not is_final_symbol or is_final_state
    final_state_implies_final_symbol = not is_final_state or is_final_symbol
    assert (
        final_state_implies_final_symbol
    ), "Only transitions with final symbol can go to final state."
    assert (
        final_symbol_implies_final_state
    ), "All transitions with final symbol must go to final state."


def _check_ergodicity(
    transitions: TransitionFunctionDict, nb_states: int, final_state: int
):
    """Check ergodicity of a transition function."""
    # reachability
    current: Set[State] = set()
    next_ = {final_state}
    while current != next_:
        current = copy(next_)
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
        0 <= state < nb_states or state == FINAL_STATE,
        "Provided state is not in the set of states, nor is a final state.",
    )


def _check_is_legal_character(character: Character, alphabet_size) -> None:
    """Check that a character is in the alphabet."""
    assert_(
        FINAL_SYMBOL <= character < alphabet_size,
        "Provided character is not in the alphabet.",
    )


def _check_is_legal_word(w: Word, alphabet_size) -> None:
    """Check that a word is consistent with the alphabet."""
    assert_(
        all(0 <= c < alphabet_size for c in w[:-1]),
        "Provided word is not in the alphabet.",
    )


def filter_transition_function(
    transition_function: TransitionFunctionDict, lower_bound: float
) -> Tuple[Set[State], TransitionFunctionDict]:
    """Filter the transition function by removing low frequency transitions."""
    visited = set()
    new_function: TransitionFunctionDict = {}

    queue: Deque = deque()
    queue.append(0)
    while len(queue) > 0:
        current = queue.pop()
        visited.add(current)
        next_transitions = transition_function.get(current, {})
        for char, (next_, prob) in next_transitions.items():
            if prob > lower_bound:
                new_function.setdefault(current, {})[char] = (next_, prob)
                if next_ not in visited:
                    queue.appendleft(next_)

    return visited, new_function
