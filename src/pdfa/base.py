"""Base module of the PDFA package."""

from dataclasses import dataclass
from typing import AbstractSet, Collection, Set, Tuple

import numpy as np

from src.helpers.base import assert_
from src.pdfa.helpers import (
    _check_ergodicity,
    _check_is_legal_character,
    _check_is_legal_state,
    _check_is_legal_word,
    _check_transitions_are_legal,
)
from src.pdfa.types import Character, State, TransitionFunctionDict, Word


@dataclass(frozen=True)
class PDFA:
    """
    Probabilistic Deterministic Finite Automaton.

    - The set of states is the set of integers {0, ..., nb_states - 1} (nb_states > 0)
    - The alphabet is the set of integers {0, ..., alphabet_size - 1}
    - The initial state is always 0
    - The final state is always "nb_states"
    - The transition function is a nested dictionary:
        - at the first level, we have states as keys and the dict of outgoing transition_dict as values
        - a dict of outgoing transition_dict has characters as keys and a tuple of next state and probability
          as value.

    At initialization times, checks on the consistency of the transition dictionary are done.
    """

    nb_states: int
    alphabet_size: int
    transition_dict: TransitionFunctionDict

    def __post_init__(self):
        """Post-initialization checks."""
        assert_(self.nb_states > 0, "Number of states must be greater than zero.")
        assert_(self.alphabet_size > 0, "Alphabet size must be greater than zero.")
        _check_transitions_are_legal(
            self.transition_dict, self.nb_states, self.alphabet_size
        )
        _check_ergodicity(self.transition_dict, self.nb_states, self.final_state)

    def get_successor(self, state: State, character: Character) -> State:
        """
        Get the successor state.

        :param state: the starting state.
        :param character: the read symbol.
        :return: the new state.
        """
        _check_is_legal_state(state, self.nb_states)
        _check_is_legal_character(character, self.alphabet_size)
        next_transitions = self.transition_dict.get(state, {})
        assert_(
            character in next_transitions,
            f"Cannot read character {state} from state {character}.",
        )
        next_state, _probability = next_transitions[character]
        return next_state

    def get_successors(self, state: State) -> AbstractSet[State]:
        """Get the successors."""
        _check_is_legal_state(state, self.nb_states)
        return {
            successor
            for _character, (successor, _probability) in self.transition_dict[
                state
            ].items()
        }

    def get_next_transitions(
        self, state: State
    ) -> Collection[Tuple[Character, float, State]]:
        """Get next transitions from a state."""
        _check_is_legal_state(state, self.nb_states)
        return {
            (character, probability, successor)
            for character, (successor, probability) in self.transition_dict[
                state
            ].items()
        }

    @property
    def initial_state(self):
        """Get the initial state."""
        return 0

    @property
    def final_state(self) -> State:
        """Get the final state."""
        return self.nb_states

    @property
    def states(self) -> Set[State]:
        """Get the set of states."""
        return set(range(self.nb_states))

    @property
    def transitions(self) -> Collection[Tuple[State, Character, float, State]]:
        """Get the transitions."""
        return {
            (start, char, prob, end)
            for start, out_transitions in self.transition_dict.items()
            for char, (end, prob) in out_transitions.items()
        }

    def get_probability(self, word: Word):
        """Get the probability of a word."""
        if len(word) == 0:
            return 0.0

        _check_is_legal_word(word, self.alphabet_size)
        result = 1.0
        current_state = self.initial_state
        for character in word:
            if current_state is None or current_state == self.final_state:
                result = 0.0
                break
            next_state, probability = self.transition_dict.get(current_state, {}).get(
                character, (None, 0.0)
            )
            current_state = next_state
            result *= probability
        return 0.0 if current_state != self.final_state else result

    def sample(self) -> Word:
        """Sample a word."""
        current_state = self.initial_state
        word = []
        while current_state != self.final_state:
            characters, probabilities, next_states = zip(
                *self.get_next_transitions(current_state)
            )
            index = np.random.choice(range(len(characters)), p=probabilities)
            next_character = characters[index]
            current_state = next_states[index]
            word.append(next_character)
        return word
