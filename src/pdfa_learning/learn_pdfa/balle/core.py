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
"""Entrypoint for the algorithm."""
import pprint
from abc import ABC
from copy import deepcopy
from math import log, sqrt
from typing import Dict, Set, Tuple, Type, cast

from pdfa_learning.helpers.base import normalize
from pdfa_learning.learn_pdfa import logger
from pdfa_learning.learn_pdfa.balle.params import BalleParams
from pdfa_learning.learn_pdfa.utils.base import (
    MultisetLike,
    get_prefix_probability,
    size,
)
from pdfa_learning.learn_pdfa.utils.multiset.tree import (
    Node,
    PrefixTreeMultiset,
    ReadOnlyPrefixTreeMultiset,
)
from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.base import FINAL_STATE, FINAL_SYMBOL
from pdfa_learning.types import Character, State, TransitionFunctionDict

ConcreteMultiset = PrefixTreeMultiset


def learn_pdfa(**kwargs) -> PDFA:
    """
    PAC-learn a PDFA.

    This is a wrapper function to the 'Learner' class, defined below.

    :param kwargs: the keyword arguments of the algorithm (see the BalleParams class).
    :return: the learnt PDFA.
    """
    params = BalleParams(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    automaton = Learner(params).learn()
    return automaton


def _compute_threshold(m_u, m_v, s_u, s_v, delta):
    """Compute distinctness threshold."""
    n1 = 2 / min(m_u, m_v)
    n2 = log(8 * (s_u + s_v) / delta)
    return sqrt(n1 * n2)


class Learner(ABC):
    """Abstract learner of subgraphs."""

    def __init__(self, params: BalleParams):
        """Initialize the learner."""
        self._params = params

    @property
    def params(self) -> BalleParams:
        """Get the parameters."""
        return self._params

    def learn(self) -> PDFA:
        """
        Do the learning.

        This is the main entry-point of the class.
        """
        manager = SampleMultisetManager(self.params, ConcreteMultiset)
        graph = Graph(self.params)
        graph.add_vertex(0, manager.main_multiset)
        candidate_nodes = CandidateNodesCalculator(manager.multiset_cls, manager, graph)
        while not candidate_nodes.do_iteration():
            continue
        return PDFAConstructor(graph, manager).get()


class SampleMultisetManager:
    """Sample multiset manager."""

    def __init__(self, params: BalleParams, multiset_cls: Type[MultisetLike]):
        """Initialize."""
        self.params = params
        self.multiset_cls = multiset_cls
        self.main_multiset = self.multiset_cls()
        self._sample_and_update()

    def _sample_and_update(self):
        """Do the sampling."""
        logger.info("Generating the sample.")
        if self.params.sample_generator:
            generator = self.params.sample_generator
            samples = generator.sample(n=self.params.nb_samples)
            samples = list(map(lambda x: tuple(x), samples))
        else:
            samples = self.params.dataset
        self.average_trace_length = sum(map(len, samples)) / len(samples)
        logger.info(f"Average trace length: {self.average_trace_length}.")
        logger.info("Populate root multiset.")
        self.main_multiset.update(samples)


class Graph:
    """Represent a PDFA subgraph."""

    def __init__(self, params: BalleParams):
        """Initialize."""
        self.params = params
        self.initial_state = 0
        self.vertices = {self.initial_state}
        self.transitions: Dict[int, Dict[Character, int]] = {}
        self.alphabet = set(range(self.params.alphabet_size))

        self.vertex2multiset: Dict[int, MultisetLike] = {}

    def add_vertex(self, new_vertex, multiset):
        """Add a vertex to the multiset manager."""
        self.vertex2multiset[new_vertex] = multiset


class PDFAConstructor:
    """Construct the PDFA."""

    def __init__(self, graph: Graph, sample: SampleMultisetManager):
        """
        Initialize PDFA constructor.

        :param graph: the graph object.
        :param sample: the sample manager.
        """
        self.graph = graph
        self.sample = sample

        self.params = self.sample.params

    def get(self) -> PDFA:
        """Build the PDFA."""
        new_transitions: Dict[int, Dict[Character, int]] = deepcopy(
            self.graph.transitions
        )
        new_vertices: Set[int] = deepcopy(self.graph.vertices)
        self._complete_graph(new_vertices, new_transitions)
        pdfa_transitions = self._compute_probabilities(new_transitions)
        return PDFA(len(new_vertices), len(self.graph.alphabet), pdfa_transitions)

    def _add_ground_node(
        self, vertices: Set[int], transitions: Dict[int, Dict[Character, int]]
    ):
        """Add a ground node."""
        ground_node = len(vertices)
        ground_node_used = False

        for vertex in vertices:
            transitions_from_vertex = transitions.get(vertex, {})
            for character in self.graph.alphabet:
                if character not in transitions_from_vertex:
                    ground_node_used = True
                    transitions_from_vertex[character] = ground_node
            transitions[vertex] = transitions_from_vertex

        if ground_node_used:
            vertices.add(ground_node)
            transitions[ground_node] = {}
            for character in self.graph.alphabet:
                transitions[ground_node][character] = ground_node

    def _complete_graph(
        self, vertices: Set[int], transitions: Dict[int, Dict[Character, int]]
    ):
        """
        Complete graph.

        Add a ground node (only if needed, and if allowed by params), and a final node.
        """
        if self.params.with_ground:
            self._add_ground_node(vertices, transitions)

        final_node = FINAL_STATE
        for vertex in vertices:
            transitions.setdefault(vertex, {})[FINAL_SYMBOL] = final_node

    def _compute_edge_probability(self, state: int, character: int):
        """Given state and character, compute probability."""
        multiset = self.graph.vertex2multiset.get(state, ConcreteMultiset())  # type: ignore
        size = sum(multiset.values())
        smoothing_probability = (
            self.params.get_gamma_min(self.sample.average_trace_length)
            if self.params.with_smoothing
            else 0.0
        )
        if size == 0:
            return self.params.get_gamma_min(self.sample.average_trace_length)
        char_prob = get_prefix_probability(multiset, (character,))
        factor = 1 - (self.params.alphabet_size + 1) * smoothing_probability
        return char_prob * factor + smoothing_probability

    def _compute_probabilities(self, transitions: Dict[int, Dict[Character, int]]):
        """Given vertices, transitions and its multisets, estimate edge probabilities."""
        pdfa_transitions: TransitionFunctionDict = {}

        # compute gammas
        for start, out_transitions in transitions.items():
            pdfa_transitions[start] = {}
            for character, next_state in out_transitions.items():
                probability = self._compute_edge_probability(start, character)
                pdfa_transitions[start][character] = (next_state, probability)

        # normalize
        pdfa_transitions = normalize(pdfa_transitions)
        return pdfa_transitions


class CandidateNodesCalculator:
    """Compute candidate nodes."""

    def __init__(
        self,
        multiset_cls: Type[MultisetLike],
        multiset_mgr: SampleMultisetManager,
        graph: Graph,
    ):
        """
        Initialize the candidate node calculator.

        :param multiset_cls: the multiset class.
        :param multiset_mgr: the multiset to use.
        :param graph: the graph object.
        """
        self.multiset_cls = multiset_cls
        self.multiset_mgr = multiset_mgr
        self.params = multiset_mgr.params
        self.graph = graph

        self.iteration = 0
        self.iteration_upper_bound = self.params.n * self.params.alphabet_size
        self.candidate_nodes_by_transitions: Dict[Tuple[State, Character], int] = {}
        self.candidate_nodes_to_transitions: Dict[int, Tuple[State, Character]] = {}
        self.multisets: Dict[int, MultisetLike] = {}

    def do_iteration(self) -> bool:
        """
        Do one iteration.

        :return: False if the current iteration failed, else True.
        """
        logger.info(f"Iteration {self.iteration}")
        done = self._reset_and_check_if_done()
        if done:
            return True

        (
            chosen_candidate_node,
            biggest_multiset,
        ) = self.compute_multisets_and_get_biggest()
        if sum(biggest_multiset.values()) == 0:
            logger.info("Biggest multiset has cardinality 0, done")
            return True

        non_distinct_vertices = self._compute_non_distinct_vertices(
            chosen_candidate_node
        )
        self._add_new_state_or_edge(chosen_candidate_node, non_distinct_vertices)
        # end of iteration
        self.iteration += 1
        done = self.iteration >= self.iteration_upper_bound
        return done

    def _add_new_state_or_edge(self, candidate_node, non_distinct_vertices):
        (
            start_state,
            character,
        ) = self.candidate_nodes_to_transitions[candidate_node]
        biggest_multiset = self.multisets[candidate_node]
        all_nodes_are_distinct = len(non_distinct_vertices) == 0
        maximum_nb_states_reached = (
            len(self.graph.vertices) == self.multiset_mgr.params.n
        )
        if all_nodes_are_distinct and not maximum_nb_states_reached:
            # we've got a new node
            new_vertex = len(self.graph.vertices)
            self.graph.vertices.add(new_vertex)
            self.graph.add_vertex(new_vertex, biggest_multiset)
            self.graph.transitions.setdefault(start_state, {})[character] = new_vertex
        else:
            # pick a safe node that has not distinguished from best candidate.
            # For deterministic behaviour, pick the smallest
            sorted_non_distinct_vertices = sorted(non_distinct_vertices)
            if len(sorted_non_distinct_vertices) > 1:
                logger.warning(
                    f"More than one non-distinct vertex: {sorted_non_distinct_vertices}"
                )
                logger.warning(
                    f"Distances and thresholds: {pprint.pformat(non_distinct_vertices)}"
                )
            old_vertex = sorted_non_distinct_vertices[0]
            self.graph.transitions.setdefault(start_state, {})[character] = old_vertex

    def _reset_and_check_if_done(self) -> bool:
        """Reset the state, and check if there are candidate nodes."""
        self._reset_candidate_nodes(
            self.graph.vertices, self.graph.alphabet, self.graph.transitions
        )
        return len(self.candidate_nodes_to_transitions) == 0

    def _reset_candidate_nodes(self, vertices, alphabet, transitions: Dict):
        """Reset multisets."""
        self.candidate_nodes_by_transitions = {}
        self.candidate_nodes_to_transitions = {}
        self.multisets = {}
        self._recompute_candidate_nodes(vertices, alphabet, transitions)

    def _recompute_candidate_nodes(self, vertices, alphabet, transitions):
        """Recompute candidate nodes."""
        for v in vertices:
            for c in alphabet:
                if transitions.get(v, {}).get(c) is None:  # if transition undefined
                    transition = (v, c)
                    new_candidate = len(vertices) + len(
                        self.candidate_nodes_to_transitions
                    )
                    self.candidate_nodes_to_transitions[new_candidate] = transition
                    self.candidate_nodes_by_transitions[transition] = new_candidate

    def compute_multisets_and_get_biggest(self) -> Tuple[int, MultisetLike]:
        """Compute multisets for the current iteration, and get biggest multiset."""
        main_multiset = cast(PrefixTreeMultiset, self.multiset_mgr.main_multiset)
        transitions = self.graph.transitions
        root = main_multiset._node
        pdfa_initial_state = 0

        def _visit(node: Node, state: State):
            outgoing_from_last: Dict[Character, int] = transitions.get(state, {})
            next_transitions = node.next_transitions()
            for c, n in next_transitions:
                transition = (state, c)
                if transition in self.candidate_nodes_by_transitions:
                    candidate_node = self.candidate_nodes_by_transitions[transition]
                    multiset = self.multisets.get(candidate_node)
                    if multiset is None:
                        self.multisets[candidate_node] = ReadOnlyPrefixTreeMultiset({n})
                    else:
                        cast(ReadOnlyPrefixTreeMultiset, multiset)._nodes.add(n)
                elif c != FINAL_SYMBOL:
                    _visit(n, outgoing_from_last[c])

        _visit(root, pdfa_initial_state)
        return self._get_biggest_multiset()

    def _get_biggest_multiset(self) -> Tuple[int, MultisetLike]:
        """Compute the biggest multiset."""
        return max(
            self.multisets.items(),
            key=lambda x: sum(x[1].values()),
            default=(-1, self.multiset_cls()),
        )

    def _compute_non_distinct_vertices(self, chosen_candidate_node):
        non_distinct_vertices: Dict[int, float] = {}
        for v in self.graph.vertices:
            # TODO remove
            """
            m1, m2 = self.multisets[chosen_candidate_node], self.graph.vertex2multiset[v]
            distance, threshold = test_distinct(m1, m2, self.multiset_mgr.params)
            is_distinct = distance > threshold
            if not is_distinct:
            """
            is_distinct = self.test_distinct(chosen_candidate_node, v)
            if not is_distinct:
                # TODO sort by distance/threshold
                non_distinct_vertices[v] = 0.0
        return non_distinct_vertices

    def test_distinct(self, chosen_candidate_node: int, v: int):
        """Test distinctness of two vertices."""
        multiset_candidate = self.multisets[chosen_candidate_node]
        multiset_safe = self.graph.vertex2multiset[v]
        prefixes_candidate = sum(
            [(len(trace) + 1) * count for trace, count in multiset_candidate.items()]
        )
        prefixes_safe = sum(
            [(len(trace) + 1) * count for trace, count in multiset_safe.items()]
        )
        threshold = _compute_threshold(
            size(multiset_candidate),
            size(multiset_safe),
            prefixes_candidate,
            prefixes_safe,
            self.params.delta_0,
        )

        return self.test_similar(multiset_candidate, 1.0, multiset_safe, 1.0, threshold)  # type: ignore

    def test_similar(
        self, m1: PrefixTreeMultiset, p1, m2: PrefixTreeMultiset, p2, t
    ) -> bool:
        """
        Test that two multisets are similar.

        :param m1: the first multiset.
        :param p1: the probability score so far.
        :param m2: the second multiset.
        :param p2: the probability score so far.
        :param t: the threshold.
        :return: True if distinct, False otherwise.
        """
        if abs(p1 - p2) > t:
            return True

        if m1 is None or m2 is None:
            return False

        m1_succ = m1.get_successors()
        m2_succ = m2.get_successors()

        next_chars = set(m1_succ.keys()).union(set(m2_succ.keys()))

        for next_char in next_chars:
            next_p1 = p1 * m1.get_prefix_probability([next_char])
            next_p2 = p2 * m2.get_prefix_probability([next_char])
            next_m1 = m1_succ.get(next_char, None)
            next_m2 = m2_succ.get(next_char, None)
            result = self.test_similar(next_m1, next_p1, next_m2, next_p2, t)  # type: ignore
            if result is True:
                return True
        return False
