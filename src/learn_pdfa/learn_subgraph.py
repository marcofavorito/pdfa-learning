"""Implement the Algorithm 1 of (Palmer and Goldberg 2007) to learn subgraph."""
import pprint
from collections import Counter
from math import ceil, log, log2
from typing import Dict, Optional, Sequence, Set, Tuple

from src.learn_pdfa import logger
from src.learn_pdfa.common import _Params
from src.pdfa.types import Character, State, Word


def _compute_m0(params: _Params):
    """Compute m0."""
    mu = params.mu
    delta_1 = params.delta_1
    n = params.n
    s = params.alphabet_size
    return ceil((16 / mu) ** 2 * (log2(16 / delta_1 / mu) + log2(n * s) + n * s))


def _compute_N(params: _Params, m0: int):
    """Compute N."""
    eps = params.epsilon
    delta = params.delta_1
    n = params.n
    s = params.alphabet_size

    N1 = 8 * (n ** 2) * (s ** 2) / (eps ** 2) * (log((2 ** (n * s)) * n * s / delta))
    N2 = 4 * m0 * n * s / eps
    N = ceil(max(N1, N2))
    logger.info(f"N1 = {N1}, N2 = {N2}. Chosen: {N}")
    return N


def _l_infty_norm(multiset1: Counter, multiset2: Counter):
    current_max = 0.0
    card1 = sum(multiset1.values())
    card2 = sum(multiset2.values())
    assert card1 > 0, "Cardinality of multiset shouldn't be zero."
    assert card2 > 0, "Cardinality of multiset shouldn't be zero."
    all_strings = set(multiset1).union(multiset2)
    for string in all_strings:
        norm = abs(multiset1[string] / card1 - multiset2[string] / card2)
        current_max = norm if norm > current_max else current_max
    return current_max


def extended_transition_fun(
    transitions: Dict[State, Dict[Character, State]], word: Word
) -> Optional[int]:
    """
    Compute the successor state from a word.

    :param transitions: the transitions indexed by state and character.
    :param word: the word.
    :return:
    """
    current_state: Optional[int] = 0
    for c in word:
        if current_state is None:
            return None
        current_state = transitions.get(current_state, {}).get(c)
    return current_state


def _compute_first_multiset(samples: Sequence[Word]) -> Counter:
    result: Counter = Counter()
    for s in samples:
        # s is always non-empty
        result.update([tuple(s)])
    return result


def _rename_final_state(
    vertices: Set[int],
    transition_dict: Dict[int, Dict[Character, int]],
    final_node: int,
) -> None:
    """
    Rename final state in set of vertices and transition dictionary.

    It does side-effects on the data structures.

    :param vertices: the set of vertices.
    :param transition_dict: the transition dictionary.
    :param final_node: the final node.
    :return: None
    """
    if final_node != len(vertices) - 1:
        node_to_rename = len(vertices) - 1
        new_final_node = len(vertices) - 1
        new_node_name = final_node
        logger.info(
            f"Renaming node {node_to_rename} with {new_node_name} (final node: {new_final_node})."
        )
        node_transitions = transition_dict.pop(node_to_rename)
        transition_dict[new_node_name] = node_transitions
        for _, out_transitions in transition_dict.items():
            for character, next_state in out_transitions.items():
                if next_state == node_to_rename:
                    out_transitions[character] = new_node_name
                elif next_state == final_node:
                    out_transitions[character] = new_final_node


def learn_subgraph(  # noqa: ignore
    params: _Params,
) -> Tuple[Set[int], Dict[int, Dict[Character, int]]]:
    """
    Learn a subgraph of the true PDFA.

    :param params: the parameters of the algorithms.
    :return: the graph
    """
    # unpack parameters
    generator = params.sample_generator
    mu = params.mu

    # initialize variables
    initial_state = 0
    vertices = {initial_state}
    transitions: Dict[int, Dict[Character, int]] = {}
    alphabet = set(range(params.alphabet_size))
    vertex2multiset: Dict[int, Counter] = {}

    m0 = _compute_m0(params)
    N = _compute_N(params, m0)
    logger.info(f"m0 = {m0}")
    logger.info(f"N = {N}")
    m0 = min(m0, params.m0_max_debug if params.m0_max_debug else m0)
    N = min(N, params.n1_max_debug if params.n1_max_debug else N)
    logger.info(f"using m0 = {m0}, N = {N}")

    samples = generator.sample(n=N)
    logger.info("Sampling done.")
    logger.info(f"Number of samples: {len(samples)}.")
    logger.info(f"Avg. length of samples: {sum(map(len, samples))/len(samples)}.")

    # multiset for initial state is the entire sample
    vertex2multiset[initial_state] = _compute_first_multiset(samples)

    done = False
    iteration = 0
    while not done:
        logger.info(f"Iteration {iteration}")

        candidate_nodes_by_transitions: Dict[Tuple[State, Character], int] = {}
        candidate_nodes_to_transitions: Dict[int, Tuple[State, Character]] = {}
        multisets: Dict[int, Counter] = {}

        for v in vertices:
            for c in alphabet:
                if transitions.get(v, {}).get(c) is None:  # if transition undefined
                    transition = (v, c)
                    new_candidate = len(vertices) + len(candidate_nodes_to_transitions)
                    candidate_nodes_to_transitions[new_candidate] = transition
                    candidate_nodes_by_transitions[transition] = new_candidate
                    multisets[new_candidate] = Counter()

        for s in samples:
            # s is always non-empty
            for i in range(len(s)):
                r, sigma, t = s[:i], s[i], s[i + 1 :]
                q = extended_transition_fun(transitions, r)
                if q is None:
                    continue
                transition = (q, sigma)
                if transition in candidate_nodes_by_transitions:
                    candidate_node = candidate_nodes_by_transitions[transition]
                    multisets[candidate_node].update([tuple(t)])

        chosen_candidate_node, biggest_multiset = max(
            multisets.items(), key=lambda x: sum(x[1].values())
        )
        cardinality = sum(biggest_multiset.values())
        if cardinality >= m0:
            # check if there is a similar vertex
            similar_vertex: Optional[int] = None
            for v in vertices:
                vertex_multiset = vertex2multiset[v]
                norm = _l_infty_norm(biggest_multiset, vertex_multiset)
                if norm <= mu / 2.0:
                    similar_vertex = v
                    break

            if similar_vertex is not None:
                transition = candidate_nodes_to_transitions[chosen_candidate_node]
                u, sigma = transition
                transitions.setdefault(u, {})[sigma] = similar_vertex
            else:
                new_node = len(vertices)
                vertices.add(new_node)
                vertex2multiset[new_node] = biggest_multiset
                transition = candidate_nodes_to_transitions.pop(chosen_candidate_node)
                _tmp = candidate_nodes_by_transitions.pop(transition)
                assert chosen_candidate_node == _tmp
                u, sigma = transition
                transitions.setdefault(u, {})[sigma] = new_node

        if cardinality < m0:
            done = True
        iteration += 1

    logger.info(f"Vertices: {pprint.pformat(vertices)}")
    logger.info(f"Transitions: {pprint.pformat(transitions)}")
    no_out_transitions = vertices.difference(set(transitions.keys()))
    # the final node is the one without outgoing transitions.
    # renumber the vertices and the transition dictionary accordingly.
    assert (
        len(no_out_transitions) == 1
    ), f"Cannot determine which is the final state. The set of candidates is: {no_out_transitions}."
    final_node = list(no_out_transitions)[0]
    logger.info(f"Computed final node: {final_node} (no outgoing transitions)")
    _rename_final_state(vertices, transitions, final_node)
    logger.info(f"Renamed vertices: {pprint.pformat(vertices)}")
    logger.info(f"Renamed transitions: {pprint.pformat(transitions)}")

    return vertices, transitions
