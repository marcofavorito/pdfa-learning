"""Implement the Algorithm 2 of (Palmer and Goldberg 2007) to estimate probabilities."""
import math
import pprint
from collections import Counter
from math import ceil, log
from typing import Dict, Optional, Set, Tuple

from src.learn_pdfa import logger
from src.learn_pdfa.common import _Params
from src.pdfa import PDFA
from src.pdfa.types import TransitionFunctionDict


def _sample_size(params: _Params) -> int:
    eps = params.epsilon
    s = params.alphabet_size
    n = params.n
    delta2 = params.delta_2
    n1 = 2 * n * s / delta2
    n2 = (64 * n * s / eps / delta2) ** 2
    n3 = 32 * n * s / eps
    n4 = log(2 * n * s / delta2)
    N = ceil(n1 * n2 * n3 * n4)
    return N


def learn_probabilities(
    graph: Tuple[Set[int], Dict[int, Dict[int, int]]], params: _Params
) -> PDFA:
    """
    Learn the probabilities of the PDFA.

    :param graph: the learned subgraph of the true PDFA.
    :param params: the parameters of the algorithms.
    :return: the PDFA.
    """
    logger.info("Start learning probabilities.")
    vertices, transitions = graph
    initial_state = 0
    N = _sample_size(params)
    logger.info(f"Sample size: {N}.")
    N = min(N, params.n2_max_debug if params.n2_max_debug else N)
    logger.info(f"Using N = {N}.")
    generator = params.sample_generator
    sample = generator.sample(N)
    n_observations: Counter = Counter()
    for word in sample:
        current_state = initial_state
        for character in word:
            # update statistics

            n_observations.update([(current_state, character)])

            # compute next state
            next_state: Optional[int] = transitions.get(current_state, {}).get(
                character
            )

            if next_state is None:
                break  # pragma: no cover
            current_state = next_state

    gammas: Dict[int, Dict[int, float]] = {}

    # compute number of times q is visited
    q_visits: Counter = Counter()
    for (q, _), counts in n_observations.items():
        q_visits[q] += counts
    # compute mean
    for (q, sigma), counts in n_observations.items():
        gammas.setdefault(q, {})[sigma] = counts / q_visits[q]
    # rescale probabilities
    for _, out_probabilities in gammas.items():
        characters, probabilities = zip(*list(out_probabilities.items()))
        probability_sum = math.fsum(probabilities)
        new_probabilities = [p / probability_sum for p in probabilities]
        out_probabilities.update(dict(zip(characters, new_probabilities)))

    # compute transition function for the PDFA
    transition_dict: TransitionFunctionDict = {}
    for q, out_transitions in transitions.items():
        transition_dict.setdefault(q, {})
        for sigma, q_prime in out_transitions.items():
            prob = gammas.get(q, {}).get(sigma, 0.0)
            transition_dict[q][sigma] = (q_prime, prob)

    logger.info("Removing final state from the set of vertices.")
    vertices.remove(len(vertices) - 1)
    logger.info(f"Computed vertices: {pprint.pformat(vertices)}")
    logger.info(f"Computed transition dictionary: {pprint.pformat(transition_dict)}")

    return PDFA(len(vertices), params.alphabet_size, transition_dict)
