"""Entrypoint for the algorithm."""

import pprint

from pdfa_learning.learn_pdfa import logger
from pdfa_learning.learn_pdfa.palmer.learn_probabilities import learn_probabilities
from pdfa_learning.learn_pdfa.palmer.learn_subgraph import learn_subgraph
from pdfa_learning.learn_pdfa.palmer.params import PalmerParams


def learn_pdfa(**kwargs):
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm (see the PalmerParams class).
    :return: the learnt PDFA.
    """
    params = PalmerParams(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    vertices, transitions = learn_subgraph(params)
    logger.info(f"Number of vertices: {len(vertices)}.")
    logger.info(f"Transitions: {pprint.pformat(transitions)}.")
    pdfa = learn_probabilities((vertices, transitions), params)
    return pdfa
