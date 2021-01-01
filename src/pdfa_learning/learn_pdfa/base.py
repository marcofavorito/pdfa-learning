"""Base module for the learn pdfa implementation."""
import pprint

from pdfa_learning.learn_pdfa import learn_probabilities, logger
from pdfa_learning.learn_pdfa.common import _Params
from pdfa_learning.learn_pdfa.learn_subgraph import learn_subgraph


def learn_pdfa(**kwargs):
    """
    PAC-learn a PDFA.

    :param kwargs: the keyword arguments of the algorithm (see the _Params class).
    :return: the learnt PDFA.
    """
    params = _Params(**kwargs)
    logger.info(f"Parameters: {pprint.pformat(str(params))}")
    vertices, transitions = learn_subgraph(params)
    logger.info(f"Number of vertices: {len(vertices)}.")
    logger.info(f"Transitions: {pprint.pformat(transitions)}.")
    pdfa = learn_probabilities((vertices, transitions), params)
    return pdfa
