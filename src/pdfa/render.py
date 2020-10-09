"""Module that implements rendering utilities."""
from typing import Callable

import graphviz

from src.pdfa import PDFA
from src.pdfa.helpers import ROUND_PRECISION


def to_graphviz(
    pdfa: PDFA,
    state2str: Callable[[int], str] = lambda x: str(x),
    char2str: Callable[[int], str] = lambda x: str(x),
) -> graphviz.Digraph:
    """Transform a PDFA to Graphviz."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")

    for state in pdfa.states:
        if state == pdfa.initial_state:
            graph.node(state2str(state), root="true")
        else:
            graph.node(state2str(state))
    graph.node(state2str(pdfa.final_state), shape="doublecircle")

    graph.edge("fake", state2str(pdfa.initial_state), style="bold")

    for (start, char, prob, end) in pdfa.transitions:
        graph.edge(
            state2str(start),
            state2str(end),
            label=f"{char2str(char)}, {round(prob, ROUND_PRECISION)}",
        )

    return graph
