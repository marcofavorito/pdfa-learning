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
"""Module that implements rendering utilities."""
from typing import Callable, Dict, Set

import graphviz

from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.helpers import (
    PROB_LOWER_BOUND,
    ROUND_PRECISION,
    filter_transition_function,
)
from pdfa_learning.types import Character


def to_graphviz(
    pdfa: PDFA,
    state2str: Callable[[int], str] = lambda x: str(x),
    char2str: Callable[[int], str] = lambda x: str(x),
    round_precision: int = ROUND_PRECISION,
    lower_bound: float = PROB_LOWER_BOUND,
    with_prob: bool = True,
) -> graphviz.Digraph:
    """Transform a PDFA to Graphviz."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")
    graph.attr(rankdir="LR")

    states, filtered_transition_function = filter_transition_function(
        pdfa.transition_dict, lower_bound
    )

    for state in states:
        if state == pdfa.initial_state:
            graph.node(state2str(state), root="true")
        else:
            graph.node(state2str(state))
    graph.node(state2str(pdfa.final_state), shape="doublecircle")

    graph.edge("fake", state2str(pdfa.initial_state), style="bold")

    for start, outgoing in filtered_transition_function.items():
        for char, (end, prob) in outgoing.items():
            new_prob = round(prob, round_precision)
            if new_prob > lower_bound:
                label = f"{char2str(char)}"
                label += f", {new_prob}" if with_prob else ""
                graph.edge(
                    state2str(start),
                    state2str(end),
                    label=label,
                )

    return graph


# TODo refactor
def to_graphviz_from_graph(
    vertices: Set[int],
    transitions: Dict[int, Dict[Character, int]],
    state2str: Callable[[int], str] = lambda x: str(x),
    char2str: Callable[[int], str] = lambda x: str(x),
):
    """To graphviz from graph."""
    graph = graphviz.Digraph(format="svg")
    graph.node("fake", style="invisible")

    for state in vertices:
        if state == 0:
            graph.node(state2str(state), root="true")
        else:
            graph.node(state2str(state))

    graph.edge("fake", state2str(0), style="bold")

    for start, char2end in transitions.items():
        for char, end in char2end.items():
            graph.edge(
                state2str(start),
                state2str(end),
                label=f"{char2str(char)}",
            )

    return graph
