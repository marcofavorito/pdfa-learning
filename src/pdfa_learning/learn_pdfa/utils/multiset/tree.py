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
"""Interface and implementation of a multiset."""
import itertools
from collections import deque
from dataclasses import dataclass
from typing import Collection, Deque, Dict, Iterator, List, Optional, Set, Tuple

import graphviz

from pdfa_learning.learn_pdfa.utils.multiset.base import Multiset
from pdfa_learning.types import Character, Word


@dataclass
class _TreeMetadata:
    """Keep tree data."""

    size: int = 0
    alphabet_size: int = 0

    def __hash__(self):
        return id(self)


class Node:
    """A node in the prefix-tree."""

    __slots__ = [
        "_index",
        "_tree_metadata",
        "_parent",
        "_symbol",
        "_symbol2child",
        "counts",
        "children_counts",
    ]

    def __init__(self, parent: Optional["Node"], symbol: Optional[int] = None):
        """Initialize the prefix-tree node."""
        self._parent = parent
        self._symbol = symbol
        self.counts = 0
        self.children_counts = 0
        self._symbol2child: Dict[int, Node] = {}
        if parent is not None:
            assert symbol is not None
            self._tree_metadata: _TreeMetadata = parent._tree_metadata
            self._index = self._tree_metadata.size
            parent.add_child(symbol, self)
        else:
            self._tree_metadata = _TreeMetadata(size=1)
            self._index = 0

    def add_child(self, symbol: int, node: "Node") -> None:
        """Add child."""
        assert symbol not in self._symbol2child
        self._tree_metadata.size += 1
        self._symbol2child[symbol] = node

    @property
    def index(self) -> int:
        """Get the index of the node."""
        return self._index

    def add(self, trace: Word, times: int = 1) -> None:
        """Add a trace to the prefix tree."""
        current_node: Node = self
        current_node.children_counts += times
        for character in trace:
            next_node = current_node._symbol2child.get(character, None)
            if next_node is None:
                # create a new node.
                next_node = Node(current_node, character)
            current_node = next_node
            current_node.children_counts += times
        current_node.counts += times

    def get_end_node(self, trace: Word) -> Optional["Node"]:
        """Get the finale node (after processing the entire trace)."""
        result: Optional[Node] = self
        for character in trace:
            if result is not None:
                result = result._symbol2child.get(character, None)
            if result is None:
                break
        return result

    def next_nodes(self) -> Set["Node"]:
        """Get the next nodes."""
        return set(self._symbol2child.values())

    def next_transitions(self) -> Collection[Tuple[Character, "Node"]]:
        """Get the next transitions."""
        return list(self._symbol2child.items())

    def traces(self) -> Set[Word]:
        """Get all traces from this node."""
        result: Set[Word] = set()

        prefix = (self._symbol,) if self._symbol is not None else ()
        if self.counts > 0:
            result.add(prefix)

        next_nodes = self.next_nodes()
        for node in next_nodes:
            next_traces = node.traces()
            if len(next_traces) > 0:
                new_traces = set(map(lambda x: prefix + tuple(x), next_traces))
                result = result.union(new_traces)
        return result

    def items(self) -> Iterator[Tuple[Word, int]]:
        """Get list of pairs, trace and its count."""
        prefix = (self._symbol,) if self._symbol is not None else ()
        if self.counts > 0:
            yield prefix, self.counts

        next_nodes = self.next_nodes()
        for node in next_nodes:
            for next_trace, next_count in node.items():
                new_trace = prefix + tuple(next_trace)
                yield new_trace, next_count

    def get_counts(self, t: Word) -> int:
        """Get the counts of a trace."""
        if len(t) == 0:
            return self.counts
        for index, character in enumerate(t):
            next_node = self._symbol2child.get(character, None)
            if next_node is not None:
                # found next node.
                return next_node.get_counts(t[index + 1 :])
        # no next node found => trace is not in the multiset.
        return 0

    def __eq__(self, other: object) -> bool:
        """Check equality."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.index == other.index

    def __hash__(self) -> int:
        """Get hash."""
        return hash((Node, self.index, self._tree_metadata))


class PrefixTreeMultiset(Multiset):
    """A multi-set based on a prefix tree."""

    def __init__(self, node: Optional[Node] = None):
        """
        Initialize a Multiset prefix-tree based.

        :param node: the node of the tree from where to start.
        """
        self._node = node if node is not None else Node(parent=None)

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return self._node.get_counts(trace)

    def add(self, t: Word, times: int = 1) -> None:
        """Add an element."""
        self._node.add(t, times=times)

    @property
    def size(self) -> int:
        """Get the size."""
        return self._node.children_counts

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        if self._node.children_counts == 0:
            return 0.0
        return self._node.get_counts(t) / self.size

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        if self._node.children_counts == 0:
            return 0.0
        final_node: Optional[Node] = self._node.get_end_node(t)
        if final_node is None:
            # never seen this prefix.
            return 0.0
        return final_node.children_counts / self.size

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return self._node.traces()

    def items(self) -> Iterator[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return self._node.items()

    def get_successors(self) -> Dict[Character, "ReadOnlyPrefixTreeMultiset"]:
        """Get successors."""
        successors: Dict[Character, Set[Node]] = {}
        for next_char, next_node in self._node.next_transitions():
            successors.setdefault(next_char, set()).add(next_node)

        result: Dict[Character, ReadOnlyPrefixTreeMultiset] = {
            k: ReadOnlyPrefixTreeMultiset(v) for k, v in successors.items()
        }
        return result


class ReadOnlyPrefixTreeMultiset(Multiset):
    """Readonly multiset."""

    def __init__(self, nodes: Optional[Set[Node]] = None):
        """
        Initialize a Multiset prefix-tree based.

        :param nodes: the nodes of the tree from where to start.
        """
        self._nodes = nodes if nodes is not None else {Node(parent=None)}

    def get_successors(self) -> Dict[Character, "ReadOnlyPrefixTreeMultiset"]:
        """Get successors."""
        successors: Dict[Character, Set[Node]] = {}
        for node in self._nodes:
            for next_char, next_node in node.next_transitions():
                successors.setdefault(next_char, set()).add(next_node)

        result: Dict[Character, ReadOnlyPrefixTreeMultiset] = {
            k: ReadOnlyPrefixTreeMultiset(v) for k, v in successors.items()
        }
        return result

    def get_counts(self, trace: Word) -> int:
        """Get counts."""
        return sum(n.get_counts(trace) for n in self._nodes)

    def add(self, t: Word, times: int = 1) -> None:
        """Add an element."""
        raise ValueError("Read-only.")

    @property
    def size(self) -> int:
        """Get the size."""
        return sum(n.children_counts for n in self._nodes)

    def get_probability(self, t: Word) -> float:
        """Get the probability of a trace."""
        probabilities = [
            n.get_counts(t) / n.children_counts
            for n in self._nodes
            if n.children_counts != 0
        ]
        return sum(probabilities)

    def get_prefix_probability(self, t: Word) -> float:
        """Get the prefix-probability of a trace."""
        final_nodes: List[Optional[Node]] = [n.get_end_node(t) for n in self._nodes]
        return sum(
            final_node.children_counts / self.size
            for final_node in final_nodes
            if final_node is not None and final_node.children_counts > 0
        )

    @property
    def traces(self) -> Set[Word]:
        """Get the set of traces."""
        return set.union(*[n.traces() for n in self._nodes])

    def items(self) -> Iterator[Tuple[Word, int]]:
        """Get the traces and their counts."""
        return itertools.chain.from_iterable([n.items() for n in self._nodes])


def node_to_graphviz(node: Node, max_depth: int = 10) -> graphviz.Digraph:
    """From prefix-tree node to Graphviz."""
    graph = graphviz.Digraph(format="svg")
    graph.graph_attr["rankdir"] = "LR"
    graph.edge("fake", str(node.index), style="bold")

    queue: Deque = deque()
    queue.append((0, node))
    while len(queue) > 0:
        i, current = queue.pop()
        graph.node(
            str(current.index),
            root="true",
            label=f"index={current.index}\ncounts={current.counts}\nchildren={current.children_counts}",
        )

        if i >= max_depth:
            continue

        transitions = current.next_transitions()
        for symbol, next_node in transitions:
            graph.edge(
                str(current.index),
                str(next_node.index),
                label=str(symbol),
            )
            queue.appendleft((i + 1, next_node))

    graph.node(
        str(node.index),
        root="true",
        label=f"index={node.index}\ncounts={node.counts}\nchildren={node.children_counts}",
    )
    return graph
