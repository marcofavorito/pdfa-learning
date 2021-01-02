"""Tests for the prefix-tree based multiset implementation."""
import pytest
from hypothesis import given, settings, strategies

from pdfa_learning.learn_pdfa.utils.multiset.naive import NaiveMultiset
from pdfa_learning.learn_pdfa.utils.multiset.tree import (
    PrefixTreeMultiset,
    ReadOnlyPrefixTreeMultiset,
)


@pytest.mark.parametrize("multiset_class", [NaiveMultiset, PrefixTreeMultiset])
def test_multiset(multiset_class):
    """Test multiset."""
    multiset = multiset_class()
    assert multiset.size == 0
    assert multiset.traces == set()
    assert set(multiset.items()) == set()
    assert multiset.get_probability((0, 1, -1)) == 0.0
    assert multiset.get_prefix_probability((0,)) == 0.0

    multiset.add((0,))
    assert multiset.size == 1
    assert multiset.traces == {(0,)}
    assert set(multiset.items()) == {((0,), 1)}
    assert multiset.get_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0,)) == 1.0

    multiset.add((0, 1))
    assert multiset.size == 2
    assert multiset.traces == {
        (0,),
        (
            0,
            1,
        ),
    }
    assert set(multiset.items()) == {((0,), 1), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 0.5
    assert (
        multiset.get_probability(
            (
                0,
                1,
            )
        )
        == 0.5
    )
    assert multiset.get_prefix_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0, 1)) == 1 / 2

    multiset.add((0,))
    assert multiset.size == 3
    assert multiset.traces == {(0,), (0, 1)}
    assert set(multiset.items()) == {((0,), 2), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 2 / 3
    assert (
        multiset.get_probability(
            (
                0,
                1,
            )
        )
        == 1 / 3
    )
    assert multiset.get_prefix_probability((0,)) == 1.0
    assert multiset.get_prefix_probability((0, 1)) == 1 / 3

    multiset.add((1,))
    assert multiset.size == 4
    assert multiset.traces == {(0,), (1,), (0, 1)}
    assert set(multiset.items()) == {((0,), 2), ((1,), 1), ((0, 1), 1)}
    assert multiset.get_probability((0,)) == 1 / 2
    assert multiset.get_probability((1,)) == 1 / 4
    assert (
        multiset.get_probability(
            (
                0,
                1,
            )
        )
        == 1 / 4
    )
    assert multiset.get_prefix_probability((0,)) == 3 / 4
    assert multiset.get_prefix_probability((0, 1)) == 1 / 4
    assert multiset.get_prefix_probability((1,)) == 1 / 4


@given(
    samples=strategies.lists(
        strategies.lists(
            strategies.integers(min_value=0, max_value=4), min_size=0, max_size=100
        ),
        min_size=0,
        max_size=1000,
    )
)
@settings(max_examples=1000)
def test_naive_multiset_and_prefix_based_multiset_equivalent(samples):
    """Test equivalence between two multiset implementations."""
    multiset_1 = NaiveMultiset()
    multiset_2 = PrefixTreeMultiset()

    for s in samples:
        s = tuple(s)
        multiset_1.add(s)
        multiset_2.add(s)

    multiset_3 = ReadOnlyPrefixTreeMultiset(
        {multiset_2._node, multiset_2._node, multiset_2._node}
    )

    assert multiset_1.size == multiset_2.size == multiset_3.size
    assert multiset_1.traces == multiset_2.traces == multiset_2.traces
    assert set(multiset_1.items()) == set(multiset_2.items()) == set(multiset_2.items())

    for s in samples:
        s = tuple(s)
        assert (
            multiset_1.get_counts(s)
            == multiset_2.get_counts(s)
            == multiset_3.get_counts(s)
        )
        assert (
            multiset_1.get_probability(s)
            == multiset_2.get_probability(s)
            == multiset_3.get_probability(s)
        )
        assert (
            multiset_1.get_prefix_probability(s)
            == multiset_1.get_prefix_probability(s)
            == multiset_3.get_prefix_probability(s)
        )
