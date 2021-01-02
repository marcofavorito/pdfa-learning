"""Base helper module."""
from pdfa_learning.types import TransitionFunctionDict


def assert_(condition: bool, message: str = ""):
    """User-defined assert."""
    if not condition:
        raise AssertionError(message)


def normalize(f: TransitionFunctionDict) -> TransitionFunctionDict:
    """Normalize a transition function."""
    result: TransitionFunctionDict = {}

    for start, outgoing in f.items():
        result[start] = {}
        total = sum([p for _, (_, p) in outgoing.items()])
        for char, (end, prob) in outgoing.items():
            result[start][char] = (end, prob / total)

    return result
