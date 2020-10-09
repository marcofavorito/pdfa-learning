"""Types for the package PDFA."""
from typing import Dict, Sequence, Tuple

State = int
Character = int
Word = Sequence[Character]
TransitionFunctionDict = Dict[State, Dict[Character, Tuple[State, float]]]
