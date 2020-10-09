"""Common utilities for the learning PDFA algorithm."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import ceil
from multiprocessing import Pool
from typing import Callable, Optional, Sequence

from src.helpers.base import assert_
from src.pdfa import PDFA
from src.pdfa.types import Word


class Generator(ABC):
    """Wrapper to a PDFA to make sampling as a function call."""

    @abstractmethod
    def sample(self, n: int = 1) -> Sequence[Word]:
        """
        Generate a sample of size n.

        :param n: the size of the sample.
        :return: the list of sampled traces.
        """


class SimpleGenerator(Generator):
    """A simple sample generator."""

    def __init__(self, pdfa: PDFA):
        """Initialize an abstract generator."""
        self._pdfa = pdfa

    def __call__(self):
        """Sample a trace."""
        return self._pdfa.sample()

    def sample(self, n: int = 1) -> Sequence[Word]:
        """Generate a sample of size n."""
        return [self() for _ in range(n)]


class MultiprocessedGenerator(Generator):
    """Generate a sample, multiprocessed."""

    def __init__(self, generator: Generator, nb_processes: int = 4):
        """
        Generate a sample.

        :param nb_processes: the number of processes.
        """
        self._generator = generator
        self._nb_processes = nb_processes
        self._pool = Pool(nb_processes)

    def __call__(self):
        """Sample a trace."""
        return self._generator.sample()

    @staticmethod
    def _job(n: int, sample_func: Callable[[int], Sequence[Word]]):
        return [sample_func(1)[0] for _ in range(n)]

    def sample(self, n: int = 1) -> Sequence[Word]:
        """Generate a sample, multiprocessed."""
        n_per_process = ceil(n / self._nb_processes)
        sample = []

        results = [
            self._pool.apply_async(
                self._job, args=[n_per_process, self._generator.sample]
            )
            for _ in range(self._nb_processes)
        ]
        for r in results:
            n_samples = r.get()
            sample.extend(n_samples)

        nb_samples_to_drop = len(sample) - n
        return sample[: len(sample) - nb_samples_to_drop]


@dataclass(frozen=True)
class _Params:
    """
    Parameters for the learning algorithm.

    sample_generator: the sample generator from the true PDFA.
    alphabet_size: the alphabet size.
    epsilon: the tolerance error.
    delta: the failure probability for the subgraph construction.
    delta: the failure probability for the probability estimation.
    mu: the distinguishability factor.
    n: the upper bound of the number of states.
    """

    sample_generator: Generator
    alphabet_size: int
    epsilon: float = 0.05
    delta_1: float = 0.1
    delta_2: float = 0.1
    mu: float = 0.4
    n: int = 3
    # debug parameters - force upper bounds
    m0_max_debug: Optional[int] = None
    n1_max_debug: Optional[int] = None
    n2_max_debug: Optional[int] = None

    def __post_init__(self):
        assert_(
            self.delta_1 + self.delta_2 <= 1.0,
            "Sum of two probabilities cannot be greater than 1.",
        )
