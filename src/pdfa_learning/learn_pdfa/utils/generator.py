"""Utilities for the generation of samples from a PDFA."""
from abc import ABC, abstractmethod
from math import ceil
from multiprocessing import Pool
from typing import Callable, Sequence

from pdfa_learning.pdfa import PDFA
from pdfa_learning.pdfa.base import FINAL_SYMBOL
from pdfa_learning.types import Word


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

    def sample(self, n: int = 1, with_final: bool = False) -> Sequence[Word]:
        """Generate a sample of size n."""
        return [
            tuple(self()) + ((FINAL_SYMBOL,) if with_final else ()) for _ in range(n)
        ]


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
