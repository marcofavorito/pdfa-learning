"""
Package that contains the implementation of [1].

- [1] Palmer N., Goldberg P.W. (2005)
      PAC-Learnability of Probabilistic Deterministic Finite State Automata
      in Terms of Variation Distance.
      In: Jain S., Simon H.U., Tomita E. (eds) Algorithmic Learning Theory. ALT 2005.
      Lecture Notes in Computer Science, vol 3734. Springer, Berlin, Heidelberg.
      https://doi.org/10.1007/11564089_14
"""
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG, format="[%(asctime)s][%(name)s][%(levelname)s] %(message)s"
)
