"""Guanaco: global-uniformity evaluation of conjunctive queries.

Implements Algorithms 1 to 3 of Abo Khamis and Chen, arXiv:2610.05440.
"""

from .cleanup import cleanup
from .config import Configuration, answers
from .eval import guanaco
from .subw import subw

__all__ = ["Configuration", "answers", "cleanup", "guanaco", "subw"]
