"""LIME local explanation sampling and surrogate fitting accelerated by Mojo."""

from . import (
    discretize,
    explanation,
    kernels,
    lime_base,
    lime_image,
    lime_tabular,
    lime_text,
)
from ._lib import build
from .lime_base import LimeBase
from .lime_image import ImageExplanation, LimeImageExplainer
from .lime_tabular import LimeTabularExplainer, TableDomainMapper
from .lime_text import (
    IndexedCharacters,
    IndexedString,
    LimeTextExplainer,
    TextDomainMapper,
)

__version__ = "0.1.0"

__all__ = [
    "LimeBase",
    "LimeTabularExplainer",
    "LimeTextExplainer",
    "LimeImageExplainer",
    "ImageExplanation",
    "TableDomainMapper",
    "TextDomainMapper",
    "IndexedString",
    "IndexedCharacters",
    "build",
    "kernels",
    "lime_base",
    "lime_tabular",
    "lime_text",
    "lime_image",
    "discretize",
    "explanation",
]
