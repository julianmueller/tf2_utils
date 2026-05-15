"""Convenience imports for the public ``tf2_utils`` helper API."""

from __future__ import annotations

from . import calculations as calc
from . import comparisons as comp
from . import conversions as conv
from .calculations import *  # noqa: F401,F403
from .comparisons import *  # noqa: F401,F403
from .conversions import *  # noqa: F401,F403
from .node import __all__ as _node_all
from .node import *  # noqa: F401,F403

calculations = calc
comparisons = comp
conversions = conv

CALCULATION_METHODS = tuple(calc.__all__)
COMPARISON_METHODS = tuple(comp.__all__)
CONVERSION_METHODS = tuple(conv.__all__)
NODE_METHODS = tuple(_node_all)


def available_methods() -> dict[str, tuple[str, ...]]:
    """Return public helper names grouped by utility module."""
    return {
        "conversions": CONVERSION_METHODS,
        "calculations": CALCULATION_METHODS,
        "comparisons": COMPARISON_METHODS,
        "node": NODE_METHODS,
    }


__all__ = (
    [
        "calc",
        "calculations",
        "comp",
        "comparisons",
        "conv",
        "conversions",
        "available_methods",
        "CALCULATION_METHODS",
        "COMPARISON_METHODS",
        "CONVERSION_METHODS",
        "NODE_METHODS",
    ]
    + calc.__all__
    + comp.__all__
    + conv.__all__
    + list(_node_all)
)
