"""Convenience imports for the public ``tf2_utils`` helper API."""

from __future__ import annotations

from . import alignment as align
from . import calculations as calc
from . import comparisons as comp
from . import conversions as conv
from . import reparenting as reparent
from . import tree as tree
from .alignment import *  # noqa: F401,F403
from .calculations import *  # noqa: F401,F403
from .comparisons import *  # noqa: F401,F403
from .conversions import *  # noqa: F401,F403
from .node import __all__ as _node_all
from .reparenting import *  # noqa: F401,F403
from .tree import *  # noqa: F401,F403
from .node import *  # noqa: F401,F403

alignment = align
calculations = calc
comparisons = comp
conversions = conv
reparenting = reparent

ALIGNMENT_METHODS = tuple(align.__all__)
CALCULATION_METHODS = tuple(calc.__all__)
COMPARISON_METHODS = tuple(comp.__all__)
CONVERSION_METHODS = tuple(conv.__all__)
NODE_METHODS = tuple(_node_all)
REPARENTING_METHODS = tuple(reparent.__all__)
TREE_METHODS = tuple(tree.__all__)


def available_methods() -> dict[str, tuple[str, ...]]:
    """Return public helper names grouped by utility module."""
    return {
        "alignment": ALIGNMENT_METHODS,
        "conversions": CONVERSION_METHODS,
        "calculations": CALCULATION_METHODS,
        "comparisons": COMPARISON_METHODS,
        "node": NODE_METHODS,
        "reparenting": REPARENTING_METHODS,
        "tree": TREE_METHODS,
    }


__all__ = (
    [
        "align",
        "alignment",
        "ALIGNMENT_METHODS",
        "calc",
        "calculations",
        "comp",
        "comparisons",
        "conv",
        "conversions",
        "reparent",
        "reparenting",
        "tree",
        "available_methods",
        "CALCULATION_METHODS",
        "COMPARISON_METHODS",
        "CONVERSION_METHODS",
        "NODE_METHODS",
        "REPARENTING_METHODS",
        "TREE_METHODS",
    ]
    + align.__all__
    + calc.__all__
    + comp.__all__
    + conv.__all__
    + list(_node_all)
    + reparent.__all__
    + tree.__all__
)
