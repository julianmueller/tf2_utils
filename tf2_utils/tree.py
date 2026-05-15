"""Helpers for building small TF trees from compact nested tuples."""

from __future__ import annotations

from collections.abc import Sequence

from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from std_msgs.msg import Header


__all__ = [
    "flatten_transform_tree",
    "transform_tree",
]


def transform_tree(
    name: str,
    transform: Transform | None = None,
    children: Sequence[tuple] | None = None,
) -> tuple[str, Transform, tuple]:
    """Create a compact transform tree node tuple.

    The returned shape is ``(name, transform, children)`` and is accepted by
    ``flatten_transform_tree``. This keeps small example trees readable without
    introducing a custom message type.
    """
    if not name:
        raise ValueError("name is required")
    if transform is None:
        transform = Transform()
    if not isinstance(transform, Transform):
        raise TypeError("transform must be a geometry_msgs/Transform")
    return (name, transform, tuple(children or ()))


def _unpack_transform_tree_node(tree: tuple) -> tuple[str, Transform, tuple]:
    """Validate and unpack one compact transform tree node."""
    if not isinstance(tree, tuple) or len(tree) != 3:
        raise TypeError("tree nodes must be created as (name, transform, children) tuples")
    name, transform, children = tree
    if not isinstance(name, str) or not name:
        raise ValueError("tree node name is required")
    if not isinstance(transform, Transform):
        raise TypeError("tree node transform must be a geometry_msgs/Transform")
    if isinstance(children, str) or not isinstance(children, Sequence):
        raise TypeError("tree node children must be a sequence")
    return name, transform, tuple(children)


def flatten_transform_tree(parent_frame: str, tree: tuple) -> list[TransformStamped]:
    """Flatten a compact transform tree into parent-to-child ``TransformStamped`` edges."""
    if not parent_frame:
        raise ValueError("parent_frame is required")

    result = []
    visited = {parent_frame}

    def _walk(parent: str, node: tuple) -> None:
        name, transform, children = _unpack_transform_tree_node(node)
        if name in visited:
            raise ValueError(f"frame {name!r} appears more than once in the tree")
        visited.add(name)
        result.append(
            TransformStamped(
                header=Header(frame_id=parent),
                child_frame_id=name,
                transform=transform,
            )
        )
        for child in children:
            _walk(name, child)

    name, _, children = _unpack_transform_tree_node(tree)
    if name == parent_frame:
        for child in children:
            _walk(parent_frame, child)
    else:
        _walk(parent_frame, tree)
    return result
