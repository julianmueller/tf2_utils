"""Helpers for changing TF tree parents while preserving global frame poses."""

from __future__ import annotations

from collections.abc import Sequence

from geometry_msgs.msg import Quaternion, Transform, TransformStamped
from std_msgs.msg import Header

from .calculations import chain_transforms, invert_transform, mult_transforms


__all__ = [
    "reparent_transform_stamped",
]


def _build_parent_lookup(
    transform: TransformStamped,
    tree_transforms: Sequence[TransformStamped],
) -> dict[str, tuple[str, Transform]]:
    """Build a child-frame keyed lookup of parent transforms for one TF tree."""
    lookup: dict[str, tuple[str, Transform]] = {}
    for tree_transform in tree_transforms:
        if not isinstance(tree_transform, TransformStamped):
            raise TypeError("tree_transforms must contain geometry_msgs/TransformStamped messages")
        if not tree_transform.header.frame_id or not tree_transform.child_frame_id:
            raise ValueError("tree transforms must have parent and child frame IDs")
        if tree_transform.child_frame_id == transform.child_frame_id:
            continue
        if tree_transform.child_frame_id in lookup:
            raise ValueError(f"frame {tree_transform.child_frame_id!r} has more than one parent")
        lookup[tree_transform.child_frame_id] = (tree_transform.header.frame_id, tree_transform.transform)

    if not transform.header.frame_id or not transform.child_frame_id:
        raise ValueError("transform must have parent and child frame IDs")
    lookup[transform.child_frame_id] = (transform.header.frame_id, transform.transform)
    return lookup


def _ancestor_frames(frame: str, lookup: dict[str, tuple[str, Transform]]) -> list[str]:
    """Return ``frame`` and all known ancestors up to the root of the lookup tree."""
    ancestors = []
    visited = set()
    current = frame
    while True:
        if current in visited:
            raise ValueError(f"TF tree contains a cycle at frame {current!r}")
        visited.add(current)
        ancestors.append(current)
        if current not in lookup:
            return ancestors
        current = lookup[current][0]


def _transform_from_ancestor(
    ancestor_frame: str,
    frame: str,
    lookup: dict[str, tuple[str, Transform]],
) -> Transform:
    """Compose the transform from ``ancestor_frame`` to ``frame``."""
    if ancestor_frame == frame:
        return Transform()

    transforms = []
    current = frame
    while current != ancestor_frame:
        if current not in lookup:
            raise ValueError(f"frame {frame!r} is not connected to ancestor {ancestor_frame!r}")
        parent_frame, transform = lookup[current]
        transforms.append(transform)
        current = parent_frame
    return chain_transforms(*reversed(transforms))


def reparent_transform_stamped(
    transform: TransformStamped,
    new_parent_frame: str,
    tree_transforms: Sequence[TransformStamped],
    root_frame: str | None = None,
) -> TransformStamped:
    """Return ``transform`` expressed relative to ``new_parent_frame`` without moving its child frame.

    ``tree_transforms`` must describe the surrounding TF tree as parent-to-child
    transforms. The supplied ``transform`` is treated as the authoritative old
    parent-to-child edge for ``transform.child_frame_id``. When ``root_frame`` is
    omitted, the nearest shared ancestor of the child and the new parent is used.
    """
    if not new_parent_frame:
        raise ValueError("new_parent_frame is required")
    if transform.child_frame_id == new_parent_frame:
        raise ValueError("new_parent_frame must not be the transform child frame")

    lookup = _build_parent_lookup(transform, tree_transforms)
    child_ancestors = _ancestor_frames(transform.child_frame_id, lookup)
    new_parent_ancestors = _ancestor_frames(new_parent_frame, lookup)
    if transform.child_frame_id in new_parent_ancestors:
        raise ValueError("new_parent_frame must not be a descendant of the transform child frame")

    if root_frame is None:
        new_parent_ancestor_set = set(new_parent_ancestors)
        common_ancestor = next(
            (ancestor for ancestor in child_ancestors if ancestor in new_parent_ancestor_set),
            None,
        )
        if common_ancestor is None:
            raise ValueError(
                f"frames {transform.child_frame_id!r} and {new_parent_frame!r} do not share a known ancestor"
            )
    else:
        if root_frame not in child_ancestors or root_frame not in new_parent_ancestors:
            raise ValueError(
                f"root_frame {root_frame!r} is not a shared ancestor of "
                f"{transform.child_frame_id!r} and {new_parent_frame!r}"
            )
        common_ancestor = root_frame

    ancestor_to_child = _transform_from_ancestor(common_ancestor, transform.child_frame_id, lookup)
    ancestor_to_new_parent = _transform_from_ancestor(common_ancestor, new_parent_frame, lookup)

    result = TransformStamped()
    result.header = Header(frame_id=new_parent_frame, stamp=transform.header.stamp)
    result.child_frame_id = transform.child_frame_id
    result.transform = mult_transforms(invert_transform(ancestor_to_new_parent), ancestor_to_child)
    return result
