"""Alignment helpers for orienting ROS 2 transforms toward geometric targets."""

from __future__ import annotations

import math

from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
import numpy as np
from scipy.spatial.transform import Rotation

from . import conversions as conv


X_AXIS = 0
Y_AXIS = 1
Z_AXIS = 2

XY_PLANE = Z_AXIS
XZ_PLANE = Y_AXIS
YZ_PLANE = X_AXIS

__all__ = [
    "X_AXIS",
    "Y_AXIS",
    "Z_AXIS",
    "XY_PLANE",
    "XZ_PLANE",
    "YZ_PLANE",
    "quat_from_axis_alignment",
    "quat_from_plane_alignment",
    "align_pose_axis_parallel_to_pose_axis",
    "align_pose_axis_to_pose_offset",
    "align_pose_axis_to_point",
    "align_pose_axis_to_vector",
    "align_pose_plane_to_normal",
    "align_transform_axis_parallel_to_transform_axis",
    "align_transform_axis_to_transform_offset",
    "align_transform_axis_to_point",
    "align_transform_axis_to_vector",
    "align_transform_plane_to_normal",
]

_AXIS_VECTORS = {
    X_AXIS: np.array([1.0, 0.0, 0.0], dtype=float),
    Y_AXIS: np.array([0.0, 1.0, 0.0], dtype=float),
    Z_AXIS: np.array([0.0, 0.0, 1.0], dtype=float),
}

_WORLD_FALLBACK_AXES = (
    np.array([1.0, 0.0, 0.0], dtype=float),
    np.array([0.0, 1.0, 0.0], dtype=float),
    np.array([0.0, 0.0, 1.0], dtype=float),
)


def _axis_vector(axis: int, flip_axis: bool = False, name: str = "axis") -> np.ndarray:
    """Return a local cardinal axis as a unit vector, optionally flipped."""
    if isinstance(axis, bool) or axis not in _AXIS_VECTORS:
        raise ValueError(f"{name} must be X_AXIS, Y_AXIS, Z_AXIS, 0, 1, or 2")
    vector = _AXIS_VECTORS[axis].copy()
    return -vector if flip_axis else vector


def _normalised_vector(vector: Vector3, name: str) -> np.ndarray:
    """Return a non-zero Vector3 as a normalized NumPy vector."""
    if not isinstance(vector, Vector3):
        raise TypeError(f"{name} must be a geometry_msgs/Vector3")
    array = conv.vector3_to_np(vector)
    norm = np.linalg.norm(array)
    if math.isclose(norm, 0.0):
        raise ValueError(f"{name} must not be a zero vector")
    return array / norm


def _orthogonal_direction(direction: np.ndarray, hint: Vector3) -> np.ndarray:
    """Project a direction hint into the plane perpendicular to ``direction``."""
    normal = _normalised_vector(hint, "normal_direction")
    normal = normal - direction * float(np.dot(normal, direction))
    if not math.isclose(np.linalg.norm(normal), 0.0):
        return normal / np.linalg.norm(normal)

    fallback = min(_WORLD_FALLBACK_AXES, key=lambda axis: abs(float(np.dot(axis, direction))))
    normal = fallback - direction * float(np.dot(fallback, direction))
    return normal / np.linalg.norm(normal)


def _world_axis_direction(
    orientation: Quaternion,
    axis: int,
    flip_axis: bool = False,
    name: str = "axis",
) -> Vector3:
    """Return a local cardinal axis expressed in the world frame."""
    direction = conv.quat_to_matrix(orientation) @ _axis_vector(axis, flip_axis, name)
    return conv.np_to_vector3(direction)


def _reference_normal_axis(reference_axis: int, normal_axis: int, reference_normal_axis: int | None) -> int:
    """Return a reference normal axis that is orthogonal to the reference primary axis."""
    _axis_vector(reference_axis, name="reference_axis")
    if reference_normal_axis is None:
        if reference_axis != normal_axis:
            return normal_axis
        return next(axis for axis in (X_AXIS, Y_AXIS, Z_AXIS) if axis != reference_axis)
    _axis_vector(reference_normal_axis, name="reference_normal_axis")
    if reference_axis == reference_normal_axis:
        raise ValueError("reference_axis and reference_normal_axis must be different cardinal axes")
    return reference_normal_axis


def _target_offset_or_origin(target_offset: Point | None) -> Point:
    """Return a target-local point offset, defaulting to the target origin."""
    if target_offset is None:
        return Point()
    if not isinstance(target_offset, Point):
        raise TypeError("target_offset must be a geometry_msgs/Point")
    return target_offset


def _pose_offset_to_point(target_pose: Pose, target_offset: Point | None) -> Point:
    """Return a target-local point offset expressed in the parent frame."""
    offset = _target_offset_or_origin(target_offset)
    point = (
        conv.point_to_np(target_pose.position)
        + conv.quat_to_matrix(target_pose.orientation) @ conv.point_to_np(offset)
    )
    return conv.np_to_point(point)


def _transform_offset_to_point(target_transform: Transform, target_offset: Point | None) -> Point:
    """Return a target-local point offset expressed in the parent frame."""
    offset = _target_offset_or_origin(target_offset)
    point = (
        conv.vector3_to_np(target_transform.translation)
        + conv.quat_to_matrix(target_transform.rotation) @ conv.point_to_np(offset)
    )
    return conv.np_to_point(point)


def _rotation_from_axis_alignment(
    axis: int,
    direction: Vector3,
    normal_axis: int,
    normal_direction: Vector3,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Rotation:
    """Build a rotation from two local axis-to-world direction constraints."""
    local_primary = _axis_vector(axis, flip_axis, "axis")
    local_normal = _axis_vector(normal_axis, flip_normal_axis, "normal_axis")
    if not math.isclose(float(np.dot(local_primary, local_normal)), 0.0):
        raise ValueError("axis and normal_axis must be different cardinal axes")

    world_primary = _normalised_vector(direction, "direction")
    world_normal = _orthogonal_direction(world_primary, normal_direction)

    local_third = np.cross(local_primary, local_normal)
    world_third = np.cross(world_primary, world_normal)
    local_basis = np.column_stack([local_primary, local_normal, local_third])
    world_basis = np.column_stack([world_primary, world_normal, world_third])
    return Rotation.from_matrix(world_basis @ local_basis.T)


def _quat_from_axis_alignment(
    axis: int,
    direction: Vector3,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Quaternion:
    """Return a quaternion whose local ``axis`` points along ``direction``.

    ``normal_axis`` controls roll around the aligned axis by pointing as close as
    possible toward ``normal_direction`` while remaining perpendicular to
    ``direction``.
    """
    rotation = _rotation_from_axis_alignment(
        axis,
        direction,
        normal_axis,
        normal_direction or Vector3(x=0.0, y=0.0, z=1.0),
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )
    return conv.np_to_quat(rotation)


def _quat_from_plane_alignment(
    plane: int,
    normal: Vector3,
    in_plane_axis: int = X_AXIS,
    in_plane_direction: Vector3 | None = None,
    flip_normal_axis: bool = False,
    flip_in_plane_axis: bool = False,
) -> Quaternion:
    """Return a quaternion whose local cardinal ``plane`` has the requested normal."""
    return _quat_from_axis_alignment(
        plane,
        normal,
        in_plane_axis,
        in_plane_direction or Vector3(x=1.0, y=0.0, z=0.0),
        flip_axis=flip_normal_axis,
        flip_normal_axis=flip_in_plane_axis,
    )


def quat_from_axis_alignment(
    axis: int,
    direction: Vector3,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Quaternion:
    """Return a quaternion whose local ``axis`` points along ``direction``."""
    return _quat_from_axis_alignment(
        axis,
        direction,
        normal_axis,
        normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def quat_from_plane_alignment(
    plane: int,
    normal: Vector3,
    in_plane_axis: int = X_AXIS,
    in_plane_direction: Vector3 | None = None,
    flip_normal_axis: bool = False,
    flip_in_plane_axis: bool = False,
) -> Quaternion:
    """Return a quaternion whose local cardinal ``plane`` has the requested normal."""
    return _quat_from_plane_alignment(
        plane,
        normal,
        in_plane_axis,
        in_plane_direction,
        flip_normal_axis=flip_normal_axis,
        flip_in_plane_axis=flip_in_plane_axis,
    )


def align_pose_axis_to_vector(
    pose: Pose,
    direction: Vector3,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Pose:
    """Return ``pose`` rotated so its local ``axis`` points along ``direction``."""
    aligned = Pose()
    aligned.position = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
    aligned.orientation = _quat_from_axis_alignment(
        axis,
        direction,
        normal_axis,
        normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )
    return aligned


def align_pose_axis_parallel_to_pose_axis(
    pose: Pose,
    reference_pose: Pose,
    axis: int = X_AXIS,
    reference_axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    reference_normal_axis: int | None = None,
    flip_axis: bool = False,
    flip_reference_axis: bool = False,
    flip_normal_axis: bool = False,
    flip_reference_normal_axis: bool = False,
) -> Pose:
    """Return ``pose`` rotated so its local ``axis`` is parallel to a reference pose axis."""
    reference_normal_axis = _reference_normal_axis(reference_axis, normal_axis, reference_normal_axis)
    return align_pose_axis_to_vector(
        pose,
        _world_axis_direction(reference_pose.orientation, reference_axis, flip_reference_axis, "reference_axis"),
        axis=axis,
        normal_axis=normal_axis,
        normal_direction=_world_axis_direction(
            reference_pose.orientation,
            reference_normal_axis,
            flip_reference_normal_axis,
            "reference_normal_axis",
        ),
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_pose_axis_to_pose_offset(
    pose: Pose,
    target_pose: Pose,
    target_offset: Point | None = None,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Pose:
    """Return ``pose`` rotated so its local ``axis`` points at a point inside ``target_pose``."""
    return align_pose_axis_to_point(
        pose,
        _pose_offset_to_point(target_pose, target_offset),
        axis=axis,
        normal_axis=normal_axis,
        normal_direction=normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_transform_axis_to_vector(
    transform: Transform,
    direction: Vector3,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Transform:
    """Return ``transform`` rotated so its local ``axis`` points along ``direction``."""
    aligned = Transform()
    aligned.translation = Vector3(
        x=transform.translation.x,
        y=transform.translation.y,
        z=transform.translation.z,
    )
    aligned.rotation = _quat_from_axis_alignment(
        axis,
        direction,
        normal_axis,
        normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )
    return aligned


def align_transform_axis_parallel_to_transform_axis(
    transform: Transform,
    reference_transform: Transform,
    axis: int = X_AXIS,
    reference_axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    reference_normal_axis: int | None = None,
    flip_axis: bool = False,
    flip_reference_axis: bool = False,
    flip_normal_axis: bool = False,
    flip_reference_normal_axis: bool = False,
) -> Transform:
    """Return ``transform`` rotated so its local ``axis`` is parallel to a reference transform axis."""
    reference_normal_axis = _reference_normal_axis(reference_axis, normal_axis, reference_normal_axis)
    return align_transform_axis_to_vector(
        transform,
        _world_axis_direction(reference_transform.rotation, reference_axis, flip_reference_axis, "reference_axis"),
        axis=axis,
        normal_axis=normal_axis,
        normal_direction=_world_axis_direction(
            reference_transform.rotation,
            reference_normal_axis,
            flip_reference_normal_axis,
            "reference_normal_axis",
        ),
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_transform_axis_to_transform_offset(
    transform: Transform,
    target_transform: Transform,
    target_offset: Point | None = None,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Transform:
    """Return ``transform`` rotated so its local ``axis`` points at a point inside ``target_transform``."""
    return align_transform_axis_to_point(
        transform,
        _transform_offset_to_point(target_transform, target_offset),
        axis=axis,
        normal_axis=normal_axis,
        normal_direction=normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_pose_axis_to_point(
    pose: Pose,
    point: Point,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Pose:
    """Return ``pose`` rotated so its local ``axis`` points from the pose to ``point``."""
    if not isinstance(point, Point):
        raise TypeError("point must be a geometry_msgs/Point")
    direction = conv.np_to_vector3(conv.point_to_np(point) - conv.point_to_np(pose.position))
    return align_pose_axis_to_vector(
        pose,
        direction,
        axis,
        normal_axis,
        normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_transform_axis_to_point(
    transform: Transform,
    point: Point,
    axis: int = X_AXIS,
    normal_axis: int = Z_AXIS,
    normal_direction: Vector3 | None = None,
    flip_axis: bool = False,
    flip_normal_axis: bool = False,
) -> Transform:
    """Return ``transform`` rotated so its local ``axis`` points from the transform to ``point``."""
    if not isinstance(point, Point):
        raise TypeError("point must be a geometry_msgs/Point")
    direction = conv.np_to_vector3(conv.point_to_np(point) - conv.vector3_to_np(transform.translation))
    return align_transform_axis_to_vector(
        transform,
        direction,
        axis,
        normal_axis,
        normal_direction,
        flip_axis=flip_axis,
        flip_normal_axis=flip_normal_axis,
    )


def align_pose_plane_to_normal(
    pose: Pose,
    normal: Vector3,
    plane: int = XY_PLANE,
    in_plane_axis: int = X_AXIS,
    in_plane_direction: Vector3 | None = None,
    flip_normal_axis: bool = False,
    flip_in_plane_axis: bool = False,
) -> Pose:
    """Return ``pose`` rotated so its local cardinal ``plane`` faces ``normal``."""
    aligned = Pose()
    aligned.position = Point(x=pose.position.x, y=pose.position.y, z=pose.position.z)
    aligned.orientation = _quat_from_plane_alignment(
        plane,
        normal,
        in_plane_axis,
        in_plane_direction,
        flip_normal_axis=flip_normal_axis,
        flip_in_plane_axis=flip_in_plane_axis,
    )
    return aligned


def align_transform_plane_to_normal(
    transform: Transform,
    normal: Vector3,
    plane: int = XY_PLANE,
    in_plane_axis: int = X_AXIS,
    in_plane_direction: Vector3 | None = None,
    flip_normal_axis: bool = False,
    flip_in_plane_axis: bool = False,
) -> Transform:
    """Return ``transform`` rotated so its local cardinal ``plane`` faces ``normal``."""
    aligned = Transform()
    aligned.translation = Vector3(
        x=transform.translation.x,
        y=transform.translation.y,
        z=transform.translation.z,
    )
    aligned.rotation = _quat_from_plane_alignment(
        plane,
        normal,
        in_plane_axis,
        in_plane_direction,
        flip_normal_axis=flip_normal_axis,
        flip_in_plane_axis=flip_in_plane_axis,
    )
    return aligned
