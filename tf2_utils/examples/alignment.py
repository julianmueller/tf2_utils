"""Example transforms showing the tf2_utils alignment helpers."""

from __future__ import annotations

from math import cos, sin, tau

from geometry_msgs.msg import Point, Quaternion, Transform, Vector3

from tf2_utils.alignment import (
    X_AXIS,
    XY_PLANE,
    XZ_PLANE,
    Y_AXIS,
    YZ_PLANE,
    Z_AXIS,
    align_transform_axis_parallel_to_transform_axis,
    align_transform_axis_to_point,
    align_transform_axis_to_transform_offset,
    align_transform_axis_to_vector,
    align_transform_plane_to_normal,
)
from tf2_utils.conversions import euler_to_quat


BASE_COUNT = 20
TARGET_COUNT = BASE_COUNT
BASE_RING_RADIUS = 1.5
TARGET_RING_RADIUS = 2.0
BASE_Z = 0.0
TARGET_Z = 1.0


def _base_point(index: int, z: float = 0.2) -> Point:
    """Place examples evenly around a circle for easier RViz inspection."""
    angle = tau * float(index) / float(BASE_COUNT)
    return Point(x=BASE_RING_RADIUS * cos(angle), y=BASE_RING_RADIUS * sin(angle), z=z)


def _base_transform(index: int) -> Transform:
    """Create a transform on the example ring."""
    point = _base_point(index, z=BASE_Z)
    return Transform(translation=Vector3(x=point.x, y=point.y, z=point.z))


def _target_point(index: int) -> Point:
    """Place reference targets on an elevated circle."""
    angle = tau * float(index) / float(TARGET_COUNT)
    return Point(
        x=TARGET_RING_RADIUS * cos(angle),
        y=TARGET_RING_RADIUS * sin(angle),
        z=TARGET_Z,
    )


def _target_orientation(index: int) -> Quaternion:
    """Create a varied target orientation so reference axes are visible in RViz."""
    return euler_to_quat(
        0.22 * float((index % 4) - 1),
        0.18 * float((index % 5) - 2),
        tau * float(index) / float(TARGET_COUNT) + 0.35,
    )


def _target_transform(index: int) -> Transform:
    """Create a static reference target transform on the elevated ring."""
    point = _target_point(index)
    return Transform(
        translation=Vector3(x=point.x, y=point.y, z=point.z),
        rotation=_target_orientation(index),
    )


def _direction_to_target(index: int, origin: Point) -> Vector3:
    """Create a vector from ``origin`` to one target on the elevated ring."""
    target = _target_point(index)
    return Vector3(
        x=target.x - origin.x,
        y=target.y - origin.y,
        z=target.z - origin.z,
    )


BASE_TARGETS = {f"target_{index:02d}": _target_transform(index) for index in range(TARGET_COUNT)}


def create(parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Create all alignment example transforms."""
    plus_x_to_00 = align_transform_axis_to_point(
        _base_transform(0),
        _target_point(0),
        axis=X_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    plus_z_to_01 = align_transform_axis_to_point(
        _base_transform(1),
        _target_point(1),
        axis=Z_AXIS,
        normal_axis=X_AXIS,
        normal_direction=Vector3(x=1.0, y=0.0, z=0.0),
    )
    plus_x_to_02 = align_transform_axis_to_point(
        _base_transform(2),
        _target_point(2),
        axis=X_AXIS,
        normal_axis=Y_AXIS,
        normal_direction=Vector3(x=-0.2, y=1.0, z=0.4),
    )
    plus_y_to_03 = align_transform_axis_to_point(
        _base_transform(3),
        _target_point(3),
        axis=Y_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=0.3, y=0.0, z=1.0),
    )
    minus_x_to_04 = align_transform_axis_to_point(
        _base_transform(4),
        _target_point(4),
        axis=X_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=0.2, y=0.6, z=1.0),
        flip_axis=True,
    )
    plus_y_to_05 = align_transform_axis_to_vector(
        _base_transform(5),
        _direction_to_target(5, _base_point(5, 0.2)),
        axis=Y_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=0.0, y=-0.25, z=1.0),
    )
    minus_z_to_06 = align_transform_axis_to_vector(
        _base_transform(6),
        _direction_to_target(6, _base_point(6, 0.2)),
        axis=Z_AXIS,
        normal_axis=Y_AXIS,
        normal_direction=Vector3(x=0.0, y=1.0, z=0.2),
        flip_axis=True,
    )
    plus_x_to_07 = align_transform_axis_to_vector(
        _base_transform(7),
        _direction_to_target(7, _base_point(7, 0.2)),
        axis=X_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=-0.8, y=0.3, z=0.2),
    )
    minus_y_to_08 = align_transform_axis_to_vector(
        _base_transform(8),
        _direction_to_target(8, _base_point(8, 0.2)),
        axis=Y_AXIS,
        normal_axis=X_AXIS,
        normal_direction=Vector3(x=1.0, y=0.3, z=-0.2),
        flip_axis=True,
    )
    xy_plane_to_09 = align_transform_plane_to_normal(
        _base_transform(9),
        _direction_to_target(9, _base_point(9, 0.2)),
        plane=XY_PLANE,
        in_plane_axis=X_AXIS,
        in_plane_direction=Vector3(x=1.0, y=0.2, z=0.0),
    )
    xy_plane_to_10 = align_transform_plane_to_normal(
        _base_transform(10),
        _direction_to_target(10, _base_point(10, 0.2)),
        plane=XY_PLANE,
        in_plane_axis=X_AXIS,
        in_plane_direction=Vector3(x=0.5, y=1.0, z=0.2),
    )
    xz_plane_to_11 = align_transform_plane_to_normal(
        _base_transform(11),
        _direction_to_target(11, _base_point(11, 0.2)),
        plane=XZ_PLANE,
        in_plane_axis=X_AXIS,
        in_plane_direction=Vector3(x=0.3, y=0.0, z=1.0),
    )
    yz_plane_to_12 = align_transform_plane_to_normal(
        _base_transform(12),
        _direction_to_target(12, _base_point(12, 0.2)),
        plane=YZ_PLANE,
        in_plane_axis=Y_AXIS,
        in_plane_direction=Vector3(x=-0.1, y=0.8, z=0.7),
    )
    minus_xz_plane_to_13 = align_transform_plane_to_normal(
        _base_transform(13),
        _direction_to_target(13, _base_point(13, 0.25)),
        plane=XZ_PLANE,
        in_plane_axis=Z_AXIS,
        in_plane_direction=Vector3(x=0.9, y=0.2, z=0.0),
        flip_normal_axis=True,
    )
    parallel_plus_x_to_plus_x_14 = align_transform_axis_parallel_to_transform_axis(
        _base_transform(14),
        _target_transform(14),
        axis=X_AXIS,
        reference_axis=X_AXIS,
        normal_axis=Z_AXIS,
        reference_normal_axis=Z_AXIS,
    )
    parallel_plus_z_to_plus_y_15 = align_transform_axis_parallel_to_transform_axis(
        _base_transform(15),
        _target_transform(15),
        axis=Z_AXIS,
        reference_axis=Y_AXIS,
        normal_axis=X_AXIS,
        reference_normal_axis=Z_AXIS,
    )
    parallel_minus_y_to_plus_z_16 = align_transform_axis_parallel_to_transform_axis(
        _base_transform(16),
        _target_transform(16),
        axis=Y_AXIS,
        reference_axis=Z_AXIS,
        normal_axis=X_AXIS,
        reference_normal_axis=X_AXIS,
        flip_axis=True,
    )
    offset_plus_x_to_x_17 = align_transform_axis_to_transform_offset(
        _base_transform(17),
        _target_transform(17),
        target_offset=Point(x=0.7, y=0.0, z=0.0),
        axis=X_AXIS,
        normal_axis=Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    offset_plus_z_to_y_18 = align_transform_axis_to_transform_offset(
        _base_transform(18),
        _target_transform(18),
        target_offset=Point(x=0.0, y=0.85, z=0.0),
        axis=Z_AXIS,
        normal_axis=X_AXIS,
        normal_direction=Vector3(x=1.0, y=0.0, z=0.0),
    )
    offset_minus_x_to_z_19 = align_transform_axis_to_transform_offset(
        _base_transform(19),
        _target_transform(19),
        target_offset=Point(x=0.0, y=0.0, z=0.6),
        axis=X_AXIS,
        normal_axis=Y_AXIS,
        normal_direction=Vector3(x=0.0, y=1.0, z=0.0),
        flip_axis=True,
    )

    return [
        (parent_frame, "+x_to_00", plus_x_to_00),
        (parent_frame, "+z_to_01", plus_z_to_01),
        (parent_frame, "+x_to_02", plus_x_to_02),
        (parent_frame, "+y_to_03", plus_y_to_03),
        (parent_frame, "-x_to_04", minus_x_to_04),
        (parent_frame, "+y_to_05", plus_y_to_05),
        (parent_frame, "-z_to_06", minus_z_to_06),
        (parent_frame, "+x_to_07", plus_x_to_07),
        (parent_frame, "-y_to_08", minus_y_to_08),
        (parent_frame, "xy_plane_to_09", xy_plane_to_09),
        (parent_frame, "xy_plane_to_10", xy_plane_to_10),
        (parent_frame, "xz_plane_to_11", xz_plane_to_11),
        (parent_frame, "yz_plane_to_12", yz_plane_to_12),
        (parent_frame, "-xz_plane_to_13", minus_xz_plane_to_13),
        (parent_frame, "parallel_+x_to_+x_14", parallel_plus_x_to_plus_x_14),
        (parent_frame, "parallel_+z_to_+y_15", parallel_plus_z_to_plus_y_15),
        (parent_frame, "parallel_-y_to_+z_16", parallel_minus_y_to_plus_z_16),
        (parent_frame, "offset_+x_to_x_17", offset_plus_x_to_x_17),
        (parent_frame, "offset_+z_to_y_18", offset_plus_z_to_y_18),
        (parent_frame, "offset_-x_to_z_19", offset_minus_x_to_z_19),
    ]
