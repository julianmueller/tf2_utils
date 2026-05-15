"""Tolerant comparisons for ROS 2 transform and geometry message types."""

from __future__ import annotations

import numpy as np

from std_msgs.msg import Header
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, TransformStamped, Vector3

from . import conversions as conv


__all__ = [
    "is_equal_header",
    "is_equal_point",
    "is_equal_pose",
    "is_equal_posestamped",
    "is_equal_quat",
    "is_equal_transform",
    "is_equal_transformstamped",
    "is_equal_vector3",
]


def is_equal_header(header_a: Header, header_b: Header, check_timestamp: bool = False) -> bool:
    """Compare two headers and optionally require identical ROS 2 timestamps."""
    if header_a.frame_id != header_b.frame_id:
        return False
    if not check_timestamp:
        return True
    return header_a.stamp.sec == header_b.stamp.sec and header_a.stamp.nanosec == header_b.stamp.nanosec


def is_equal_point(point_a: Point, point_b: Point, tol: float = 1e-4) -> bool:
    """Check whether two points are equal within ``tol``."""
    return bool(np.allclose(conv.point_to_np(point_a), conv.point_to_np(point_b), atol=tol, rtol=0.0))


def is_equal_vector3(vector_a: Vector3, vector_b: Vector3, tol: float = 1e-4) -> bool:
    """Check whether two vectors are equal within ``tol``."""
    return bool(
        np.allclose(
            conv.vector3_to_np(vector_a),
            conv.vector3_to_np(vector_b),
            atol=tol,
            rtol=0.0,
        )
    )


def is_equal_quat(quaternion_a: Quaternion, quaternion_b: Quaternion, tol: float = 1e-4) -> bool:
    """Check whether two quaternions represent the same rotation within ``tol``."""
    quat_a = conv.quat_to_np(quaternion_a)
    quat_b = conv.quat_to_np(quaternion_b)
    return bool(np.allclose(quat_a, quat_b, atol=tol, rtol=0.0) or np.allclose(quat_a, -quat_b, atol=tol, rtol=0.0))


def is_equal_pose(pose_a: Pose, pose_b: Pose, tol: float = 1e-4) -> bool:
    """Check whether two poses are equal within ``tol``."""
    return is_equal_point(pose_a.position, pose_b.position, tol) and is_equal_quat(
        pose_a.orientation, pose_b.orientation, tol
    )


def is_equal_posestamped(
    pose_a: PoseStamped,
    pose_b: PoseStamped,
    tol: float = 1e-4,
) -> bool:
    """Check whether two stamped poses are equal within ``tol`` and have the same frame."""
    return is_equal_pose(pose_a.pose, pose_b.pose, tol) and is_equal_header(
        pose_a.header,
        pose_b.header,
    )


def is_equal_transform(transform_a: Transform, transform_b: Transform, tol: float = 1e-4) -> bool:
    """Check whether two transforms are equal within ``tol``."""
    return is_equal_vector3(
        transform_a.translation,
        transform_b.translation,
        tol,
    ) and is_equal_quat(
        transform_a.rotation,
        transform_b.rotation,
        tol,
    )


def is_equal_transformstamped(
    transform_a: TransformStamped,
    transform_b: TransformStamped,
    tol: float = 1e-4,
) -> bool:
    """Check whether two stamped transforms are equal within ``tol`` and have matching frames."""
    return (
        is_equal_transform(transform_a.transform, transform_b.transform, tol)
        and is_equal_header(transform_a.header, transform_b.header)
        and transform_a.child_frame_id == transform_b.child_frame_id
    )
