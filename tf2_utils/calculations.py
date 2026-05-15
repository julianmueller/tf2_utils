"""NumPy-backed transform calculations for ROS 2 geometry message types."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, TransformStamped, Vector3

from . import conversions as conv


__all__ = [
    "add_points",
    "add_poses",
    "add_vector3s",
    "angular_distance_quat",
    "chain_poses",
    "chain_transform_stamped",
    "chain_transforms",
    "dist_point",
    "dist_pose",
    "dist_transform",
    "dist_vector3",
    "invert_matrix",
    "invert_pose",
    # "invert_pose_stamped",
    "invert_quat",
    "invert_transform",
    # "invert_transform_stamped",
    "lerp_float",
    "lerp_point",
    "lerp_pose",
    "lerp_quat",
    "lerp_transform",
    "lerp_vector3",
    "mult_poses",
    "mult_transform_stamped",
    "mult_transforms",
    "norm_point",
    "norm_quat",
    "norm_vector3",
    "normalise_pose",
    "normalise_pose_stamped",
    "normalise_quat",
    "normalise_transform",
    "normalise_transform_stamped",
    "scale_point",
    "scale_vector3",
    "slerp_quat",
    "transform_point",
    "transform_vector3",
]


def add_points(point_a: Point, point_b: Point) -> Point:
    """Add two ``geometry_msgs/Point`` messages component-wise."""
    return conv.np_to_point(conv.point_to_np(point_a) + conv.point_to_np(point_b))


def add_vector3s(vector_a: Vector3, vector_b: Vector3) -> Vector3:
    """Add two ``geometry_msgs/Vector3`` messages component-wise."""
    return conv.np_to_vector3(conv.vector3_to_np(vector_a) + conv.vector3_to_np(vector_b))


def add_poses(pose_a: Pose, pose_b: Pose) -> Pose:
    """Add pose translations and compose pose rotations without rotating the second translation."""
    pose = Pose()
    pose.position = add_points(pose_a.position, pose_b.position)
    rotation = Rotation.from_quat(conv.quat_to_np(pose_a.orientation)) * Rotation.from_quat(
        conv.quat_to_np(pose_b.orientation)
    )
    pose.orientation = conv.np_to_quat(rotation)
    return pose


def mult_poses(pose_a: Pose, pose_b: Pose) -> Pose:
    """Compose two poses as homogeneous transforms and return ``pose_a * pose_b``."""
    return conv.np_to_pose(conv.pose_to_np(pose_a) @ conv.pose_to_np(pose_b))


def mult_transforms(transform_a: Transform, transform_b: Transform) -> Transform:
    """Compose two transforms as homogeneous matrices and return ``transform_a * transform_b``."""
    return conv.np_to_transform(conv.transform_to_np(transform_a) @ conv.transform_to_np(transform_b))


def mult_transform_stamped(
    transform_a: TransformStamped,
    transform_b: TransformStamped,
) -> TransformStamped:
    """Compose two stamped transforms when ``a.child_frame_id`` matches ``b.header.frame_id``."""
    if transform_a.child_frame_id != transform_b.header.frame_id:
        raise ValueError(
            f"cannot chain transforms: {transform_a.child_frame_id} does not match {transform_b.header.frame_id}"
        )
    result = TransformStamped()
    result.header = transform_a.header
    result.child_frame_id = transform_b.child_frame_id
    result.transform = mult_transforms(transform_a.transform, transform_b.transform)
    return result


def chain_poses(*poses: Pose) -> Pose:
    """Compose any number of poses from left to right."""
    if not poses:
        return Pose()
    result = poses[0]
    for pose in poses[1:]:
        result = mult_poses(result, pose)
    return result


def chain_transforms(*transforms: Transform) -> Transform:
    """Compose any number of transforms from left to right."""
    if not transforms:
        return Transform()
    result = transforms[0]
    for transform in transforms[1:]:
        result = mult_transforms(result, transform)
    return result


def chain_transform_stamped(*transforms: TransformStamped) -> TransformStamped:
    """Compose a connected sequence of stamped transforms from left to right."""
    if not transforms:
        raise ValueError("at least one TransformStamped is required")
    result = transforms[0]
    for transform in transforms[1:]:
        result = mult_transform_stamped(result, transform)
    return result



def invert_matrix(matrix: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    """Return the inverse of a 4x4 homogeneous transform matrix."""
    array = np.asarray(matrix, dtype=float)
    if array.shape != (4, 4):
        raise ValueError(f"matrix must have shape (4, 4), but has shape {array.shape}")

    inverse = np.eye(4, dtype=float)
    rotation = array[:3, :3]
    translation = array[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def invert_quat(quaternion: Quaternion) -> Quaternion:
    """Return the inverse of a ``geometry_msgs/Quaternion``."""
    x, y, z, w = conv.quat_to_np(quaternion)
    return Quaternion(x=-x, y=-y, z=-z, w=w)


def invert_pose(pose: Pose | PoseStamped) -> Pose:
    """Return the inverse of a ``geometry_msgs/Pose``."""
    if isinstance(pose, PoseStamped):
        return PoseStamped(
            header=pose.header,
            pose=invert_pose(pose.pose),
        )
    return conv.np_to_pose(invert_matrix(conv.pose_to_np(pose)))


def invert_transform(transform: Transform | TransformStamped) -> Transform:
    """Return the inverse of a ``geometry_msgs/Transform``."""
    if isinstance(transform, TransformStamped):
        return TransformStamped(
            header=transform.header,
            child_frame_id=transform.child_frame_id,
            transform=invert_transform(transform.transform),
        )
    return conv.np_to_transform(invert_matrix(conv.transform_to_np(transform)))


# def invert_pose_stamped(pose_stamped: PoseStamped, child_frame: str) -> TransformStamped:
#     """Invert a stamped pose into a transform from ``child_frame`` to the pose frame."""
#     return TransformStamped(
#         transform=conv.pose_to_transform(invert_pose(pose_stamped.pose)),
#         child_frame_id=pose_stamped.header.frame_id,
#         header=Header(frame_id=child_frame, stamp=pose_stamped.header.stamp),
#     )


# def invert_transform_stamped(transform_stamped: TransformStamped) -> TransformStamped:
#     """Invert a stamped transform and swap parent and child frame ids."""
#     return TransformStamped(
#         transform=invert_transform(transform_stamped.transform),
#         child_frame_id=transform_stamped.header.frame_id,
#         header=Header(frame_id=transform_stamped.child_frame_id, stamp=transform_stamped.header.stamp),
#     )


def scale_point(point: Point, scale: float) -> Point:
    """Scale a ``geometry_msgs/Point`` uniformly in x, y, and z."""
    return conv.np_to_point(conv.point_to_np(point) * scale)


def scale_vector3(vector: Vector3, scale: float) -> Vector3:
    """Scale a ``geometry_msgs/Vector3`` uniformly in x, y, and z."""
    return conv.np_to_vector3(conv.vector3_to_np(vector) * scale)


def dist_point(point_a: Point, point_b: Point) -> float:
    """Return the Euclidean distance between two ``geometry_msgs/Point`` messages."""
    return float(np.linalg.norm(conv.point_to_np(point_b) - conv.point_to_np(point_a)))


def dist_vector3(vector_a: Vector3, vector_b: Vector3) -> float:
    """Return the Euclidean distance between two ``geometry_msgs/Vector3`` messages."""
    return float(np.linalg.norm(conv.vector3_to_np(vector_b) - conv.vector3_to_np(vector_a)))


def dist_pose(pose_a: Pose, pose_b: Pose) -> float:
    """Return the translation distance between two ``geometry_msgs/Pose`` messages."""
    return dist_point(pose_a.position, pose_b.position)


def dist_transform(transform_a: Transform, transform_b: Transform) -> float:
    """Return the translation distance between two ``geometry_msgs/Transform`` messages."""
    return dist_vector3(transform_a.translation, transform_b.translation)


def angular_distance_quat(quaternion_a: Quaternion, quaternion_b: Quaternion) -> float:
    """Return the shortest angular distance between two quaternions in radians."""
    quat_a = conv.quat_to_np(quaternion_a)
    quat_b = conv.quat_to_np(quaternion_b)
    dot = abs(float(np.dot(quat_a, quat_b)))
    return float(2.0 * math.acos(np.clip(dot, -1.0, 1.0)))


def lerp_float(value_a: float, value_b: float, weight: float = 0.5) -> float:
    """Linearly interpolate between two numeric values."""
    return float(value_a) + (float(value_b) - float(value_a)) * float(weight)


def lerp_point(point_a: Point, point_b: Point, weight: float = 0.5) -> Point:
    """Linearly interpolate between two ``geometry_msgs/Point`` messages."""
    delta = conv.point_to_np(point_b) - conv.point_to_np(point_a)
    return conv.np_to_point(conv.point_to_np(point_a) + delta * float(weight))


def lerp_vector3(vector_a: Vector3, vector_b: Vector3, weight: float = 0.5) -> Vector3:
    """Linearly interpolate between two ``geometry_msgs/Vector3`` messages."""
    delta = conv.vector3_to_np(vector_b) - conv.vector3_to_np(vector_a)
    return conv.np_to_vector3(conv.vector3_to_np(vector_a) + delta * float(weight))


def lerp_quat(
    quaternion_a: Quaternion,
    quaternion_b: Quaternion,
    weight: float = 0.5,
) -> Quaternion:
    """Linearly interpolate between two quaternions and normalize the result."""
    quat_a = conv.quat_to_np(quaternion_a)
    quat_b = conv.quat_to_np(quaternion_b)
    if np.dot(quat_a, quat_b) < 0.0:
        quat_b = -quat_b
    return conv.np_to_quat(quat_a + (quat_b - quat_a) * float(weight))


def slerp_quat(
    quaternion_a: Quaternion,
    quaternion_b: Quaternion,
    weight: float = 0.5,
) -> Quaternion:
    """Spherically interpolate between two quaternions."""
    rotations = Rotation.from_quat([conv.quat_to_np(quaternion_a), conv.quat_to_np(quaternion_b)])
    return conv.np_to_quat(Slerp([0.0, 1.0], rotations)([float(weight)]))


def lerp_pose(pose_a: Pose, pose_b: Pose, weight: float = 0.5, slerp: bool = False) -> Pose:
    """Interpolate between two poses, optionally using spherical rotation interpolation."""
    pose = Pose()
    pose.position = lerp_point(pose_a.position, pose_b.position, weight)
    pose.orientation = (
        slerp_quat(pose_a.orientation, pose_b.orientation, weight)
        if slerp
        else lerp_quat(pose_a.orientation, pose_b.orientation, weight)
    )
    return pose


def lerp_transform(
    transform_a: Transform,
    transform_b: Transform,
    weight: float = 0.5,
    slerp: bool = False,
) -> Transform:
    """Interpolate between two transforms, optionally using spherical rotation interpolation."""
    transform = Transform()
    transform.translation = lerp_vector3(transform_a.translation, transform_b.translation, weight)
    transform.rotation = (
        slerp_quat(transform_a.rotation, transform_b.rotation, weight)
        if slerp
        else lerp_quat(transform_a.rotation, transform_b.rotation, weight)
    )
    return transform


def norm_point(point: Point) -> float:
    """Return the Euclidean norm of a ``geometry_msgs/Point``."""
    return float(np.linalg.norm(conv.point_to_np(point)))


def norm_vector3(vector: Vector3) -> float:
    """Return the Euclidean norm of a ``geometry_msgs/Vector3``."""
    return float(np.linalg.norm(conv.vector3_to_np(vector)))


def norm_quat(quaternion: Quaternion) -> float:
    """Return the Euclidean norm of a ``geometry_msgs/Quaternion``."""
    values = np.array([quaternion.x, quaternion.y, quaternion.z, quaternion.w], dtype=float)
    return float(np.linalg.norm(values))


def normalise_quat(quaternion: Quaternion) -> Quaternion:
    """Return a normalized copy of a ``geometry_msgs/Quaternion``."""
    return conv.np_to_quat([quaternion.x, quaternion.y, quaternion.z, quaternion.w])


def normalise_pose(pose: Pose) -> Pose:
    """Return a copy of ``pose`` with a normalized orientation quaternion."""
    normalized = Pose()
    normalized.position = conv.np_to_point(conv.point_to_np(pose.position))
    normalized.orientation = normalise_quat(pose.orientation)
    return normalized


def normalise_transform(transform: Transform) -> Transform:
    """Return a copy of ``transform`` with a normalized rotation quaternion."""
    normalized = Transform()
    normalized.translation = conv.np_to_vector3(conv.vector3_to_np(transform.translation))
    normalized.rotation = normalise_quat(transform.rotation)
    return normalized


def normalise_pose_stamped(pose_stamped: PoseStamped) -> PoseStamped:
    """Return a stamped pose with a normalized orientation quaternion."""
    normalized = PoseStamped()
    normalized.header = pose_stamped.header
    normalized.pose = normalise_pose(pose_stamped.pose)
    return normalized


def normalise_transform_stamped(transform_stamped: TransformStamped) -> TransformStamped:
    """Return a stamped transform with a normalized rotation quaternion."""
    normalized = TransformStamped()
    normalized.header = transform_stamped.header
    normalized.child_frame_id = transform_stamped.child_frame_id
    normalized.transform = normalise_transform(transform_stamped.transform)
    return normalized


def transform_point(
    point: Point,
    transform: Pose | Transform | Sequence[Sequence[float]] | np.ndarray,
) -> Point:
    """Apply a pose, transform, or homogeneous matrix to a point."""
    if isinstance(transform, Pose):
        matrix = conv.pose_to_np(transform)
    elif isinstance(transform, Transform):
        matrix = conv.transform_to_np(transform)
    else:
        matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), but has shape {matrix.shape}")
    homogeneous = np.append(conv.point_to_np(point), 1.0)
    return conv.np_to_point((matrix @ homogeneous)[:3])


def transform_vector3(
    vector: Vector3,
    transform: Pose | Transform | Sequence[Sequence[float]] | np.ndarray,
) -> Vector3:
    """Apply only the rotational part of a pose, transform, or homogeneous matrix to a vector."""
    if isinstance(transform, Pose):
        matrix = conv.pose_to_np(transform)
    elif isinstance(transform, Transform):
        matrix = conv.transform_to_np(transform)
    else:
        matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"transform must have shape (4, 4), but has shape {matrix.shape}")
    return conv.np_to_vector3(matrix[:3, :3] @ conv.vector3_to_np(vector))
