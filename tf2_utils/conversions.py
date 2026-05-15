"""Conversions between ROS 2 transform messages, NumPy arrays, and lists."""

from __future__ import annotations

from collections.abc import Sequence
import math
import numpy as np
from scipy.spatial.transform import Rotation

from rclpy.clock import Clock
from rclpy.time import Time as RclpyTime
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion, Transform, TransformStamped, Vector3
from std_msgs.msg import Header


__all__ = [
    "euler_to_quat",
    "get_timestamp",
    "list_to_point",
    "list_to_pose",
    "list_to_quat",
    "list_to_transform",
    "list_to_vector3",
    "np_to_point",
    "np_to_pose",
    "np_to_quat",
    "np_to_transform",
    "np_to_vector3",
    "point_to_list",
    "point_to_np",
    "point_to_str",
    "point_to_vector3",
    "pose_stamped_to_np",
    "pose_stamped_to_transform_stamped",
    "pose_to_list",
    "pose_to_np",
    "pose_to_str",
    "pose_to_transform",
    "posestamped_to_str",
    "quat_to_euler",
    "quat_to_list",
    "quat_to_matrix",
    "quat_to_np",
    "quat_to_rot_matrix",
    "quat_to_str",
    "rot_matrix_to_quat",
    "stamp_pose",
    "stamp_transform",
    "transform_stamped_to_np",
    "transform_stamped_to_pose_stamped",
    "transform_to_list",
    "transform_to_np",
    "transform_to_pose",
    "transform_to_str",
    "transformstamped_to_str",
    "vector3_to_list",
    "vector3_to_np",
    "vector3_to_point",
    "vector3_to_str",
]


def _as_vector(values: Sequence[float] | np.ndarray, length: int, name: str) -> np.ndarray:
    """Return ``values`` as a one-dimensional float array of ``length`` entries."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape != (length,):
        raise ValueError(f"{name} must contain {length} values, but has shape {np.asarray(values).shape}")
    return array


def _as_matrix(
    values: Sequence[Sequence[float]] | np.ndarray,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """Return ``values`` as a float matrix with the requested ``shape``."""
    matrix = np.asarray(values, dtype=float)
    if matrix.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, but has shape {matrix.shape}")
    return matrix


def get_timestamp(stamp: Time | RclpyTime | None, clock: Clock | None) -> Time:
    """Convert an optional ROS 2 time value into a ``builtin_interfaces/Time`` message."""
    if stamp is None:
        return (clock if clock is not None else Clock()).now().to_msg()
    if isinstance(stamp, RclpyTime):
        return stamp.to_msg()
    if isinstance(stamp, Time):
        return stamp
    raise TypeError(f"stamp is not a builtin_interfaces/Time or rclpy.time.Time, but a {type(stamp).__name__}")


def _normalised_quat_values(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return normalized xyzw quaternion values, using identity for zero-length input."""
    array = _as_vector(values, 4, "values")
    if math.isclose(np.linalg.norm(array), 0.0):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    return array / np.linalg.norm(array)


def point_to_str(point: Point) -> str:
    """Return a compact string representation of a ``geometry_msgs/Point``."""
    return f"geometry_msgs/Point [x: {point.x:.5g}, y: {point.y:.5g}, z: {point.z:.5g}]"


def vector3_to_str(vector: Vector3) -> str:
    """Return a compact string representation of a ``geometry_msgs/Vector3``."""
    return f"geometry_msgs/Vector3 [x: {vector.x:.5g}, y: {vector.y:.5g}, z: {vector.z:.5g}]"


def quat_to_str(quaternion: Quaternion, euler: bool = False, degrees: bool = False) -> str:
    """Return a compact string representation of a ``geometry_msgs/Quaternion``."""
    if euler:
        roll, pitch, yaw = quat_to_euler(quaternion, degrees=degrees)
        unit = "deg" if degrees else "rad"
        return (
            f"geometry_msgs/Quaternion [roll: {roll:.3g}{unit}, pitch: {pitch:.3g}{unit}, yaw: {yaw:.3g}{unit}] (euler)"
        )
    return (
        "geometry_msgs/Quaternion "
        f"[x: {quaternion.x:.5g}, y: {quaternion.y:.5g}, "
        f"z: {quaternion.z:.5g}, w: {quaternion.w:.5g}]"
    )


def pose_to_str(pose: Pose) -> str:
    """Return a compact string representation of a ``geometry_msgs/Pose``."""
    point = pose.position
    quaternion = pose.orientation
    return (
        "geometry_msgs/Pose "
        f"[px: {point.x:.5g}, py: {point.y:.5g}, pz: {point.z:.5g}, "
        f"qx: {quaternion.x:.5g}, qy: {quaternion.y:.5g}, "
        f"qz: {quaternion.z:.5g}, qw: {quaternion.w:.5g}]"
    )


def posestamped_to_str(pose_stamped: PoseStamped) -> str:
    """Return a compact string representation of a ``geometry_msgs/PoseStamped``."""
    point = pose_stamped.pose.position
    quaternion = pose_stamped.pose.orientation
    return (
        "geometry_msgs/PoseStamped "
        f"[px: {point.x:.5g}, py: {point.y:.5g}, pz: {point.z:.5g}, "
        f"qx: {quaternion.x:.5g}, qy: {quaternion.y:.5g}, "
        f"qz: {quaternion.z:.5g}, qw: {quaternion.w:.5g}, "
        f"frame: {pose_stamped.header.frame_id}]"
    )


def transform_to_str(transform: Transform) -> str:
    """Return a compact string representation of a ``geometry_msgs/Transform``."""
    vector = transform.translation
    quaternion = transform.rotation
    return (
        "geometry_msgs/Transform "
        f"[tx: {vector.x:.5g}, ty: {vector.y:.5g}, tz: {vector.z:.5g}, "
        f"qx: {quaternion.x:.5g}, qy: {quaternion.y:.5g}, "
        f"qz: {quaternion.z:.5g}, qw: {quaternion.w:.5g}]"
    )


def transformstamped_to_str(transform_stamped: TransformStamped) -> str:
    """Return a compact string representation of a ``geometry_msgs/TransformStamped``."""
    vector = transform_stamped.transform.translation
    quaternion = transform_stamped.transform.rotation
    return (
        "geometry_msgs/TransformStamped "
        f"[tx: {vector.x:.5g}, ty: {vector.y:.5g}, tz: {vector.z:.5g}, "
        f"qx: {quaternion.x:.5g}, qy: {quaternion.y:.5g}, "
        f"qz: {quaternion.z:.5g}, qw: {quaternion.w:.5g}, "
        f"frame: {transform_stamped.header.frame_id}, "
        f"child_frame: {transform_stamped.child_frame_id}]"
    )


def euler_to_quat(roll: float, pitch: float, yaw: float, degrees: bool = False) -> Quaternion:
    """Convert roll, pitch, and yaw Euler angles into a normalized quaternion."""
    quaternion = Rotation.from_euler("xyz", [roll, pitch, yaw], degrees=degrees).as_quat()
    return np_to_quat(quaternion)


def quat_to_euler(
    quaternion: Quaternion | Sequence[float] | np.ndarray | float,
    qy: float | None = None,
    qz: float | None = None,
    qw: float | None = None,
    degrees: bool = False,
) -> list[float]:
    """Convert a quaternion into ``[roll, pitch, yaw]`` Euler angles."""
    if isinstance(quaternion, Quaternion):
        values = quat_to_np(quaternion)
    elif qy is not None and qz is not None and qw is not None:
        values = _normalised_quat_values([float(quaternion), float(qy), float(qz), float(qw)])
    else:
        values = _normalised_quat_values(quaternion)
    return Rotation.from_quat(values).as_euler("xyz", degrees=degrees).tolist()


def point_to_np(point: Point) -> np.ndarray:
    """Convert a ``geometry_msgs/Point`` into a NumPy xyz vector with shape ``(3,)``."""
    return np.array([point.x, point.y, point.z], dtype=float)


def np_to_point(vector: Sequence[float] | np.ndarray) -> Point:
    """Convert a NumPy xyz vector into a ``geometry_msgs/Point``."""
    x, y, z = _as_vector(vector, 3, "vector")
    return Point(x=x, y=y, z=z)


def vector3_to_np(vector: Vector3) -> np.ndarray:
    """Convert a ``geometry_msgs/Vector3`` into a NumPy xyz vector with shape ``(3,)``."""
    return np.array([vector.x, vector.y, vector.z], dtype=float)


def np_to_vector3(vector: Sequence[float] | np.ndarray) -> Vector3:
    """Convert a NumPy xyz vector into a ``geometry_msgs/Vector3``."""
    x, y, z = _as_vector(vector, 3, "vector")
    return Vector3(x=x, y=y, z=z)


def quat_to_np(quaternion: Quaternion) -> np.ndarray:
    """Convert a ``geometry_msgs/Quaternion`` into a normalized xyzw NumPy vector."""
    return _normalised_quat_values([quaternion.x, quaternion.y, quaternion.z, quaternion.w])


def np_to_quat(values: Sequence[float] | np.ndarray | Rotation) -> Quaternion:
    """Convert a quaternion vector, matrix, or SciPy rotation into a quaternion."""
    if isinstance(values, Rotation):
        quaternion = values.as_quat()
        if quaternion.ndim == 2:
            if quaternion.shape[0] != 1:
                raise ValueError(f"rotation must contain exactly one quaternion, but contains {quaternion.shape[0]}")
            quaternion = quaternion[0]
        x, y, z, w = _normalised_quat_values(quaternion)
        return Quaternion(x=x, y=y, z=z, w=w)

    array = np.asarray(values, dtype=float)
    if array.shape == (3, 3):
        x, y, z, w = Rotation.from_matrix(array).as_quat()
        return Quaternion(x=x, y=y, z=z, w=w)
    x, y, z, w = _normalised_quat_values(array)
    return Quaternion(x=x, y=y, z=z, w=w)


def quat_to_matrix(quaternion: Quaternion) -> np.ndarray:
    """Convert a ``geometry_msgs/Quaternion`` into a 3x3 rotation matrix."""
    return Rotation.from_quat(quat_to_np(quaternion)).as_matrix()


def quat_to_rot_matrix(quaternion: Quaternion) -> np.ndarray:
    """Convert a ``geometry_msgs/Quaternion`` into a 3x3 rotation matrix."""
    return quat_to_matrix(quaternion)


def rot_matrix_to_quat(matrix: Sequence[Sequence[float]] | np.ndarray) -> Quaternion:
    """Convert a 3x3 rotation matrix into a normalized ``geometry_msgs/Quaternion``."""
    return np_to_quat(_as_matrix(matrix, (3, 3), "matrix"))


def pose_to_np(pose: Pose) -> np.ndarray:
    """Convert a ``geometry_msgs/Pose`` into a 4x4 homogeneous transform matrix."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quat_to_matrix(pose.orientation)
    matrix[:3, 3] = point_to_np(pose.position)
    return matrix


def np_to_pose(matrix: Sequence[Sequence[float]] | np.ndarray) -> Pose:
    """Convert a 4x4 homogeneous transform matrix into a ``geometry_msgs/Pose``."""
    transform = _as_matrix(matrix, (4, 4), "matrix")
    pose = Pose()
    pose.position = np_to_point(transform[:3, 3])
    pose.orientation = np_to_quat(transform[:3, :3])
    return pose


def transform_to_np(transform: Transform) -> np.ndarray:
    """Convert a ``geometry_msgs/Transform`` into a 4x4 homogeneous transform matrix."""
    matrix = np.eye(4, dtype=float)
    matrix[:3, :3] = quat_to_matrix(transform.rotation)
    matrix[:3, 3] = vector3_to_np(transform.translation)
    return matrix


def np_to_transform(matrix: Sequence[Sequence[float]] | np.ndarray) -> Transform:
    """Convert a 4x4 homogeneous transform matrix into a ``geometry_msgs/Transform``."""
    array = _as_matrix(matrix, (4, 4), "matrix")
    transform = Transform()
    transform.translation = np_to_vector3(array[:3, 3])
    transform.rotation = np_to_quat(array[:3, :3])
    return transform


def pose_stamped_to_np(pose_stamped: PoseStamped) -> np.ndarray:
    """Convert the pose inside a ``geometry_msgs/PoseStamped`` into a 4x4 matrix."""
    return pose_to_np(pose_stamped.pose)


def transform_stamped_to_np(transform_stamped: TransformStamped) -> np.ndarray:
    """Convert the transform inside a ``geometry_msgs/TransformStamped`` into a 4x4 matrix."""
    return transform_to_np(transform_stamped.transform)


def point_to_vector3(point: Point) -> Vector3:
    """Convert a ``geometry_msgs/Point`` into a ``geometry_msgs/Vector3``."""
    return np_to_vector3(point_to_np(point))


def vector3_to_point(vector: Vector3) -> Point:
    """Convert a ``geometry_msgs/Vector3`` into a ``geometry_msgs/Point``."""
    return np_to_point(vector3_to_np(vector))


def pose_to_transform(pose: Pose) -> Transform:
    """Convert a ``geometry_msgs/Pose`` into a ``geometry_msgs/Transform``."""
    transform = Transform()
    transform.translation = point_to_vector3(pose.position)
    transform.rotation = np_to_quat(quat_to_np(pose.orientation))
    return transform


def transform_to_pose(transform: Transform) -> Pose:
    """Convert a ``geometry_msgs/Transform`` into a ``geometry_msgs/Pose``."""
    pose = Pose()
    pose.position = vector3_to_point(transform.translation)
    pose.orientation = np_to_quat(quat_to_np(transform.rotation))
    return pose


def transform_stamped_to_pose_stamped(transform_stamped: TransformStamped) -> PoseStamped:
    """Convert a stamped transform into a stamped pose in the parent frame."""
    pose_stamped = PoseStamped()
    pose_stamped.header = transform_stamped.header
    pose_stamped.pose = transform_to_pose(transform_stamped.transform)
    return pose_stamped


def pose_stamped_to_transform_stamped(
    pose_stamped: PoseStamped,
    child_frame: str,
) -> TransformStamped:
    """Convert a stamped pose into a stamped transform with ``child_frame``."""
    transform_stamped = TransformStamped()
    transform_stamped.header = pose_stamped.header
    transform_stamped.child_frame_id = child_frame
    transform_stamped.transform = pose_to_transform(pose_stamped.pose)
    return transform_stamped


def point_to_list(point: Point) -> list[float]:
    """Convert a ``geometry_msgs/Point`` into ``[x, y, z]``."""
    return point_to_np(point).tolist()


def list_to_point(values: Sequence[float] | np.ndarray) -> Point:
    """Convert ``[x, y, z]`` into a ``geometry_msgs/Point``."""
    return np_to_point(values)


def vector3_to_list(vector: Vector3) -> list[float]:
    """Convert a ``geometry_msgs/Vector3`` into ``[x, y, z]``."""
    return vector3_to_np(vector).tolist()


def list_to_vector3(values: Sequence[float] | np.ndarray) -> Vector3:
    """Convert ``[x, y, z]`` into a ``geometry_msgs/Vector3``."""
    return np_to_vector3(values)


def quat_to_list(quaternion: Quaternion) -> list[float]:
    """Convert a ``geometry_msgs/Quaternion`` into ``[x, y, z, w]``."""
    return quat_to_np(quaternion).tolist()


def list_to_quat(values: Sequence[float] | np.ndarray) -> Quaternion:
    """Convert ``[x, y, z, w]`` into a normalized ``geometry_msgs/Quaternion``."""
    return np_to_quat(values)


def pose_to_list(pose: Pose) -> list[float]:
    """Convert a ``geometry_msgs/Pose`` into ``[x, y, z, qx, qy, qz, qw]``."""
    return point_to_list(pose.position) + quat_to_list(pose.orientation)


def list_to_pose(values: Sequence[float] | np.ndarray) -> Pose:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` into a ``geometry_msgs/Pose``."""
    array = _as_vector(values, 7, "values")
    pose = Pose()
    pose.position = np_to_point(array[:3])
    pose.orientation = np_to_quat(array[3:])
    return pose


def transform_to_list(transform: Transform) -> list[float]:
    """Convert a ``geometry_msgs/Transform`` into ``[x, y, z, qx, qy, qz, qw]``."""
    return vector3_to_list(transform.translation) + quat_to_list(transform.rotation)


def list_to_transform(values: Sequence[float] | np.ndarray) -> Transform:
    """Convert ``[x, y, z, qx, qy, qz, qw]`` into a ``geometry_msgs/Transform``."""
    array = _as_vector(values, 7, "values")
    transform = Transform()
    transform.translation = np_to_vector3(array[:3])
    transform.rotation = np_to_quat(array[3:])
    return transform


def stamp_pose(
    pose: Pose,
    frame: str = "world",
    stamp: Time | RclpyTime | None = None,
    clock: Clock | None = None,
) -> PoseStamped:
    """Wrap a ``geometry_msgs/Pose`` in a ``geometry_msgs/PoseStamped``."""
    pose_stamped = PoseStamped()
    pose_stamped.header = Header(frame_id=frame, stamp=get_timestamp(stamp, clock))
    pose_stamped.pose = pose
    return pose_stamped


def stamp_transform(
    transform: Transform,
    child_frame: str,
    frame: str = "world",
    stamp: Time | RclpyTime | None = None,
    clock: Clock | None = None,
) -> TransformStamped:
    """Wrap a ``geometry_msgs/Transform`` in a ``geometry_msgs/TransformStamped``."""
    transform_stamped = TransformStamped()
    transform_stamped.header = Header(frame_id=frame, stamp=get_timestamp(stamp, clock))
    transform_stamped.transform = transform
    transform_stamped.child_frame_id = child_frame
    return transform_stamped
