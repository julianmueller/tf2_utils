"""Example transforms showing how to reparent TF frames without moving them globally."""

from __future__ import annotations

from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3

from tf2_utils.reparenting import reparent_transform_stamped
from tf2_utils.tree import flatten_transform_tree, transform_tree
from tf2_utils.conversions import euler_to_quat


ROOT_FRAME = "world"
ROBOT_BASE_FRAME = "robot_base"
SHOULDER_FRAME = "shoulder"
WRIST_FRAME = "wrist"
FIXTURE_BASE_FRAME = "fixture_base"
INSPECTION_MOUNT_FRAME = "inspection_mount"
STORAGE_MOUNT_FRAME = "storage_mount"
CAMERA_RAIL_FRAME = "camera_rail"
CAMERA_MOUNT_FRAME = "camera_mount"


def _transform(
    x: float,
    y: float,
    z: float,
    roll: float = 0.0,
    pitch: float = 0.0,
    yaw: float = 0.0,
) -> Transform:
    """Create a transform with translation and Euler orientation."""
    return Transform(
        translation=Vector3(x=x, y=y, z=z),
        rotation=euler_to_quat(roll, pitch, yaw),
    )


def _tree_node() -> tuple[str, Transform, tuple]:
    """Create a compact branched tree with robot, fixture, and camera branches."""
    return transform_tree(
        ROOT_FRAME,
        children=[
            transform_tree(
                ROBOT_BASE_FRAME,
                _transform(-1.0, -0.9, 0.0, yaw=0.25),
                children=[
                    transform_tree(
                        SHOULDER_FRAME,
                        _transform(0.45, 0.0, 0.75, pitch=-0.45, yaw=0.25),
                        children=[
                            transform_tree(
                                WRIST_FRAME,
                                _transform(0.55, 0.25, 0.45, roll=0.55, pitch=0.25, yaw=-0.35),
                            ),
                        ],
                    ),
                ],
            ),
            transform_tree(
                FIXTURE_BASE_FRAME,
                _transform(1.25, 0.75, 0.0, yaw=-0.55),
                children=[
                    transform_tree(
                        INSPECTION_MOUNT_FRAME,
                        _transform(-0.25, 0.55, 0.65, roll=0.25, yaw=0.8),
                    ),
                    transform_tree(
                        STORAGE_MOUNT_FRAME,
                        _transform(0.5, -0.55, 0.45, pitch=0.35, yaw=-0.35),
                    ),
                ],
            ),
            transform_tree(
                CAMERA_RAIL_FRAME,
                _transform(-0.35, 1.25, 1.15, roll=-0.25, yaw=1.2),
                children=[
                    transform_tree(
                        CAMERA_MOUNT_FRAME,
                        _transform(0.8, 0.0, 0.25, pitch=-0.3, yaw=-0.25),
                    ),
                ],
            ),
        ],
    )


def _tree(parent_frame: str) -> list[TransformStamped]:
    """Create parent-to-child edges from the compact example tree."""
    return flatten_transform_tree(parent_frame, _tree_node())


def _tool_transform(parent_frame: str, child_frame: str) -> TransformStamped:
    """Create the local wrist-to-tool transform used for original and reparented frames."""
    return flatten_transform_tree(
        parent_frame,
        transform_tree(child_frame, _transform(0.35, 0.0, 0.18, roll=0.1, pitch=-0.25, yaw=0.4)),
    )[0]


def _sensor_transform(parent_frame: str, child_frame: str) -> TransformStamped:
    """Create the local camera-mount-to-sensor transform used for a second reparenting case."""
    return flatten_transform_tree(
        parent_frame,
        transform_tree(child_frame, _transform(0.15, -0.12, 0.32, roll=-0.45, pitch=0.15, yaw=-0.2)),
    )[0]


def static(parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Create the static source tree and original child frames."""
    tree = _tree(parent_frame)
    originals = [
        _tool_transform(WRIST_FRAME, "original_tool"),
        _sensor_transform(CAMERA_MOUNT_FRAME, "original_sensor"),
    ]
    return [
        (transform.header.frame_id, transform.child_frame_id, transform.transform) for transform in [*tree, *originals]
    ]


def create(parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Create reparented transforms that keep their original global poses."""
    tree = _tree(parent_frame)
    tool_to_reparent = _tool_transform(WRIST_FRAME, "reparented_tool")
    sensor_to_reparent = _sensor_transform(CAMERA_MOUNT_FRAME, "reparented_sensor")
    reparented_tool = reparent_transform_stamped(
        tool_to_reparent,
        INSPECTION_MOUNT_FRAME,
        tree,
        root_frame=ROOT_FRAME,
    )
    reparented_sensor = reparent_transform_stamped(
        sensor_to_reparent,
        STORAGE_MOUNT_FRAME,
        tree,
        root_frame=ROOT_FRAME,
    )
    return [
        (reparented_tool.header.frame_id, reparented_tool.child_frame_id, reparented_tool.transform),
        (reparented_sensor.header.frame_id, reparented_sensor.child_frame_id, reparented_sensor.transform),
    ]
