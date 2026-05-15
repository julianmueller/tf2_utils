"""Example transforms showing compact TF tree generation helpers."""

from __future__ import annotations

from geometry_msgs.msg import Transform, TransformStamped, Vector3

from tf2_utils.conversions import euler_to_quat
from tf2_utils.tree import flatten_transform_tree, transform_tree


ROOT_FRAME = "generated_tree"


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
    """Create a compact nested tree for visual inspection in RViz."""
    return transform_tree(
        ROOT_FRAME,
        children=[
            transform_tree(
                "generated_robot_cell",
                _transform(-1.2, -0.8, 0.0, yaw=0.25),
                children=[
                    transform_tree(
                        "generated_robot_base",
                        _transform(0.0, 0.0, 0.35),
                        children=[
                            transform_tree(
                                "generated_arm_link_01",
                                _transform(0.45, 0.0, 0.55, pitch=-0.35, yaw=0.2),
                                children=[
                                    transform_tree(
                                        "generated_arm_link_02",
                                        _transform(0.55, 0.2, 0.35, roll=0.4, pitch=0.2),
                                        children=[
                                            transform_tree(
                                                "generated_tool_flange",
                                                _transform(0.32, -0.1, 0.18, roll=0.15, yaw=-0.35),
                                            ),
                                        ],
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            transform_tree(
                "generated_fixture_cell",
                _transform(1.15, 0.75, 0.0, yaw=-0.45),
                children=[
                    transform_tree(
                        "generated_fixture_left",
                        _transform(-0.35, 0.45, 0.45, roll=0.25, yaw=0.4),
                    ),
                    transform_tree(
                        "generated_fixture_right",
                        _transform(0.4, -0.35, 0.5, pitch=0.35, yaw=-0.3),
                    ),
                    transform_tree(
                        "generated_part_nest",
                        _transform(0.05, 0.05, 0.78, roll=-0.15, pitch=0.2, yaw=0.7),
                    ),
                ],
            ),
            transform_tree(
                "generated_sensor_cell",
                _transform(-0.25, 1.3, 1.0, roll=-0.2, yaw=1.1),
                children=[
                    transform_tree(
                        "generated_camera_mount",
                        _transform(0.75, 0.0, 0.25, pitch=-0.45),
                        children=[
                            transform_tree(
                                "generated_camera_optical",
                                _transform(0.12, 0.0, 0.18, roll=-1.57, pitch=0.0, yaw=-1.57),
                            ),
                        ],
                    ),
                    transform_tree(
                        "generated_light_bar",
                        _transform(0.25, -0.45, 0.1, roll=0.15, yaw=-0.25),
                    ),
                ],
            ),
        ],
    )


def _tree(parent_frame: str) -> list[TransformStamped]:
    """Create stamped transforms from the compact tree."""
    return flatten_transform_tree(parent_frame, _tree_node())


def static(parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Create static transforms for the generated tree example."""
    return [
        (transform.header.frame_id, transform.child_frame_id, transform.transform) for transform in _tree(parent_frame)
    ]


def create(parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Create dynamic transforms for the generated tree example."""
    del parent_frame
    return []
