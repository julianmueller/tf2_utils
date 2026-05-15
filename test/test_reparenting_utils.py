import math

import pytest
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from std_msgs.msg import Header

from tf2_utils import calculations as calc
from tf2_utils import reparenting as reparent


TESTED_METHODS = {
    "reparent_transform_stamped",
}


@pytest.fixture
def yaw_90_quaternion():
    return Quaternion(x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5))


def test_all_public_reparenting_methods_are_covered():
    assert set(reparent.__all__) == TESTED_METHODS


def test_reparent_transform_stamped_preserves_root_pose(yaw_90_quaternion):
    world_to_base = TransformStamped(
        header=Header(frame_id="world"),
        child_frame_id="base",
        transform=Transform(
            translation=Vector3(x=1.0, y=0.0, z=0.0),
            rotation=Quaternion(),
        ),
    )
    base_to_arm = TransformStamped(
        header=Header(frame_id="base"),
        child_frame_id="arm",
        transform=Transform(
            translation=Vector3(x=0.0, y=2.0, z=0.0),
            rotation=yaw_90_quaternion,
        ),
    )
    world_to_table = TransformStamped(
        header=Header(frame_id="world"),
        child_frame_id="table",
        transform=Transform(
            translation=Vector3(x=4.0, y=0.0, z=0.0),
            rotation=Quaternion(),
        ),
    )
    table_to_fixture = TransformStamped(
        header=Header(frame_id="table"),
        child_frame_id="fixture",
        transform=Transform(
            translation=Vector3(x=0.0, y=1.0, z=0.0),
            rotation=Quaternion(),
        ),
    )
    arm_to_camera = TransformStamped(
        header=Header(frame_id="arm"),
        child_frame_id="camera",
        transform=Transform(
            translation=Vector3(x=0.0, y=0.0, z=3.0),
            rotation=Quaternion(),
        ),
    )

    fixture_to_camera = reparent.reparent_transform_stamped(
        arm_to_camera,
        "fixture",
        [world_to_base, base_to_arm, world_to_table, table_to_fixture],
        root_frame="world",
    )
    original_world_to_camera = calc.chain_transform_stamped(
        world_to_base,
        base_to_arm,
        arm_to_camera,
    )
    reparented_world_to_camera = calc.chain_transform_stamped(
        world_to_table,
        table_to_fixture,
        fixture_to_camera,
    )

    assert fixture_to_camera.header.frame_id == "fixture"
    assert fixture_to_camera.child_frame_id == "camera"
    assert [
        reparented_world_to_camera.transform.translation.x,
        reparented_world_to_camera.transform.translation.y,
        reparented_world_to_camera.transform.translation.z,
    ] == pytest.approx(
        [
            original_world_to_camera.transform.translation.x,
            original_world_to_camera.transform.translation.y,
            original_world_to_camera.transform.translation.z,
        ]
    )
    assert [
        reparented_world_to_camera.transform.rotation.x,
        reparented_world_to_camera.transform.rotation.y,
        reparented_world_to_camera.transform.rotation.z,
        reparented_world_to_camera.transform.rotation.w,
    ] == pytest.approx(
        [
            original_world_to_camera.transform.rotation.x,
            original_world_to_camera.transform.rotation.y,
            original_world_to_camera.transform.rotation.z,
            original_world_to_camera.transform.rotation.w,
        ]
    )


def test_reparent_transform_stamped_rejects_invalid_trees():
    world_to_parent = TransformStamped(
        header=Header(frame_id="world"),
        child_frame_id="parent",
        transform=Transform(rotation=Quaternion()),
    )
    parent_to_child = TransformStamped(
        header=Header(frame_id="parent"),
        child_frame_id="child",
        transform=Transform(rotation=Quaternion()),
    )
    child_to_descendant = TransformStamped(
        header=Header(frame_id="child"),
        child_frame_id="descendant",
        transform=Transform(rotation=Quaternion()),
    )

    with pytest.raises(ValueError):
        reparent.reparent_transform_stamped(parent_to_child, "missing", [world_to_parent])
    with pytest.raises(ValueError):
        reparent.reparent_transform_stamped(parent_to_child, "descendant", [world_to_parent, child_to_descendant])
    with pytest.raises(ValueError):
        reparent.reparent_transform_stamped(parent_to_child, "child", [world_to_parent])
