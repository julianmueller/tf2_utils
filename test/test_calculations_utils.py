import math

import numpy as np
import pytest
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from std_msgs.msg import Header

from tf2_utils import calculations as calc


TESTED_METHODS = {
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
    "invert_quat",
    "invert_transform",
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
}


@pytest.fixture
def point_a():
    return Point(x=1.0, y=2.0, z=3.0)


@pytest.fixture
def point_b():
    return Point(x=4.0, y=6.0, z=3.0)


@pytest.fixture
def vector_a():
    return Vector3(x=1.0, y=2.0, z=3.0)


@pytest.fixture
def vector_b():
    return Vector3(x=4.0, y=6.0, z=3.0)


@pytest.fixture
def yaw_90_quaternion():
    return Quaternion(x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5))


@pytest.fixture
def identity_quaternion():
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


@pytest.fixture
def pose_a(yaw_90_quaternion):
    return Pose(position=Point(x=1.0, y=0.0, z=0.0), orientation=yaw_90_quaternion)


@pytest.fixture
def pose_b(identity_quaternion):
    return Pose(position=Point(x=1.0, y=0.0, z=0.0), orientation=identity_quaternion)


@pytest.fixture
def transform_a(yaw_90_quaternion):
    return Transform(translation=Vector3(x=1.0, y=0.0, z=0.0), rotation=yaw_90_quaternion)


@pytest.fixture
def transform_b(identity_quaternion):
    return Transform(translation=Vector3(x=1.0, y=0.0, z=0.0), rotation=identity_quaternion)


@pytest.fixture
def known_composed_pose(yaw_90_quaternion):
    return Pose(position=Point(x=1.0, y=1.0, z=0.0), orientation=yaw_90_quaternion)


@pytest.fixture
def known_composed_transform(yaw_90_quaternion):
    return Transform(translation=Vector3(x=1.0, y=1.0, z=0.0), rotation=yaw_90_quaternion)


def test_all_public_calculation_methods_are_covered():
    assert set(calc.__all__) == TESTED_METHODS


def test_add_scale_norm_and_distance_helpers(point_a, point_b, vector_a, vector_b):
    added_point = calc.add_points(point_a, point_b)
    added_vector = calc.add_vector3s(vector_a, vector_b)
    scaled_point = calc.scale_point(point_a, 2.0)
    scaled_vector = calc.scale_vector3(vector_a, 2.0)

    assert [added_point.x, added_point.y, added_point.z] == pytest.approx([5.0, 8.0, 6.0])
    assert [added_vector.x, added_vector.y, added_vector.z] == pytest.approx([5.0, 8.0, 6.0])
    assert [scaled_point.x, scaled_point.y, scaled_point.z] == pytest.approx([2.0, 4.0, 6.0])
    assert [scaled_vector.x, scaled_vector.y, scaled_vector.z] == pytest.approx([2.0, 4.0, 6.0])
    assert calc.norm_point(Point(x=3.0, y=4.0, z=0.0)) == pytest.approx(5.0)
    assert calc.norm_vector3(Vector3(x=3.0, y=4.0, z=0.0)) == pytest.approx(5.0)
    assert calc.norm_quat(Quaternion(x=0.0, y=0.0, z=0.0, w=2.0)) == pytest.approx(2.0)
    assert calc.dist_point(point_a, point_b) == pytest.approx(5.0)
    assert calc.dist_vector3(vector_a, vector_b) == pytest.approx(5.0)

    pose_a = Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(w=1.0))
    pose_b = Pose(position=Point(x=0.0, y=3.0, z=4.0), orientation=Quaternion(w=1.0))
    transform_a = Transform(translation=Vector3(x=0.0, y=0.0, z=0.0), rotation=Quaternion(w=1.0))
    transform_b = Transform(translation=Vector3(x=0.0, y=3.0, z=4.0), rotation=Quaternion(w=1.0))
    assert calc.dist_pose(pose_a, pose_b) == pytest.approx(5.0)
    assert calc.dist_transform(transform_a, transform_b) == pytest.approx(5.0)


def test_pose_and_transform_composition_helpers(
    pose_a,
    pose_b,
    transform_a,
    transform_b,
    known_composed_pose,
    known_composed_transform,
):
    composed_pose = calc.mult_poses(pose_a, pose_b)
    composed_transform = calc.mult_transforms(transform_a, transform_b)
    added_pose = calc.add_poses(pose_a, pose_b)

    assert [composed_pose.position.x, composed_pose.position.y, composed_pose.position.z] == pytest.approx(
        [
            known_composed_pose.position.x,
            known_composed_pose.position.y,
            known_composed_pose.position.z,
        ]
    )
    assert [
        composed_pose.orientation.x,
        composed_pose.orientation.y,
        composed_pose.orientation.z,
        composed_pose.orientation.w,
    ] == pytest.approx(
        [
            known_composed_pose.orientation.x,
            known_composed_pose.orientation.y,
            known_composed_pose.orientation.z,
            known_composed_pose.orientation.w,
        ]
    )
    assert [added_pose.position.x, added_pose.position.y, added_pose.position.z] == pytest.approx([2.0, 0.0, 0.0])
    assert [
        composed_transform.translation.x,
        composed_transform.translation.y,
        composed_transform.translation.z,
    ] == pytest.approx(
        [
            known_composed_transform.translation.x,
            known_composed_transform.translation.y,
            known_composed_transform.translation.z,
        ]
    )
    assert [
        calc.chain_poses(pose_a, pose_b).position.x,
        calc.chain_poses(pose_a, pose_b).position.y,
        calc.chain_poses(pose_a, pose_b).position.z,
    ] == pytest.approx([1.0, 1.0, 0.0])
    assert [
        calc.chain_transforms(transform_a, transform_b).translation.x,
        calc.chain_transforms(transform_a, transform_b).translation.y,
        calc.chain_transforms(transform_a, transform_b).translation.z,
    ] == pytest.approx([1.0, 1.0, 0.0])
    assert isinstance(calc.chain_poses(), Pose)
    assert isinstance(calc.chain_transforms(), Transform)


def test_stamped_transform_chaining_validates_connected_frames(identity_quaternion):
    world_to_base = TransformStamped(
        header=Header(frame_id="world"),
        child_frame_id="base",
        transform=Transform(
            translation=Vector3(x=1.0, y=0.0, z=0.0),
            rotation=identity_quaternion,
        ),
    )
    base_to_tool = TransformStamped(
        header=Header(frame_id="base"),
        child_frame_id="tool",
        transform=Transform(
            translation=Vector3(x=0.0, y=2.0, z=0.0),
            rotation=identity_quaternion,
        ),
    )

    result = calc.mult_transform_stamped(world_to_base, base_to_tool)
    chained = calc.chain_transform_stamped(world_to_base, base_to_tool)
    assert result.header.frame_id == "world"
    assert result.child_frame_id == "tool"
    assert [
        result.transform.translation.x,
        result.transform.translation.y,
        result.transform.translation.z,
    ] == pytest.approx([1.0, 2.0, 0.0])
    assert chained.child_frame_id == "tool"

    with pytest.raises(ValueError):
        calc.mult_transform_stamped(base_to_tool, world_to_base)
    with pytest.raises(ValueError):
        calc.chain_transform_stamped()


def test_inverse_helpers_for_matrices_poses_transforms_and_quaternions(pose_a, transform_a):
    pose_stamped = PoseStamped(header=Header(frame_id="world"), pose=pose_a)
    transform_stamped = TransformStamped(
        header=Header(frame_id="world"),
        child_frame_id="tool",
        transform=transform_a,
    )
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    inverse_matrix = calc.invert_matrix(matrix)
    inverse_pose = calc.invert_pose(pose_a)
    inverse_transform = calc.invert_transform(transform_a)
    inverse_quat = calc.invert_quat(pose_a.orientation)

    np.testing.assert_allclose(inverse_matrix @ matrix, np.eye(4), atol=1e-8)
    assert [inverse_pose.position.x, inverse_pose.position.y, inverse_pose.position.z] == pytest.approx([0.0, 1.0, 0.0])
    assert [
        inverse_transform.translation.x,
        inverse_transform.translation.y,
        inverse_transform.translation.z,
    ] == pytest.approx([0.0, 1.0, 0.0])
    assert abs(inverse_pose.orientation.z) == pytest.approx(math.sqrt(0.5))
    assert abs(inverse_pose.orientation.w) == pytest.approx(math.sqrt(0.5))
    assert inverse_pose.orientation.z * inverse_pose.orientation.w < 0.0
    assert inverse_quat.z == pytest.approx(-math.sqrt(0.5))
    assert inverse_quat.w == pytest.approx(math.sqrt(0.5))
    assert calc.invert_pose(pose_stamped).header.frame_id == "world"
    assert calc.invert_transform(transform_stamped).child_frame_id == "tool"

    with pytest.raises(ValueError):
        calc.invert_matrix(np.eye(3))


def test_interpolation_helpers(identity_quaternion):
    point_a = Point(x=0.0, y=0.0, z=0.0)
    point_b = Point(x=10.0, y=20.0, z=30.0)
    vector_a = Vector3(x=0.0, y=0.0, z=0.0)
    vector_b = Vector3(x=10.0, y=20.0, z=30.0)
    quat_b = Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)

    lerped_point = calc.lerp_point(point_a, point_b, 0.25)
    lerped_vector = calc.lerp_vector3(vector_a, vector_b, 0.25)
    lerped_quat = calc.lerp_quat(identity_quaternion, quat_b, 0.5)
    slerped_quat = calc.slerp_quat(identity_quaternion, quat_b, 0.5)
    lerped_pose = calc.lerp_pose(
        Pose(position=point_a, orientation=identity_quaternion),
        Pose(position=Point(x=2.0, y=4.0, z=6.0), orientation=quat_b),
        0.5,
        slerp=True,
    )
    lerped_transform = calc.lerp_transform(
        Transform(translation=vector_a, rotation=identity_quaternion),
        Transform(translation=Vector3(x=2.0, y=4.0, z=6.0), rotation=identity_quaternion),
        0.5,
    )

    assert calc.lerp_float(2.0, 10.0, 0.25) == pytest.approx(4.0)
    assert [lerped_point.x, lerped_point.y, lerped_point.z] == pytest.approx([2.5, 5.0, 7.5])
    assert [lerped_vector.x, lerped_vector.y, lerped_vector.z] == pytest.approx([2.5, 5.0, 7.5])
    assert calc.angular_distance_quat(identity_quaternion, quat_b) == pytest.approx(math.pi)
    assert [lerped_quat.x, lerped_quat.y, lerped_quat.z, lerped_quat.w] == pytest.approx(
        [
            0.0,
            0.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
        ]
    )
    assert [slerped_quat.x, slerped_quat.y, slerped_quat.z, slerped_quat.w] == pytest.approx(
        [
            0.0,
            0.0,
            math.sqrt(0.5),
            math.sqrt(0.5),
        ]
    )
    assert [lerped_pose.position.x, lerped_pose.position.y, lerped_pose.position.z] == pytest.approx([1.0, 2.0, 3.0])
    assert [
        lerped_transform.translation.x,
        lerped_transform.translation.y,
        lerped_transform.translation.z,
    ] == pytest.approx([1.0, 2.0, 3.0])


def test_normalisation_helpers():
    pose = Pose(position=Point(x=1.0), orientation=Quaternion(w=2.0))
    transform = Transform(translation=Vector3(x=1.0), rotation=Quaternion(w=2.0))
    pose_stamped = PoseStamped(header=Header(frame_id="map"), pose=pose)
    transform_stamped = TransformStamped(header=Header(frame_id="map"), transform=transform)

    assert calc.normalise_quat(Quaternion(w=2.0)).w == pytest.approx(1.0)
    assert calc.normalise_pose(pose).orientation.w == pytest.approx(1.0)
    assert calc.normalise_transform(transform).rotation.w == pytest.approx(1.0)
    assert calc.normalise_pose_stamped(pose_stamped).header.frame_id == "map"
    assert calc.normalise_transform_stamped(transform_stamped).header.frame_id == "map"


def test_apply_transform_to_points_and_vectors(transform_a):
    point = Point(x=1.0, y=0.0, z=0.0)
    vector = Vector3(x=1.0, y=0.0, z=0.0)
    matrix = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed_point = calc.transform_point(point, transform_a)
    transformed_vector = calc.transform_vector3(vector, transform_a)
    transformed_point_from_pose = calc.transform_point(
        point,
        Pose(position=Point(x=1.0, y=0.0, z=0.0), orientation=transform_a.rotation),
    )
    transformed_vector_from_matrix = calc.transform_vector3(vector, matrix)

    assert [transformed_point.x, transformed_point.y, transformed_point.z] == pytest.approx([1.0, 1.0, 0.0])
    assert [transformed_vector.x, transformed_vector.y, transformed_vector.z] == pytest.approx([0.0, 1.0, 0.0])
    assert [
        transformed_point_from_pose.x,
        transformed_point_from_pose.y,
        transformed_point_from_pose.z,
    ] == pytest.approx([1.0, 1.0, 0.0])
    assert [
        transformed_vector_from_matrix.x,
        transformed_vector_from_matrix.y,
        transformed_vector_from_matrix.z,
    ] == pytest.approx([0.0, 1.0, 0.0])

    with pytest.raises(ValueError):
        calc.transform_point(point, np.eye(3))
    with pytest.raises(ValueError):
        calc.transform_vector3(vector, np.eye(3))
