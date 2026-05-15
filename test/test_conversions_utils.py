import math

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
)
from rclpy.time import Time as RclpyTime
from scipy.spatial.transform import Rotation
from std_msgs.msg import Header

from tf2_utils import conversions as conv


TESTED_METHODS = {
    'euler_to_quat',
    'get_timestamp',
    'list_to_point',
    'list_to_pose',
    'list_to_quat',
    'list_to_transform',
    'list_to_vector3',
    'np_to_point',
    'np_to_pose',
    'np_to_quat',
    'np_to_transform',
    'np_to_vector3',
    'point_to_list',
    'point_to_np',
    'point_to_str',
    'point_to_vector3',
    'pose_stamped_to_np',
    'pose_stamped_to_transform_stamped',
    'pose_to_list',
    'pose_to_np',
    'pose_to_str',
    'pose_to_transform',
    'posestamped_to_str',
    'quat_to_euler',
    'quat_to_list',
    'quat_to_matrix',
    'quat_to_np',
    'quat_to_rot_matrix',
    'quat_to_str',
    'rot_matrix_to_quat',
    'stamp_pose',
    'stamp_transform',
    'transform_stamped_to_np',
    'transform_stamped_to_pose_stamped',
    'transform_to_list',
    'transform_to_np',
    'transform_to_pose',
    'transform_to_str',
    'transformstamped_to_str',
    'vector3_to_list',
    'vector3_to_np',
    'vector3_to_point',
    'vector3_to_str',
}


@pytest.fixture
def point_model():
    return Point(x=1.0, y=2.0, z=3.0)


@pytest.fixture
def vector_model():
    return Vector3(x=4.0, y=5.0, z=6.0)


@pytest.fixture
def identity_quaternion_model():
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


@pytest.fixture
def yaw_90_quaternion_model():
    return Quaternion(x=0.0, y=0.0, z=math.sqrt(0.5), w=math.sqrt(0.5))


@pytest.fixture
def pose_model(identity_quaternion_model):
    return Pose(position=Point(x=1.0, y=2.0, z=3.0), orientation=identity_quaternion_model)


@pytest.fixture
def transform_model(identity_quaternion_model):
    return Transform(
        translation=Vector3(x=1.0, y=2.0, z=3.0),
        rotation=identity_quaternion_model,
    )


@pytest.fixture
def stamped_pose_model(pose_model):
    return PoseStamped(
        header=Header(frame_id='map', stamp=Time(sec=12, nanosec=34)),
        pose=pose_model,
    )


@pytest.fixture
def stamped_transform_model(transform_model):
    return TransformStamped(
        header=Header(frame_id='map', stamp=Time(sec=12, nanosec=34)),
        child_frame_id='tool',
        transform=transform_model,
    )


@pytest.fixture
def known_identity_matrix():
    return np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 2.0],
        [0.0, 0.0, 1.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


@pytest.fixture
def known_yaw_90_matrix():
    return np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])


def test_all_public_conversion_methods_are_covered():
    assert set(conv.__all__) == TESTED_METHODS


def test_timestamp_conversion_accepts_builtin_and_rclpy_time():
    builtin = Time(sec=3, nanosec=4)
    assert conv.get_timestamp(builtin, None) is builtin

    stamp = conv.get_timestamp(RclpyTime(seconds=5, nanoseconds=6), None)
    assert stamp.sec == 5
    assert stamp.nanosec == 6

    with pytest.raises(TypeError):
        conv.get_timestamp('now', None)


def test_point_vector_and_list_conversions_round_trip(point_model, vector_model):
    point_from_np = conv.np_to_point(np.array([1.0, 2.0, 3.0]))
    vector_from_np = conv.np_to_vector3([4.0, 5.0, 6.0])
    point_from_list = conv.list_to_point([7.0, 8.0, 9.0])
    vector_from_list = conv.list_to_vector3([10.0, 11.0, 12.0])
    vector_from_point = conv.point_to_vector3(point_model)
    point_from_vector = conv.vector3_to_point(vector_model)

    assert [point_from_np.x, point_from_np.y, point_from_np.z] == pytest.approx([1.0, 2.0, 3.0])
    assert [vector_from_np.x, vector_from_np.y, vector_from_np.z] == pytest.approx([4.0, 5.0, 6.0])
    assert conv.point_to_np(point_model).tolist() == pytest.approx([1.0, 2.0, 3.0])
    assert conv.vector3_to_np(vector_model).tolist() == pytest.approx([4.0, 5.0, 6.0])
    assert conv.point_to_list(point_model) == pytest.approx([1.0, 2.0, 3.0])
    assert conv.vector3_to_list(vector_model) == pytest.approx([4.0, 5.0, 6.0])
    assert [point_from_list.x, point_from_list.y, point_from_list.z] == pytest.approx([7.0, 8.0, 9.0])
    assert [vector_from_list.x, vector_from_list.y, vector_from_list.z] == pytest.approx([10.0, 11.0, 12.0])
    assert [vector_from_point.x, vector_from_point.y, vector_from_point.z] == pytest.approx([1.0, 2.0, 3.0])
    assert [point_from_vector.x, point_from_vector.y, point_from_vector.z] == pytest.approx([4.0, 5.0, 6.0])

    with pytest.raises(ValueError):
        conv.list_to_point([1.0, 2.0])


def test_quaternion_euler_matrix_and_rotation_conversions(
    yaw_90_quaternion_model,
    known_yaw_90_matrix,
):
    quaternion = conv.euler_to_quat(0.0, 0.0, math.pi / 2.0)
    euler = conv.quat_to_euler(yaw_90_quaternion_model)
    euler_from_scalars = conv.quat_to_euler(
        0.0,
        0.0,
        yaw_90_quaternion_model.z,
        yaw_90_quaternion_model.w,
    )
    matrix = conv.quat_to_matrix(yaw_90_quaternion_model)
    quat_from_matrix = conv.rot_matrix_to_quat(known_yaw_90_matrix)
    quat_from_list = conv.list_to_quat([0.0, 0.0, 1.0, 1.0])
    quat_from_rotation = conv.np_to_quat(Rotation.from_euler('x', 0.25))

    assert [quaternion.x, quaternion.y, quaternion.z, quaternion.w] == pytest.approx([
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    ])
    assert euler == pytest.approx([0.0, 0.0, math.pi / 2.0])
    assert euler_from_scalars[2] == pytest.approx(math.pi / 2.0)
    assert conv.quat_to_list(yaw_90_quaternion_model) == pytest.approx([
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    ])
    np.testing.assert_allclose(matrix, known_yaw_90_matrix, atol=1e-8)
    np.testing.assert_allclose(conv.quat_to_rot_matrix(yaw_90_quaternion_model), matrix)
    assert [quat_from_matrix.x, quat_from_matrix.y, quat_from_matrix.z, quat_from_matrix.w] == pytest.approx([
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    ])
    assert [quat_from_list.x, quat_from_list.y, quat_from_list.z, quat_from_list.w] == pytest.approx([
        0.0,
        0.0,
        math.sqrt(0.5),
        math.sqrt(0.5),
    ])
    assert quat_from_rotation.x == pytest.approx(math.sin(0.125))


def test_pose_transform_matrix_and_list_conversions_round_trip(
    pose_model,
    transform_model,
    known_identity_matrix,
):
    pose_from_list = conv.list_to_pose([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0])
    transform_from_pose = conv.pose_to_transform(pose_model)
    pose_from_transform = conv.transform_to_pose(transform_model)
    pose_from_np = conv.np_to_pose(known_identity_matrix)
    transform_from_np = conv.np_to_transform(known_identity_matrix)
    transform_from_list = conv.list_to_transform(conv.transform_to_list(transform_model))

    assert conv.pose_to_list(pose_model) == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert conv.transform_to_list(transform_model) == pytest.approx(
        [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0],
    )
    assert [
        pose_from_list.position.x,
        pose_from_list.position.y,
        pose_from_list.position.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    assert [
        transform_from_pose.translation.x,
        transform_from_pose.translation.y,
        transform_from_pose.translation.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    assert [
        pose_from_transform.position.x,
        pose_from_transform.position.y,
        pose_from_transform.position.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    np.testing.assert_allclose(conv.pose_to_np(pose_model), known_identity_matrix)
    np.testing.assert_allclose(conv.transform_to_np(transform_model), known_identity_matrix)
    assert [
        pose_from_np.position.x,
        pose_from_np.position.y,
        pose_from_np.position.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    assert [
        transform_from_np.translation.x,
        transform_from_np.translation.y,
        transform_from_np.translation.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    assert [
        transform_from_list.translation.x,
        transform_from_list.translation.y,
        transform_from_list.translation.z,
    ] == pytest.approx([1.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        conv.np_to_pose(np.eye(3))


def test_stamped_pose_and_transform_conversions(
    pose_model,
    transform_model,
    stamped_pose_model,
    stamped_transform_model,
    known_identity_matrix,
):
    pose_stamped = conv.stamp_pose(pose_model, frame='map', stamp=Time(sec=12, nanosec=34))
    transform_stamped = conv.stamp_transform(
        transform_model,
        child_frame='tool',
        frame='map',
        stamp=Time(sec=12, nanosec=34),
    )
    pose_from_transform = conv.transform_stamped_to_pose_stamped(stamped_transform_model)
    transform_from_pose = conv.pose_stamped_to_transform_stamped(stamped_pose_model, 'camera')

    assert pose_stamped.header.frame_id == 'map'
    assert pose_stamped.header.stamp.sec == 12
    assert transform_stamped.header.frame_id == 'map'
    assert transform_stamped.child_frame_id == 'tool'
    np.testing.assert_allclose(conv.pose_stamped_to_np(stamped_pose_model), known_identity_matrix)
    np.testing.assert_allclose(conv.transform_stamped_to_np(stamped_transform_model), known_identity_matrix)
    assert pose_from_transform.header.frame_id == 'map'
    assert [pose_from_transform.pose.position.x, pose_from_transform.pose.position.y] == pytest.approx([1.0, 2.0])
    assert transform_from_pose.child_frame_id == 'camera'


def test_string_conversions_include_type_and_frame_context(
    pose_model,
    transform_model,
    stamped_pose_model,
    stamped_transform_model,
):
    assert 'Point' in conv.point_to_str(pose_model.position)
    assert 'Vector3' in conv.vector3_to_str(transform_model.translation)
    assert 'Quaternion' in conv.quat_to_str(pose_model.orientation)
    assert 'euler' in conv.quat_to_str(pose_model.orientation, euler=True)
    assert 'Pose ' in conv.pose_to_str(pose_model)
    assert 'PoseStamped' in conv.posestamped_to_str(stamped_pose_model)
    assert 'Transform ' in conv.transform_to_str(transform_model)
    assert 'child_frame: tool' in conv.transformstamped_to_str(stamped_transform_model)
