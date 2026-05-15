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
from std_msgs.msg import Header

from tf2_utils import comparisons as comp


TESTED_METHODS = {
    'is_equal_header',
    'is_equal_point',
    'is_equal_pose',
    'is_equal_posestamped',
    'is_equal_quat',
    'is_equal_transform',
    'is_equal_transformstamped',
    'is_equal_vector3',
}


@pytest.fixture
def header_model():
    return Header(frame_id='world', stamp=Time(sec=1, nanosec=2))


@pytest.fixture
def header_same_frame_other_time():
    return Header(frame_id='world', stamp=Time(sec=3, nanosec=4))


@pytest.fixture
def header_other_frame():
    return Header(frame_id='map', stamp=Time(sec=1, nanosec=2))


@pytest.fixture
def point_model():
    return Point(x=1.0, y=0.0, z=0.0)


@pytest.fixture
def point_close_model():
    return Point(x=1.0 + 1e-5, y=0.0, z=0.0)


@pytest.fixture
def point_far_model():
    return Point(x=1.1, y=0.0, z=0.0)


@pytest.fixture
def vector_model():
    return Vector3(x=0.0, y=2.0, z=0.0)


@pytest.fixture
def vector_close_model():
    return Vector3(x=0.0, y=2.0 + 1e-5, z=0.0)


@pytest.fixture
def vector_far_model():
    return Vector3(x=0.0, y=2.1, z=0.0)


@pytest.fixture
def quaternion_model():
    return Quaternion(x=0.0, y=0.0, z=1.0, w=0.0)


@pytest.fixture
def quaternion_same_rotation_model():
    return Quaternion(x=0.0, y=0.0, z=-1.0, w=0.0)


@pytest.fixture
def quaternion_other_rotation_model():
    return Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)


@pytest.fixture
def pose_model():
    return Pose(position=Point(x=1.0), orientation=Quaternion(w=1.0))


@pytest.fixture
def pose_close_model():
    return Pose(position=Point(x=1.0 + 1e-5), orientation=Quaternion(w=1.0))


@pytest.fixture
def pose_far_model():
    return Pose(position=Point(x=2.0), orientation=Quaternion(w=1.0))


@pytest.fixture
def transform_model():
    return Transform(translation=Vector3(x=1.0), rotation=Quaternion(w=1.0))


@pytest.fixture
def transform_close_model():
    return Transform(
        translation=Vector3(x=1.0 + 1e-5),
        rotation=Quaternion(w=1.0),
    )


@pytest.fixture
def transform_far_model():
    return Transform(translation=Vector3(x=2.0), rotation=Quaternion(w=1.0))


def test_all_public_comparison_methods_are_covered():
    assert set(comp.__all__) == TESTED_METHODS


def test_header_comparison_can_ignore_or_check_timestamps(
    header_model,
    header_same_frame_other_time,
    header_other_frame,
):
    assert comp.is_equal_header(header_model, header_same_frame_other_time)
    assert not comp.is_equal_header(
        header_model,
        header_same_frame_other_time,
        check_timestamp=True,
    )
    assert comp.is_equal_header(header_model, header_model, check_timestamp=True)
    assert not comp.is_equal_header(header_model, header_other_frame)


def test_point_vector_and_quaternion_comparisons_use_tolerance(
    point_model,
    point_close_model,
    point_far_model,
    vector_model,
    vector_close_model,
    vector_far_model,
    quaternion_model,
    quaternion_same_rotation_model,
    quaternion_other_rotation_model,
):
    assert comp.is_equal_point(point_model, point_close_model)
    assert not comp.is_equal_point(point_model, point_far_model)
    assert comp.is_equal_vector3(vector_model, vector_close_model)
    assert not comp.is_equal_vector3(vector_model, vector_far_model)
    assert comp.is_equal_quat(quaternion_model, quaternion_same_rotation_model)
    assert not comp.is_equal_quat(quaternion_model, quaternion_other_rotation_model)


def test_pose_transform_and_stamped_comparisons_check_nested_fields(
    header_model,
    header_other_frame,
    pose_model,
    pose_close_model,
    pose_far_model,
    transform_model,
    transform_close_model,
    transform_far_model,
):
    assert comp.is_equal_pose(pose_model, pose_close_model)
    assert not comp.is_equal_pose(pose_model, pose_far_model)
    assert comp.is_equal_posestamped(
        PoseStamped(header=header_model, pose=pose_model),
        PoseStamped(header=header_model, pose=pose_close_model),
    )
    assert not comp.is_equal_posestamped(
        PoseStamped(header=header_model, pose=pose_model),
        PoseStamped(header=header_other_frame, pose=pose_model),
    )
    assert comp.is_equal_transform(transform_model, transform_close_model)
    assert not comp.is_equal_transform(transform_model, transform_far_model)
    assert comp.is_equal_transformstamped(
        TransformStamped(header=header_model, child_frame_id='tool', transform=transform_model),
        TransformStamped(
            header=header_model,
            child_frame_id='tool',
            transform=transform_close_model,
        ),
    )
    assert not comp.is_equal_transformstamped(
        TransformStamped(header=header_model, child_frame_id='tool', transform=transform_model),
        TransformStamped(header=header_model, child_frame_id='camera', transform=transform_model),
    )
