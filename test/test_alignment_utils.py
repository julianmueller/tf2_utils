import numpy as np
import pytest
from geometry_msgs.msg import Point, Pose, Quaternion, Transform, Vector3
from scipy.spatial.transform import Rotation

from tf2_utils import alignment as align


TESTED_METHODS = {
    "X_AXIS",
    "XY_PLANE",
    "XZ_PLANE",
    "Y_AXIS",
    "YZ_PLANE",
    "Z_AXIS",
    "align_pose_axis_parallel_to_pose_axis",
    "align_pose_axis_to_pose_offset",
    "align_pose_axis_to_point",
    "align_pose_axis_to_vector",
    "align_pose_plane_to_normal",
    "align_transform_axis_parallel_to_transform_axis",
    "align_transform_axis_to_transform_offset",
    "align_transform_axis_to_point",
    "align_transform_axis_to_vector",
    "align_transform_plane_to_normal",
    "quat_from_axis_alignment",
    "quat_from_plane_alignment",
}


@pytest.fixture
def pose_model():
    return Pose(
        position=Point(x=1.0, y=2.0, z=3.0),
        orientation=Quaternion(),
    )


@pytest.fixture
def transform_model():
    return Transform(
        translation=Vector3(x=2.0, y=0.0, z=0.0),
        rotation=Quaternion(),
    )


@pytest.fixture
def yaw_quarter_turn_quaternion():
    value = np.sqrt(0.5)
    return Quaternion(x=0.0, y=0.0, z=value, w=value)


@pytest.fixture
def yaw_quarter_turn_pose_model(yaw_quarter_turn_quaternion):
    return Pose(
        position=Point(x=1.0, y=2.0, z=3.0),
        orientation=yaw_quarter_turn_quaternion,
    )


@pytest.fixture
def yaw_quarter_turn_transform_model(yaw_quarter_turn_quaternion):
    return Transform(
        translation=Vector3(x=2.0, y=0.0, z=0.0),
        rotation=yaw_quarter_turn_quaternion,
    )


@pytest.fixture
def known_world_x():
    return np.array([1.0, 0.0, 0.0], dtype=float)


@pytest.fixture
def known_world_y():
    return np.array([0.0, 1.0, 0.0], dtype=float)


@pytest.fixture
def known_world_z():
    return np.array([0.0, 0.0, 1.0], dtype=float)


def test_all_public_alignment_methods_are_covered():
    assert set(align.__all__) == TESTED_METHODS


def test_quat_from_axis_alignment_points_axis_and_respects_normal_hint(
    known_world_x,
    known_world_y,
    known_world_z,
):
    quaternion = align.quat_from_axis_alignment(
        align.X_AXIS,
        Vector3(x=0.0, y=1.0, z=0.0),
        normal_axis=align.Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    matrix = Rotation.from_quat(
        [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(matrix @ known_world_x, known_world_y, atol=1e-8)
    np.testing.assert_allclose(matrix @ known_world_z, known_world_z, atol=1e-8)


def test_axis_alignment_accepts_flip_axis_and_parallel_normal_hint(
    known_world_y,
    known_world_z,
):
    quaternion = align.quat_from_axis_alignment(
        align.Z_AXIS,
        Vector3(x=0.0, y=0.0, z=-1.0),
        normal_axis=align.Y_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=-1.0),
        flip_axis=True,
    )
    matrix = Rotation.from_quat(
        [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(matrix @ -known_world_z, -known_world_z, atol=1e-8)
    np.testing.assert_allclose(np.linalg.norm(matrix @ known_world_y), 1.0, atol=1e-8)
    assert abs(float(np.dot(matrix @ known_world_y, -known_world_z))) == pytest.approx(0.0)


def test_pose_and_transform_axis_to_point_alignment_preserves_translation(
    pose_model,
    transform_model,
    known_world_x,
    known_world_y,
    known_world_z,
):
    aligned_pose = align.align_pose_axis_to_point(
        pose_model,
        Point(x=1.0, y=5.0, z=3.0),
        axis=align.X_AXIS,
        normal_axis=align.Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    aligned_transform = align.align_transform_axis_to_point(
        transform_model,
        Point(x=2.0, y=0.0, z=-4.0),
        axis=align.Z_AXIS,
        normal_axis=align.Y_AXIS,
        flip_axis=True,
        normal_direction=Vector3(x=0.0, y=1.0, z=0.0),
    )
    pose_matrix = Rotation.from_quat(
        [
            aligned_pose.orientation.x,
            aligned_pose.orientation.y,
            aligned_pose.orientation.z,
            aligned_pose.orientation.w,
        ]
    ).as_matrix()
    transform_matrix = Rotation.from_quat(
        [
            aligned_transform.rotation.x,
            aligned_transform.rotation.y,
            aligned_transform.rotation.z,
            aligned_transform.rotation.w,
        ]
    ).as_matrix()

    assert [
        aligned_pose.position.x,
        aligned_pose.position.y,
        aligned_pose.position.z,
    ] == pytest.approx([1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose_matrix @ known_world_x, known_world_y, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_z, known_world_z, atol=1e-8)
    assert [
        aligned_transform.translation.x,
        aligned_transform.translation.y,
        aligned_transform.translation.z,
    ] == pytest.approx([2.0, 0.0, 0.0])
    np.testing.assert_allclose(transform_matrix @ -known_world_z, -known_world_z, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_y, known_world_y, atol=1e-8)


def test_axis_to_vector_alignment_wrappers(
    pose_model,
    transform_model,
    known_world_x,
    known_world_y,
    known_world_z,
):
    aligned_pose = align.align_pose_axis_to_vector(
        pose_model,
        Vector3(x=1.0, y=0.0, z=0.0),
        axis=align.Z_AXIS,
        normal_axis=align.Y_AXIS,
        normal_direction=Vector3(x=0.0, y=1.0, z=0.0),
    )
    aligned_transform = align.align_transform_axis_to_vector(
        transform_model,
        Vector3(x=0.0, y=1.0, z=0.0),
        axis=align.Y_AXIS,
        normal_axis=align.Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    pose_matrix = Rotation.from_quat(
        [
            aligned_pose.orientation.x,
            aligned_pose.orientation.y,
            aligned_pose.orientation.z,
            aligned_pose.orientation.w,
        ]
    ).as_matrix()
    transform_matrix = Rotation.from_quat(
        [
            aligned_transform.rotation.x,
            aligned_transform.rotation.y,
            aligned_transform.rotation.z,
            aligned_transform.rotation.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(pose_matrix @ known_world_z, known_world_x, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_y, known_world_y, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_y, known_world_y, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_z, known_world_z, atol=1e-8)


def test_axis_parallel_alignment_uses_reference_axis_and_reference_normal(
    pose_model,
    transform_model,
    yaw_quarter_turn_pose_model,
    yaw_quarter_turn_transform_model,
    known_world_x,
    known_world_y,
    known_world_z,
):
    aligned_pose = align.align_pose_axis_parallel_to_pose_axis(
        pose_model,
        yaw_quarter_turn_pose_model,
        axis=align.Y_AXIS,
        reference_axis=align.X_AXIS,
        normal_axis=align.Z_AXIS,
        reference_normal_axis=align.Z_AXIS,
    )
    aligned_transform = align.align_transform_axis_parallel_to_transform_axis(
        transform_model,
        yaw_quarter_turn_transform_model,
        axis=align.X_AXIS,
        reference_axis=align.Y_AXIS,
        normal_axis=align.Z_AXIS,
        reference_normal_axis=align.Z_AXIS,
    )
    pose_matrix = Rotation.from_quat(
        [
            aligned_pose.orientation.x,
            aligned_pose.orientation.y,
            aligned_pose.orientation.z,
            aligned_pose.orientation.w,
        ]
    ).as_matrix()
    transform_matrix = Rotation.from_quat(
        [
            aligned_transform.rotation.x,
            aligned_transform.rotation.y,
            aligned_transform.rotation.z,
            aligned_transform.rotation.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(pose_matrix @ known_world_y, known_world_y, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_z, known_world_z, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_x, -known_world_x, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_z, known_world_z, atol=1e-8)


def test_axis_to_target_offset_points_at_target_local_point(
    pose_model,
    transform_model,
    yaw_quarter_turn_pose_model,
    yaw_quarter_turn_transform_model,
    known_world_x,
    known_world_y,
    known_world_z,
):
    aligned_pose = align.align_pose_axis_to_pose_offset(
        pose_model,
        yaw_quarter_turn_pose_model,
        target_offset=Point(x=2.0, y=0.0, z=0.0),
        axis=align.X_AXIS,
        normal_axis=align.Z_AXIS,
        normal_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    aligned_transform = align.align_transform_axis_to_transform_offset(
        transform_model,
        yaw_quarter_turn_transform_model,
        target_offset=Point(x=1.0, y=0.0, z=0.0),
        axis=align.Z_AXIS,
        normal_axis=align.X_AXIS,
        normal_direction=Vector3(x=1.0, y=0.0, z=0.0),
    )
    pose_matrix = Rotation.from_quat(
        [
            aligned_pose.orientation.x,
            aligned_pose.orientation.y,
            aligned_pose.orientation.z,
            aligned_pose.orientation.w,
        ]
    ).as_matrix()
    transform_matrix = Rotation.from_quat(
        [
            aligned_transform.rotation.x,
            aligned_transform.rotation.y,
            aligned_transform.rotation.z,
            aligned_transform.rotation.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(pose_matrix @ known_world_x, known_world_y, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_z, known_world_z, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_z, known_world_y, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_x, known_world_x, atol=1e-8)


def test_plane_alignment_points_plane_normal_and_in_plane_axis(
    pose_model,
    transform_model,
    known_world_x,
    known_world_y,
    known_world_z,
):
    quaternion = align.quat_from_plane_alignment(
        align.XY_PLANE,
        Vector3(x=0.0, y=0.0, z=1.0),
        in_plane_axis=align.X_AXIS,
        in_plane_direction=Vector3(x=0.0, y=1.0, z=0.0),
    )
    aligned_pose = align.align_pose_plane_to_normal(
        pose_model,
        Vector3(x=0.0, y=0.0, z=1.0),
        plane=align.XY_PLANE,
        in_plane_axis=align.X_AXIS,
        in_plane_direction=Vector3(x=0.0, y=1.0, z=0.0),
    )
    aligned_transform = align.align_transform_plane_to_normal(
        transform_model,
        Vector3(x=0.0, y=1.0, z=0.0),
        plane=align.XZ_PLANE,
        in_plane_axis=align.X_AXIS,
        in_plane_direction=Vector3(x=0.0, y=0.0, z=1.0),
    )
    matrix = Rotation.from_quat(
        [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]
    ).as_matrix()
    pose_matrix = Rotation.from_quat(
        [
            aligned_pose.orientation.x,
            aligned_pose.orientation.y,
            aligned_pose.orientation.z,
            aligned_pose.orientation.w,
        ]
    ).as_matrix()
    transform_matrix = Rotation.from_quat(
        [
            aligned_transform.rotation.x,
            aligned_transform.rotation.y,
            aligned_transform.rotation.z,
            aligned_transform.rotation.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(matrix @ known_world_z, known_world_z, atol=1e-8)
    np.testing.assert_allclose(matrix @ known_world_x, known_world_y, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_z, known_world_z, atol=1e-8)
    np.testing.assert_allclose(pose_matrix @ known_world_x, known_world_y, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_y, known_world_y, atol=1e-8)
    np.testing.assert_allclose(transform_matrix @ known_world_x, known_world_z, atol=1e-8)


def test_alignment_helpers_reject_invalid_constraints(pose_model):
    with pytest.raises(ValueError):
        align.quat_from_axis_alignment("forward", Vector3(x=1.0, y=0.0, z=0.0))
    with pytest.raises(ValueError):
        align.quat_from_axis_alignment(align.X_AXIS, Vector3(x=0.0, y=0.0, z=0.0))
    with pytest.raises(ValueError):
        align.quat_from_axis_alignment(
            align.X_AXIS,
            Vector3(x=1.0, y=0.0, z=0.0),
            normal_axis=align.X_AXIS,
            flip_normal_axis=True,
        )
    with pytest.raises(ValueError):
        align.quat_from_plane_alignment(9, Vector3(x=0.0, y=0.0, z=1.0))
    with pytest.raises(ValueError):
        align.align_pose_axis_to_point(pose_model, Point(x=1.0, y=2.0, z=3.0))
    with pytest.raises(TypeError):
        align.align_pose_axis_to_vector(pose_model, Point(x=1.0, y=0.0, z=0.0))
    with pytest.raises(TypeError):
        align.align_pose_axis_to_point(pose_model, Vector3(x=1.0, y=0.0, z=0.0))


def test_axis_alignment_accepts_axis_numbers(known_world_x, known_world_y):
    quaternion = align.quat_from_axis_alignment(
        0,
        Vector3(x=0.0, y=1.0, z=0.0),
        normal_axis=2,
    )
    matrix = Rotation.from_quat(
        [
            quaternion.x,
            quaternion.y,
            quaternion.z,
            quaternion.w,
        ]
    ).as_matrix()

    np.testing.assert_allclose(matrix @ known_world_x, known_world_y, atol=1e-8)
