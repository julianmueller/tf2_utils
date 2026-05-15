from types import MethodType, SimpleNamespace

import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    Quaternion,
    Transform,
    TransformStamped,
    Vector3,
    Vector3Stamped,
)
from rclpy.duration import Duration
from rclpy.time import Time as RclpyTime
from std_msgs.msg import Header

from std_srvs.srv import Trigger

from tf2_utils.node import TF2UtilsNode


class _Clock:

    def now(self):
        return SimpleNamespace(to_msg=lambda: Time(sec=9, nanosec=10))


class _Buffer:

    def __init__(self, transform):
        self.transform = transform
        self.lookup_calls = []
        self.can_calls = []
        self.clear_calls = 0

    def lookup_transform(self, target_frame, source_frame, stamp, timeout=None):
        self.lookup_calls.append((target_frame, source_frame, stamp, timeout))
        return self.transform

    def can_transform(self, target_frame, source_frame, stamp, timeout=None):
        self.can_calls.append((target_frame, source_frame, stamp, timeout))
        return True

    def clear(self):
        self.clear_calls += 1


class _Broadcaster:

    def __init__(self):
        self.sent = []

    def sendTransform(self, transform):
        self.sent.append(transform)


@pytest.fixture
def target_transform_model():
    return TransformStamped(
        header=Header(frame_id='world', stamp=Time(sec=1, nanosec=2)),
        child_frame_id='tool',
        transform=Transform(
            translation=Vector3(x=1.0, y=2.0, z=3.0),
            rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


@pytest.fixture
def node_model(monkeypatch, target_transform_model):
    node = SimpleNamespace()
    node.tf_buffer = _Buffer(target_transform_model)
    node.buffer = node.tf_buffer
    node.tf_broadcaster = _Broadcaster()
    node.broadcaster = node.tf_broadcaster
    node.static_tf_broadcaster = _Broadcaster()
    node.static_broadcaster = node.static_tf_broadcaster
    node.destroyed_publishers = []
    node.lookup_transform = MethodType(TF2UtilsNode.lookup_transform, node)
    node.can_transform = MethodType(TF2UtilsNode.can_transform, node)
    node.lookup_pose = MethodType(TF2UtilsNode.lookup_pose, node)
    node.transform_pose = MethodType(TF2UtilsNode.transform_pose, node)
    node.transform_point = MethodType(TF2UtilsNode.transform_point, node)
    node.transform_vector3 = MethodType(TF2UtilsNode.transform_vector3, node)
    node.broadcast_transform = MethodType(TF2UtilsNode.broadcast_transform, node)
    node.broadcast_static_transform = MethodType(TF2UtilsNode.broadcast_static_transform, node)
    node.clear_static_transforms = MethodType(TF2UtilsNode.clear_static_transforms, node)
    node._handle_clear_static = MethodType(TF2UtilsNode._handle_clear_static, node)
    node._make_transform_stamped = MethodType(TF2UtilsNode._make_transform_stamped, node)
    monkeypatch.setattr(node, 'destroy_publisher', node.destroyed_publishers.append, raising=False)
    monkeypatch.setattr(node, 'get_clock', lambda: _Clock(), raising=False)
    return node


@pytest.fixture
def pose_model():
    return Pose(
        position=Point(x=1.0, y=0.0, z=0.0),
        orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


@pytest.fixture
def stamped_pose_model(pose_model):
    return PoseStamped(
        header=Header(frame_id='tool', stamp=Time(sec=3, nanosec=0)),
        pose=pose_model,
    )


@pytest.fixture
def point_model():
    return Point(x=1.0, y=0.0, z=0.0)


@pytest.fixture
def stamped_point_model(point_model):
    return PointStamped(header=Header(frame_id='tool'), point=point_model)


@pytest.fixture
def vector_model():
    return Vector3(x=1.0, y=0.0, z=0.0)


@pytest.fixture
def stamped_vector_model(vector_model):
    return Vector3Stamped(header=Header(frame_id='tool'), vector=vector_model)


@pytest.fixture
def broadcast_transform_model():
    return Transform(
        translation=Vector3(x=1.0, y=0.0, z=0.0),
        rotation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
    )


def test_node_module_only_exports_the_node_class():
    assert set(__import__('tf2_utils.node', fromlist=['__all__']).__all__) == {'TF2UtilsNode'}


def test_lookup_transform_and_can_transform_delegate_to_buffer(node_model):
    stamp = RclpyTime(seconds=4)

    result = TF2UtilsNode.lookup_transform(
        node_model,
        'world',
        'tool',
        stamp=stamp,
        timeout=1.5,
    )
    assert result.child_frame_id == 'tool'
    target_frame, source_frame, sent_stamp, timeout = node_model.tf_buffer.lookup_calls[-1]
    assert (target_frame, source_frame) == ('world', 'tool')
    assert sent_stamp is stamp
    assert isinstance(timeout, Duration)
    assert timeout.nanoseconds == 1_500_000_000

    assert TF2UtilsNode.can_transform(node_model, 'world', 'tool', timeout=0.25)
    assert node_model.tf_buffer.can_calls[-1][3].nanoseconds == 250_000_000


def test_lookup_pose_converts_transform_to_pose(node_model):
    pose_stamped = TF2UtilsNode.lookup_pose(node_model, 'world', 'tool')

    assert pose_stamped.header.frame_id == 'world'
    assert [
        pose_stamped.pose.position.x,
        pose_stamped.pose.position.y,
        pose_stamped.pose.position.z,
    ] == pytest.approx([1.0, 2.0, 3.0])


def test_transform_pose_accepts_stamped_and_unstamped_inputs(
    node_model,
    pose_model,
    stamped_pose_model,
):
    transformed_stamped = TF2UtilsNode.transform_pose(node_model, stamped_pose_model, 'world')
    transformed_unstamped = TF2UtilsNode.transform_pose(
        node_model,
        pose_model,
        'world',
        source_frame='tool',
    )

    assert transformed_stamped.header.frame_id == 'world'
    assert [
        transformed_stamped.pose.position.x,
        transformed_stamped.pose.position.y,
        transformed_stamped.pose.position.z,
    ] == pytest.approx([2.0, 2.0, 3.0])
    assert node_model.tf_buffer.lookup_calls[-1][1] == 'tool'
    assert [
        transformed_unstamped.pose.position.x,
        transformed_unstamped.pose.position.y,
        transformed_unstamped.pose.position.z,
    ] == pytest.approx([2.0, 2.0, 3.0])

    with pytest.raises(ValueError):
        TF2UtilsNode.transform_pose(node_model, pose_model, 'world')


def test_transform_point_and_vector3(
    node_model,
    point_model,
    stamped_point_model,
    vector_model,
    stamped_vector_model,
):
    transformed_point = TF2UtilsNode.transform_point(node_model, stamped_point_model, 'world')
    transformed_vector = TF2UtilsNode.transform_vector3(node_model, stamped_vector_model, 'world')
    transformed_unstamped_point = TF2UtilsNode.transform_point(
        node_model,
        point_model,
        'world',
        source_frame='tool',
    )
    transformed_unstamped_vector = TF2UtilsNode.transform_vector3(
        node_model,
        vector_model,
        'world',
        source_frame='tool',
    )

    assert [
        transformed_point.point.x,
        transformed_point.point.y,
        transformed_point.point.z,
    ] == pytest.approx([2.0, 2.0, 3.0])
    assert [
        transformed_vector.vector.x,
        transformed_vector.vector.y,
        transformed_vector.vector.z,
    ] == pytest.approx([1.0, 0.0, 0.0])
    assert [
        transformed_unstamped_point.point.x,
        transformed_unstamped_point.point.y,
        transformed_unstamped_point.point.z,
    ] == pytest.approx([2.0, 2.0, 3.0])
    assert [
        transformed_unstamped_vector.vector.x,
        transformed_unstamped_vector.vector.y,
        transformed_unstamped_vector.vector.z,
    ] == pytest.approx([1.0, 0.0, 0.0])

    with pytest.raises(ValueError):
        TF2UtilsNode.transform_point(node_model, Point(), 'world')
    with pytest.raises(ValueError):
        TF2UtilsNode.transform_vector3(node_model, Vector3(), 'world')


def test_broadcast_methods_stamp_and_send_transforms(
    node_model,
    broadcast_transform_model,
):
    stamp = Time(sec=5, nanosec=6)
    sent = TF2UtilsNode.broadcast_transform(
        node_model,
        broadcast_transform_model,
        parent_frame='world',
        child_frame='tool',
        stamp=stamp,
    )
    pose = PoseStamped(
        header=Header(frame_id='map', stamp=Time(sec=7, nanosec=0)),
        pose=Pose(orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
    )
    static_sent = TF2UtilsNode.broadcast_static_transform(
        node_model,
        pose,
        child_frame='camera',
    )

    assert sent.header.frame_id == 'world'
    assert sent.header.stamp is stamp
    assert sent.child_frame_id == 'tool'
    assert node_model.tf_broadcaster.sent == [sent]
    assert static_sent.header.frame_id == 'map'
    assert static_sent.child_frame_id == 'camera'
    assert node_model.static_tf_broadcaster.sent == [static_sent]

    with pytest.raises(ValueError):
        TF2UtilsNode.broadcast_transform(
            node_model,
            broadcast_transform_model,
            parent_frame='world',
        )


def test_clear_static_transforms_resets_static_broadcaster_and_buffer(node_model, monkeypatch):
    old_publisher = object()
    node_model.static_tf_broadcaster.pub_tf = old_publisher

    class ReplacementBroadcaster(_Broadcaster):

        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr('tf2_utils.node.StaticTransformBroadcaster', ReplacementBroadcaster)

    TF2UtilsNode.clear_static_transforms(node_model)

    assert node_model.destroyed_publishers == [old_publisher]
    assert isinstance(node_model.static_tf_broadcaster, ReplacementBroadcaster)
    assert node_model.static_broadcaster is node_model.static_tf_broadcaster
    assert node_model.tf_buffer.clear_calls == 1


def test_clear_static_service_callback_reports_limitations(node_model, monkeypatch):
    class ReplacementBroadcaster(_Broadcaster):

        def __init__(self, *_args, **_kwargs):
            super().__init__()

    monkeypatch.setattr('tf2_utils.node.StaticTransformBroadcaster', ReplacementBroadcaster)

    response = TF2UtilsNode._handle_clear_static(
        node_model,
        Trigger.Request(),
        Trigger.Response(),
    )

    assert response.success
    assert "Other nodes may keep static transforms" in response.message
    assert isinstance(node_model.static_tf_broadcaster, ReplacementBroadcaster)
    assert node_model.tf_buffer.clear_calls == 1
