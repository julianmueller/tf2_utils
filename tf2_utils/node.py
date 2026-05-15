from __future__ import annotations

from typing import Any

import rclpy
from rclpy.clock import Clock
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time as RclpyTime
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener

from builtin_interfaces.msg import Time
from geometry_msgs.msg import (
    Point,
    PointStamped,
    Pose,
    PoseStamped,
    Transform,
    TransformStamped,
    Vector3,
    Vector3Stamped,
)
from std_msgs.msg import Header
from std_srvs.srv import Trigger

from . import calculations as calc
from . import conversions as conv


__all__ = [
    "TF2UtilsNode",
]


def _as_rclpy_time(stamp: Time | RclpyTime | None) -> RclpyTime:
    """Convert optional ROS 2 time values into the ``rclpy`` time used by TF2."""
    if stamp is None:
        return RclpyTime()
    if isinstance(stamp, RclpyTime):
        return stamp
    if isinstance(stamp, Time):
        return RclpyTime.from_msg(stamp)
    raise TypeError(f"stamp is not a builtin_interfaces/Time or rclpy.time.Time, but a {type(stamp).__name__}")


def _as_duration(timeout: Duration | float | int | None) -> Duration:
    """Convert optional timeout seconds into an ``rclpy.duration.Duration``."""
    if timeout is None:
        return Duration()
    if isinstance(timeout, Duration):
        return timeout
    return Duration(seconds=float(timeout))


def _make_header(frame: str, stamp: Time | RclpyTime | None, clock: Clock | None) -> Header:
    """Create a stamped header for TF2 helper messages."""
    return Header(frame_id=frame, stamp=conv.get_timestamp(stamp, clock))


def _require_source_frame(source_frame: str | None) -> str:
    """Return ``source_frame`` when it is set, otherwise raise a readable error."""
    if source_frame:
        return source_frame
    raise ValueError("source_frame is required")


class TF2UtilsNode(Node):
    """Small ``rclpy`` node that owns the common TF2 buffer, listener, and broadcasters."""

    def __init__(
        self,
        node_name: str = "tf2_utils_node",
        *,
        buffer_cache_time: Duration | None = None,
        spin_thread: bool = False,
        clear_static_service_name: str | None = "/clear_static",
        **kwargs: Any,
    ) -> None:
        """Create a node with TF2 lookup, broadcast utilities, and optional services ready to use."""
        super().__init__(node_name, **kwargs)
        try:
            self.tf_buffer = Buffer(cache_time=buffer_cache_time, node=self)
        except TypeError:
            self.tf_buffer = Buffer(cache_time=buffer_cache_time)
        self.buffer = self.tf_buffer
        self.tf_listener = TransformListener(
            self.tf_buffer, self, spin_thread=spin_thread
        )  # TODO check what the spin thread does
        self.tf_broadcaster = TransformBroadcaster(self)
        self.broadcaster = self.tf_broadcaster
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.static_broadcaster = self.static_tf_broadcaster
        self.clear_static_service = (
            self.create_service(Trigger, clear_static_service_name, self._handle_clear_static)
            if clear_static_service_name
            else None
        )

    def clear_static_transforms(self) -> None:
        """Clear static transforms broadcast by this node and reset this node's TF buffer."""
        publisher = getattr(self.static_tf_broadcaster, "pub_tf", None)
        if publisher is not None:
            self.destroy_publisher(publisher)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        self.static_broadcaster = self.static_tf_broadcaster
        clear = getattr(self.tf_buffer, "clear", None)
        if clear is not None:
            clear()

    def _handle_clear_static(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """Handle the ``/clear_static`` service request."""
        del request
        self.clear_static_transforms()
        response.success = True
        response.message = (
            "Reset this node's static transform broadcaster and local TF buffer. "
            "Other nodes may keep static transforms they already received."
        )
        return response

    def lookup_transform(
        self,
        target_frame: str,
        source_frame: str,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> TransformStamped:
        """Look up the transform that converts data from ``source_frame`` into ``target_frame``."""
        return self.tf_buffer.lookup_transform(
            target_frame,
            source_frame,
            _as_rclpy_time(stamp),
            timeout=_as_duration(timeout),
        )

    def _make_transform_stamped(
        self,
        transform: Transform | TransformStamped | Pose | PoseStamped,
        parent_frame: str | None = None,
        child_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
    ) -> TransformStamped:
        """Return a ``TransformStamped`` from a transform or pose plus frame ids."""
        result = TransformStamped()

        if isinstance(transform, TransformStamped):
            result.transform = transform.transform
            result.header = Header(
                frame_id=parent_frame if parent_frame is not None else transform.header.frame_id,
                stamp=conv.get_timestamp(
                    stamp if stamp is not None else transform.header.stamp,
                    self.get_clock(),
                ),
            )
            result.child_frame_id = child_frame if child_frame is not None else transform.child_frame_id
            return result

        if isinstance(transform, PoseStamped):
            result.transform = conv.pose_to_transform(transform.pose)
            frame = parent_frame if parent_frame is not None else transform.header.frame_id
            result.header = Header(
                frame_id=frame,
                stamp=conv.get_timestamp(
                    stamp if stamp is not None else transform.header.stamp,
                    self.get_clock(),
                ),
            )
        elif isinstance(transform, Pose):
            result.transform = conv.pose_to_transform(transform)
            if parent_frame is None:
                raise ValueError("parent_frame is required when stamping a Pose")
            result.header = _make_header(parent_frame, stamp, self.get_clock())
        elif isinstance(transform, Transform):
            result.transform = transform
            if parent_frame is None:
                raise ValueError("parent_frame is required when stamping a Transform")
            result.header = _make_header(parent_frame, stamp, self.get_clock())
        else:
            raise TypeError(
                "transform must be a geometry_msgs/Transform, TransformStamped, "
                "Pose, or PoseStamped, "
                f"but is a {type(transform).__name__}"
            )

        if child_frame is None:
            raise ValueError("child_frame is required")
        result.child_frame_id = child_frame
        return result

    def can_transform(
        self,
        target_frame: str,
        source_frame: str,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> bool:
        """Return whether TF2 can transform from ``source_frame`` into ``target_frame``."""
        return bool(
            self.tf_buffer.can_transform(
                target_frame,
                source_frame,
                _as_rclpy_time(stamp),
                timeout=_as_duration(timeout),
            )
        )

    def lookup_pose(
        self,
        target_frame: str,
        source_frame: str,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> PoseStamped:
        """Look up ``source_frame`` as a pose expressed in ``target_frame``."""
        return conv.transform_stamped_to_pose_stamped(
            self.lookup_transform(
                target_frame,
                source_frame,
                stamp=stamp,
                timeout=timeout,
            )
        )

    def transform_pose(
        self,
        pose: Pose | PoseStamped,
        target_frame: str,
        source_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> PoseStamped:
        """Transform a pose into ``target_frame`` using this node's TF2 buffer."""
        if isinstance(pose, PoseStamped):
            source_frame = source_frame or pose.header.frame_id
            stamp = stamp if stamp is not None else pose.header.stamp
            pose_value = pose.pose
        elif isinstance(pose, Pose):
            if source_frame is None:
                raise ValueError("source_frame is required when transforming an unstamped Pose")
            pose_value = pose
        else:
            raise TypeError(f"pose must be a Pose or PoseStamped, but is a {type(pose).__name__}")

        target_to_source = self.lookup_transform(
            target_frame,
            _require_source_frame(source_frame),
            stamp=stamp,
            timeout=timeout,
        )
        transformed = PoseStamped()
        transformed.header = target_to_source.header
        transformed.pose = calc.mult_poses(
            conv.transform_to_pose(target_to_source.transform),
            pose_value,
        )
        return transformed

    def transform_point(
        self,
        point: Point | PointStamped,
        target_frame: str,
        source_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> PointStamped:
        """Transform a point into ``target_frame`` using this node's TF2 buffer."""
        if isinstance(point, PointStamped):
            source_frame = source_frame or point.header.frame_id
            stamp = stamp if stamp is not None else point.header.stamp
            point_value = point.point
        elif isinstance(point, Point):
            if source_frame is None:
                raise ValueError("source_frame is required when transforming an unstamped Point")
            point_value = point
        else:
            raise TypeError(f"point must be a Point or PointStamped, but is a {type(point).__name__}")

        target_to_source = self.lookup_transform(
            target_frame,
            _require_source_frame(source_frame),
            stamp=stamp,
            timeout=timeout,
        )
        transformed = PointStamped()
        transformed.header = target_to_source.header
        transformed.point = calc.transform_point(point_value, target_to_source.transform)
        return transformed

    def transform_vector3(
        self,
        vector: Vector3 | Vector3Stamped,
        target_frame: str,
        source_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
        timeout: Duration | float | int | None = 0.0,
    ) -> Vector3Stamped:
        """Transform a vector into ``target_frame`` using this node's TF2 buffer."""
        if isinstance(vector, Vector3Stamped):
            source_frame = source_frame or vector.header.frame_id
            stamp = stamp if stamp is not None else vector.header.stamp
            vector_value = vector.vector
        elif isinstance(vector, Vector3):
            if source_frame is None:
                raise ValueError("source_frame is required when transforming an unstamped Vector3")
            vector_value = vector
        else:
            raise TypeError(f"vector must be a Vector3 or Vector3Stamped, but is a {type(vector).__name__}")

        target_to_source = self.lookup_transform(
            target_frame,
            _require_source_frame(source_frame),
            stamp=stamp,
            timeout=timeout,
        )
        transformed = Vector3Stamped()
        transformed.header = target_to_source.header
        transformed.vector = calc.transform_vector3(vector_value, target_to_source.transform)
        return transformed

    def broadcast_transform(
        self,
        transform: Transform | TransformStamped | Pose | PoseStamped,
        parent_frame: str | None = None,
        child_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
    ) -> TransformStamped:
        """Broadcast a dynamic transform and return the stamped message that was sent."""
        transform_stamped = self._make_transform_stamped(
            transform,
            parent_frame=parent_frame,
            child_frame=child_frame,
            stamp=stamp,
        )
        self.tf_broadcaster.sendTransform(transform_stamped)
        return transform_stamped

    def broadcast_static_transform(
        self,
        transform: Transform | TransformStamped | Pose | PoseStamped,
        parent_frame: str | None = None,
        child_frame: str | None = None,
        stamp: Time | RclpyTime | None = None,
    ) -> TransformStamped:
        """Broadcast a static transform and return the stamped message that was sent."""
        transform_stamped = self._make_transform_stamped(
            transform,
            parent_frame=parent_frame,
            child_frame=child_frame,
            stamp=stamp,
        )
        self.static_tf_broadcaster.sendTransform(transform_stamped)
        return transform_stamped


def main(args=None):
    rclpy.init(args=args)
    node = TF2UtilsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
