"""Executable runner for tf2_utils examples."""

from __future__ import annotations

import argparse
import time

import rclpy

from geometry_msgs.msg import Transform

from tf2_utils import TF2UtilsNode
from tf2_utils.examples import alignment, reparenting, tree


EXAMPLE_MODULES = {
    "alignment": alignment,
    "reparenting": reparenting,
    "tree": tree,
}


def _static_reference_frames(example_module, parent_frame: str) -> list[tuple[str, str, Transform]]:
    """Return static helper frames for an example module."""
    if hasattr(example_module, "static"):
        return example_module.static(parent_frame)
    return [
        (parent_frame, child_frame, transform)
        for child_frame, transform in getattr(example_module, "BASE_TARGETS", {}).items()
    ]


def _broadcast_static_reference_frames(node: TF2UtilsNode, transforms: list[tuple[str, str, Transform]]) -> None:
    """Publish static helper frames for the selected example."""
    for parent_frame, child_frame, transform in transforms:
        node.broadcast_static_transform(
            transform,
            parent_frame=parent_frame,
            child_frame=child_frame,
        )


def _broadcast_dynamic_examples(
    node: TF2UtilsNode,
    transforms: list[tuple[str, str, Transform]],
    rate_hz: float,
    end_time: float,
) -> None:
    """Publish example transforms until ``end_time``."""
    period = 1.0 / max(rate_hz, 0.1)

    def _broadcast_once() -> None:
        for parent_frame, child_frame, transform in transforms:
            node.broadcast_transform(
                transform,
                parent_frame=parent_frame,
                child_frame=child_frame,
            )

    _broadcast_once()
    while rclpy.ok() and time.monotonic() < end_time:
        rclpy.spin_once(node, timeout_sec=0.0)
        time.sleep(period)
        _broadcast_once()


def main() -> None:
    """Broadcast tf2_utils example frames."""
    parser = argparse.ArgumentParser(description="Broadcast tf2_utils examples.")
    parser.add_argument(
        "example",
        nargs="?",
        choices=sorted(EXAMPLE_MODULES),
        default="alignment",
        help="Example scene to broadcast.",
    )
    parser.add_argument("--frame", default="world", help="Root frame ID used by the example transforms.")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to keep broadcasting dynamic transforms.")
    parser.add_argument("--rate", type=float, default=10.0, help="Dynamic transform broadcast rate in Hz.")
    args = parser.parse_args()

    rclpy.init()
    node = TF2UtilsNode(f"tf2_{args.example}_examples")
    try:
        example_module = EXAMPLE_MODULES[args.example]
        _broadcast_static_reference_frames(node, _static_reference_frames(example_module, args.frame))
        transforms = example_module.create(args.frame)
        end_time = time.monotonic() + max(args.duration, 0.0)
        _broadcast_dynamic_examples(node, transforms, args.rate, end_time)
    finally:
        node.destroy_node()
        rclpy.shutdown()
