# TF2 Utils

Small ROS 2 helpers for `geometry_msgs` transforms, poses, quaternions, NumPy matrices, and common transform math.

## Examples

Open `rviz2`, add a tf display, and run the following examples:

```bash
ros2 run tf2_utils examples alignment
ros2 run tf2_utils examples reparenting
ros2 run tf2_utils examples tree
```

## Usage

Add this package to your package's `package.xml`:

```xml
<depend>tf2_utils</depend>
```

Use top-level imports for concise scripts:

```python
import tf2_utils

world_to_base = Transform(0.4, 0.0, 0.2)
base_to_tool = Transform(
    transform=Vector3(0.0, 0.0, 0.1),
    rotation=tf2_utils.euler_to_quat(0.0, 0.0, 0.5),
)
world_to_tool = tf2_utils.chain_transforms(world_to_base, base_to_tool)
tool_to_world = tf2_utils.invert_transform(world_to_tool)
matrix = tf2_utils.transform_to_np(world_to_tool)
```

Use module aliases when grouping matters:

```python
from tf2_utils import calc, comp, conv

transform = conv.list_to_transform([0.2, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
inverse = calc.invert_transform(transform)
same = comp.is_equal_transform(calc.mult_transforms(transform, inverse), Transform())
```

Use `TF2UtilsNode` when helpers need live TF2 lookup or broadcasting:

```python
from geometry_msgs.msg import Point
from tf2_utils import TF2UtilsNode, list_to_transform

node = TF2UtilsNode("example_tf_node")

node.broadcast_static_transform(
    list_to_transform([0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]),
    parent_frame="world",
    child_frame="base",
)

tool_pose = node.lookup_pose("world", "tool0", timeout=1.0)
point_in_world = node.transform_point(Point(x=0.0, y=0.0, z=0.1), "world", source_frame="tool0")
```

## Module Split

- `tf2_utils.node`: a `TF2UtilsNode` with a TF2 buffer, listener, dynamic/static broadcasters, and lookup/transform/broadcast helpers.
- `tf2_utils.conversions`: constructors, string/list conversions, NumPy matrices, Euler/quaternion helpers, stamped poses, and stamped transforms.
- `tf2_utils.calculations`: transform composition, chaining, inversion, distances, interpolation, normalization, and point/vector transforms.
- `tf2_utils.alignment`: orientation helpers for aiming transform axes at points/vectors and aligning cardinal planes to normals.
- `tf2_utils.comparisons`: tolerant equality checks for headers, geometry messages, and stamped transforms.
- `tf2_utils.tree`: convenience functions to create a tf tree from a list structure
- `tf2_utils.reparenting`: switch parents of a tf without affecting its global position

## Highlights

- Look up live TF2 transforms and convert frame transforms to `PoseStamped`.
- Transform stamped or unstamped poses, points, and vectors through a TF2 buffer.
- Broadcast dynamic or static transforms from `Transform`, `TransformStamped`, `Pose`, or `PoseStamped` inputs.
- Convert `Point`, `Vector3`, `Quaternion`, `Pose`, `Transform`, `PoseStamped`, and `TransformStamped` to and from NumPy-friendly forms.
- Convert between `Pose` and `Transform`, including stamped variants.
- Chain `Transform` or connected `TransformStamped` messages.
- Invert matrices, quaternions, poses, transforms, and stamped transforms.
- Interpolate positions, orientations, poses, and transforms.
- Align pose/transform axes toward points or vectors, with a secondary normal direction controlling roll.
- Align cardinal pose/transform planes to target normals.
- Compare quaternions as rotations, including the `q` / `-q` equivalence.
- Change parents of tfs without affecting their global positions.

## Colcon Pytest

```bash
colcon test --packages-select tf2_utils --event-handlers console_cohesion+   --pytest-args -rs -s
```