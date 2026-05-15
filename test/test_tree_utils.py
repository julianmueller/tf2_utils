import pytest
from geometry_msgs.msg import Point, Transform, Vector3

from tf2_utils import tree as tree_utils


TESTED_METHODS = {
    "flatten_transform_tree",
    "transform_tree",
}


@pytest.fixture
def base_transform():
    return Transform(translation=Vector3(x=1.0, y=0.0, z=0.0))


@pytest.fixture
def tool_transform():
    return Transform(translation=Vector3(x=0.0, y=2.0, z=0.0))


def test_all_public_tree_methods_are_covered():
    assert set(tree_utils.__all__) == TESTED_METHODS


def test_transform_tree_helpers_flatten_nested_frames(base_transform, tool_transform):
    tree = tree_utils.transform_tree(
        "world",
        children=[
            tree_utils.transform_tree(
                "base",
                base_transform,
                children=[
                    tree_utils.transform_tree("tool", tool_transform),
                ],
            ),
            tree_utils.transform_tree("fixture"),
        ],
    )

    transforms = tree_utils.flatten_transform_tree("world", tree)

    assert [(transform.header.frame_id, transform.child_frame_id) for transform in transforms] == [
        ("world", "base"),
        ("base", "tool"),
        ("world", "fixture"),
    ]
    assert [
        transforms[0].transform.translation.x,
        transforms[0].transform.translation.y,
        transforms[0].transform.translation.z,
    ] == pytest.approx([1.0, 0.0, 0.0])
    assert transforms[2].transform.rotation.w == pytest.approx(1.0)


def test_transform_tree_helpers_reject_invalid_trees():
    with pytest.raises(ValueError):
        tree_utils.transform_tree("")
    with pytest.raises(TypeError):
        tree_utils.transform_tree("bad", transform=Point())
    with pytest.raises(ValueError):
        tree_utils.flatten_transform_tree("", tree_utils.transform_tree("root"))
    with pytest.raises(ValueError):
        tree_utils.flatten_transform_tree(
            "world",
            tree_utils.transform_tree(
                "root",
                children=[
                    tree_utils.transform_tree("duplicate"),
                    tree_utils.transform_tree("duplicate"),
                ],
            ),
        )
    with pytest.raises(TypeError):
        tree_utils.flatten_transform_tree(
            "world",
            tree_utils.transform_tree(
                "root",
                children=[("child", Transform(), "bad")],
            ),
        )
