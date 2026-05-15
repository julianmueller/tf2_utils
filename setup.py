from pathlib import Path
from typing import Sequence

from setuptools import find_packages, setup

package_name = "tf2_utils"


def get_data_files(folder: str, file_endings: Sequence[str] = [""]) -> tuple[str, list[str]]:
    folder_path = Path("share") / package_name / folder
    files = []
    for ext in file_endings:
        files.extend(str(f) for f in Path(folder).glob(ext))
    return (str(folder_path), files)


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        get_data_files("launch", ["*.launch.py"]),
        get_data_files("rviz", ["*.rviz"]),
        get_data_files("config", ["*.yaml", "*.rviz"]),
    ],
    install_requires=["numpy", "scipy", "setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="jmueller",
    maintainer_email="julian.mueller@iwb.tum.de",
    description="Reusable ROS 2 helpers for transforms, NumPy conversion, and transform math.",
    license="Apache-2.0",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "node = tf2_utils.node:main",
            "examples = tf2_utils.examples:main",
        ],
    },
)
