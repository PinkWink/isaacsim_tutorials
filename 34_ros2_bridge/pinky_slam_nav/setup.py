import os
from glob import glob

from setuptools import setup

package_name = "pinky_slam_nav"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
        (os.path.join("share", package_name, "maps"), glob("maps/*.yaml") + glob("maps/*.pgm")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="pinkwink",
    maintainer_email="pinkwink.korea@gmail.com",
    description="Pinky Pro SLAM + Nav2 실습 패키지 (IsaacSim 34-1강 연동)",
    license="MIT",
    entry_points={"console_scripts": []},
)
