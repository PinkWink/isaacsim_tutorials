"""nav.launch.py - 저장된 지도 + AMCL + Nav2 (2단계 흐름의 2단계)

slam.launch.py로 만든 지도를 불러와 AMCL로 로컬라이즈하고
Nav2로 자율주행합니다. RViz의 "Nav2 Goal"로 목표를 지정하세요.

사용 (map은 절대경로 권장):
    ros2 launch pinky_slam_nav nav.launch.py \
      map:=/home/pw/isaac/isaacsim_tutorials/34_ros2_bridge/pinky_slam_nav/maps/maze.yaml

초기 위치:
    지도를 로봇 스폰 상태에서 만들기 시작했다면 초기 포즈 (0,0,0)이 자동 설정됩니다.
    위치가 어긋나 보이면 RViz의 "2D Pose Estimate"로 다시 지정하세요.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("pinky_slam_nav")
    nav2_bringup_launch_dir = os.path.join(
        get_package_share_directory("nav2_bringup"), "launch"
    )

    map_yaml = LaunchConfiguration("map")
    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "map",
            description="저장된 지도 yaml (절대경로), 예: .../pinky_slam_nav/maps/maze.yaml",
        ),
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="IsaacSim의 /clock(sim time) 사용",
        ),
        DeclareLaunchArgument(
            "params_file",
            default_value=os.path.join(pkg_share, "config", "nav2_params.yaml"),
            description="Nav2 파라미터 파일",
        ),
        DeclareLaunchArgument(
            "use_rviz", default_value="true",
            description="RViz 실행 여부",
        ),

        # Nav2 전체 스택: map_server + AMCL(로컬라이제이션) + planner/controller/...
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_launch_dir, "bringup_launch.py")
            ),
            launch_arguments={
                "map": map_yaml,
                "use_sim_time": use_sim_time,
                "params_file": params_file,
                "autostart": "True",
                "slam": "False",
            }.items(),
        ),

        # RViz — 지도/코스트맵/경로/Goal 도구
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg_share, "rviz", "nav.rviz")],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_rviz),
            output="screen",
        ),
    ])
