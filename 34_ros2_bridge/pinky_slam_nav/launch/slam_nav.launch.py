"""slam_nav.launch.py - SLAM + Nav2 동시 실행 (지도 없이 바로 자율주행)

slam_toolbox가 실시간으로 지도를 만들며 map→odom TF를 게시하고,
Nav2는 그 지도를 그대로 받아 경로를 계획합니다 (AMCL/map_server 불필요).
RViz에서 "Nav2 Goal"을 찍으면 미탐사 영역으로도 주행하며 지도가 늘어납니다.

사용:
    ros2 launch pinky_slam_nav slam_nav.launch.py

참고:
    nav2_bringup의 bringup_launch.py에도 slam:=True 옵션이 있지만,
    여기서는 slam_toolbox(우리 파라미터) + navigation_launch.py(AMCL 제외
    내비게이션만)를 직접 조합해 구조가 눈에 보이도록 했습니다.
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

    use_sim_time = LaunchConfiguration("use_sim_time")
    params_file = LaunchConfiguration("params_file")
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
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

        # SLAM (RViz는 아래에서 nav 설정으로 하나만 띄움)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg_share, "launch", "slam.launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "use_rviz": "false",
            }.items(),
        ),

        # Nav2 내비게이션만 (map_server/AMCL 없음 — 지도는 slam_toolbox가 공급)
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(nav2_bringup_launch_dir, "navigation_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "params_file": params_file,
                "autostart": "True",
            }.items(),
        ),

        # RViz — nav 설정 (지도 + 코스트맵 + Goal 도구)
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg_share, "rviz", "nav.rviz")],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_rviz),
            output="screen",
        ),
    ])
