"""slam.launch.py - slam_toolbox(online_async) + RViz

IsaacSim(34_1_slam_nav.py)이 게시하는 /scan /odom /tf /clock을 이용해
지도를 실시간으로 작성합니다. slam_toolbox가 map→odom TF를 게시합니다.

사용:
    ros2 launch pinky_slam_nav slam.launch.py
    ros2 launch pinky_slam_nav slam.launch.py use_rviz:=false     # RViz 없이

지도 저장 (매핑 주행 완료 후, slam이 켜져 있는 상태에서):
    ros2 run nav2_map_server map_saver_cli \
      -f <절대경로>/pinky_slam_nav/maps/maze --ros-args -p use_sim_time:=true
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
    slam_toolbox_launch_dir = os.path.join(
        get_package_share_directory("slam_toolbox"), "launch"
    )

    use_sim_time = LaunchConfiguration("use_sim_time")
    slam_params_file = LaunchConfiguration("slam_params_file")
    use_rviz = LaunchConfiguration("use_rviz")

    return LaunchDescription([
        DeclareLaunchArgument(
            "use_sim_time", default_value="true",
            description="IsaacSim의 /clock(sim time) 사용",
        ),
        DeclareLaunchArgument(
            "slam_params_file",
            default_value=os.path.join(pkg_share, "config", "slam_params.yaml"),
            description="slam_toolbox 파라미터 파일",
        ),
        DeclareLaunchArgument(
            "use_rviz", default_value="true",
            description="RViz 실행 여부",
        ),

        # slam_toolbox online_async — /scan + odom→base_footprint TF로 지도 작성,
        # map→odom TF 게시
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(slam_toolbox_launch_dir, "online_async_launch.py")
            ),
            launch_arguments={
                "use_sim_time": use_sim_time,
                "slam_params_file": slam_params_file,
            }.items(),
        ),

        # RViz — 지도/스캔/TF 시각화
        Node(
            package="rviz2",
            executable="rviz2",
            arguments=["-d", os.path.join(pkg_share, "rviz", "slam.rviz")],
            parameters=[{"use_sim_time": use_sim_time}],
            condition=IfCondition(use_rviz),
            output="screen",
        ),
    ])
