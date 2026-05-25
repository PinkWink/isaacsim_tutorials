"""
34_ros2_bridge.py - Pinky Pro × ROS2 Jazzy 통합 (IsaacSim ROS2 Bridge)

본 강의는 "Gazebo 대신 IsaacSim"이라는 시나리오를 그대로 재현합니다.
IsaacSim의 isaacsim.ros2.bridge 익스텐션을 켜고, OmniGraph(Action Graph)에
ROS2 노드들을 배치하여 시뮬레이터를 ROS2 그래프의 한 구성원으로 만듭니다.

게시(publish) — IsaacSim → ROS2:
  - /clock            (rosgraph_msgs/Clock)              ← sim time
  - /tf               (tf2_msgs/TFMessage)               ← 로봇 링크 TF 트리
  - /joint_states     (sensor_msgs/JointState)           ← 모든 관절
  - /odom             (nav_msgs/Odometry)                ← 베이스 오도메트리
  - /scan             (sensor_msgs/LaserScan)            ← RTX 2D LiDAR

구독(subscribe) — ROS2 → IsaacSim:
  - /cmd_vel          (geometry_msgs/Twist)              ← 차동 구동 명령

왜 IsaacSim의 rclpy/외부 ROS2를 따로 쓰지 않는가:
  - IsaacLab venv는 Python 3.11, Ubuntu 24.04의 ROS2 Jazzy는 Python 3.12 기반.
  - 직접 `import rclpy`는 ABI 충돌이 발생할 수 있어, 대신 IsaacSim에 내장된 ROS2
    브릿지(C++ 확장)가 DDS 버스에 직접 토픽을 띄웁니다.
  - 외부에서는 시스템 ROS2(Jazzy)의 ros2/rviz2/teleop/nav2가 그대로 동작합니다.

호스트(별도 셸)에서:
  source /opt/ros/jazzy/setup.bash
  ros2 topic list                                      # /clock /tf /joint_states /odom /scan /cmd_vel
  ros2 topic echo /scan --once
  ros2 run teleop_twist_keyboard teleop_twist_keyboard # 키보드 주행 (use_sim_time 권장)
  rviz2                                                # Fixed Frame=odom, LaserScan, TF, RobotModel

실행:  ※ ROS2 sourcing이 IsaacSim 실행보다 먼저여야 합니다.
  source /opt/ros/jazzy/setup.bash       # ROS_DISTRO=jazzy, LD_LIBRARY_PATH 설정
  source env_isaaclab/bin/activate
  python lectures/34_ros2_bridge/34_ros2_bridge.py
  # headless는 비추 — RViz로 시각화 확인용 강의이므로 GUI 권장
"""

# ── 1. AppLauncher (다른 모든 import보다 먼저) ────────────────────────────
import argparse
import os
import sys

# ── 0. 환경 사전 점검 ────────────────────────────────────────────────────
# IsaacSim ROS2 브릿지의 jazzy 번들 lib에는 libament_index_cpp.so가 빠져 있어
# /opt/ros/jazzy/setup.bash를 먼저 source하지 않으면 브릿지 startup이 실패합니다.
# 실패하면 RtxLidar...PublishLaserScan 같은 라이터도 등록되지 않아 후속 에러가 줄줄이 납니다.
_ros_distro = os.environ.get("ROS_DISTRO", "")
if _ros_distro != "jazzy":
    sep = "=" * 70
    msg = (
        f"\n{sep}\n"
        f"[ERROR] ROS_DISTRO 환경변수가 'jazzy'가 아닙니다 (현재: '{_ros_distro}').\n"
        f"        IsaacSim ROS2 브릿지의 번들 jazzy lib에는 libament_index_cpp.so가\n"
        f"        빠져 있어, 시스템 ROS2를 먼저 source하지 않으면 브릿지 startup이 실패합니다.\n"
        f"\n"
        f"        실행 전에 다음 순서로 환경을 준비하세요:\n"
        f"            source /opt/ros/jazzy/setup.bash\n"
        f"            source env_isaaclab/bin/activate    # 그 다음에 venv\n"
        f"            python lectures/34_ros2_bridge/34_ros2_bridge.py\n"
        f"{sep}\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="34 - Pinky Pro × ROS2 Jazzy 브릿지")
parser.add_argument(
    "--cmd_topic", type=str, default="cmd_vel",
    help="Twist 구독 토픽 (기본: cmd_vel)",
)
parser.add_argument(
    "--scan_topic", type=str, default="scan",
    help="LaserScan 게시 토픽 (기본: scan)",
)
parser.add_argument(
    "--robot_namespace", type=str, default="",
    help="모든 ROS2 토픽/프레임에 붙일 네임스페이스 (예: 'pinky')",
)
parser.add_argument(
    "--with_rsp", action="store_true",
    help="호스트에서 robot_state_publisher를 띄울 때 사용. 이 모드에서는 IsaacSim의 "
         "정적 TF(base_footprint→base_link, base_link→rplidar_link)를 게시하지 않아 "
         "rsp가 URDF에서 발행하는 TF와 충돌하지 않습니다 (odom→base_footprint 동적 "
         "TF만 IsaacSim이 담당).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. ROS2 브릿지 활성화 ────────────────────────────────────────────────
# isaacsim.ros2.bridge: IsaacSim과 함께 빌드된 C++ ROS2 클라이언트.
# 시스템 ROS2 Jazzy 빌드와 별개로 DDS 버스에 토픽/서비스를 직접 띄웁니다.
from isaacsim.core.utils.extensions import enable_extension

if not enable_extension("isaacsim.ros2.bridge"):
    print("[ERROR] isaacsim.ros2.bridge 익스텐션을 활성화할 수 없습니다.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)
simulation_app.update()

# ── 3. IsaacSim / OmniGraph import ──────────────────────────────────────
import numpy as np
import omni
import omni.graph.core as og
import omni.kit.commands
import omni.replicator.core as rep
import usdrt.Sdf
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Usd, UsdPhysics

# ── 4. Pinky Pro USD 경로 ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "31_urdf_to_usd", "usd_output", "pinky_pro.usd"
))
assert os.path.exists(USD_PATH), (
    f"Pinky USD 파일 없음: {USD_PATH}\n먼저 31강(URDF→USD 변환)을 실행하세요."
)

# Pinky 차동 구동 파라미터 (32강과 동일)
WHEEL_RADIUS = 0.028   # [m]
WHEEL_DISTANCE = 0.0811  # [m]
LEFT_WHEEL_JOINT = "l_wheel_joint"
RIGHT_WHEEL_JOINT = "r_wheel_joint"

# 프림 경로 / TF 프레임 이름
ROBOT_PRIM = "/World/pinky"
LIDAR_FRAME = "rplidar_link"   # 33강에서 사용한 URDF link 이름과 일치
# LIDAR_PRIM은 articulation root 발견 후 동적으로 결정 (rplidar_link 위치가
# 변환 결과에 따라 /World/pinky/rplidar_link 또는 /World/pinky/<root>/rplidar_link)

# ── 5. 월드 + 로봇 + 지면 ────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

# Pinky USD를 단일 인스턴스로 로드 (IsaacLab의 멀티 env 패턴 안 씀 — ROS2는 1대 기준)
add_reference_to_stage(USD_PATH, ROBOT_PRIM)
simulation_app.update()

# ── 5b. Articulation root / 링크 prim 동적 탐색 ──────────────────────────
# URDF→USD 변환기는 ArticulationRootAPI를 보통 base_link에 붙입니다.
# 정확한 경로는 변환 옵션/버전에 따라 다르므로 USD stage를 traverse해서 찾습니다.
def _find_first_prim_with_api(root_prim: Usd.Prim, api_type) -> Usd.Prim | None:
    """root_prim 아래(자신 포함)를 깊이우선 탐색해 주어진 API가 적용된 첫 prim을 반환."""
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI(api_type):
            return prim
    return None


def _find_child_by_name(root_prim: Usd.Prim, name: str) -> Usd.Prim | None:
    """root_prim 아래(자신 포함)에서 prim name이 정확히 일치하는 첫 prim 반환."""
    for prim in Usd.PrimRange(root_prim):
        if prim.GetName() == name:
            return prim
    return None


stage = omni.usd.get_context().get_stage()
robot_root_prim = stage.GetPrimAtPath(ROBOT_PRIM)
if not robot_root_prim.IsValid():
    raise RuntimeError(f"로봇 prim을 찾을 수 없습니다: {ROBOT_PRIM}")

art_prim = _find_first_prim_with_api(robot_root_prim, UsdPhysics.ArticulationRootAPI)
if art_prim is None:
    raise RuntimeError(
        f"{ROBOT_PRIM} 아래에서 ArticulationRootAPI가 붙은 prim을 찾지 못했습니다.\n"
        f"31강 USD 변환 결과를 확인하세요 (보통 base_link에 자동 부착됨)."
    )
ART_ROOT = str(art_prim.GetPath())

lidar_link_prim = _find_child_by_name(robot_root_prim, LIDAR_FRAME)
if lidar_link_prim is None:
    raise RuntimeError(
        f"LiDAR가 부착될 '{LIDAR_FRAME}' 링크를 {ROBOT_PRIM} 아래에서 찾지 못했습니다."
    )
LIDAR_PRIM = str(lidar_link_prim.GetPath()) + "/Lidar"

print(f"[INFO] 발견된 articulation root: {ART_ROOT}")
print(f"[INFO] LiDAR 부착 위치:         {LIDAR_PRIM}")
simulation_app.update()

# ── 5b-2. 휠 조인트에 velocity drive 명시 설정 ──────────────────────────
# URDF→USD 변환기는 continuous 조인트를 만들지만, UsdPhysics.DriveAPI는
# 자동으로 채워주지 않을 수 있습니다. drive가 없으면 ArticulationController가
# velocityCommand를 보내도 휠이 무반응 — /cmd_vel 다 받아도 안 움직임.
# 32강의 IsaacLab ImplicitActuatorCfg(stiffness=0.0, damping=10.0)와 동등하게
# 직접 USD에 박아 넣습니다.
def _setup_wheel_velocity_drive(stage, joint_name: str,
                                stiffness: float = 0.0,
                                damping: float = 10.0,
                                max_force: float = 100.0) -> Usd.Prim | None:
    for prim in stage.Traverse():
        if prim.GetName() != joint_name:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        drive = UsdPhysics.DriveAPI.Apply(prim, "angular")
        drive.CreateTypeAttr().Set("force")
        drive.CreateStiffnessAttr().Set(stiffness)
        drive.CreateDampingAttr().Set(damping)
        drive.CreateMaxForceAttr().Set(max_force)
        print(f"[INFO] {joint_name} 휠 drive 설정: stiffness={stiffness}, "
              f"damping={damping} @ {prim.GetPath()}")
        return prim
    print(f"[WARN] 조인트 '{joint_name}'를 stage에서 못 찾음 (drive 미설정)")
    return None


_setup_wheel_velocity_drive(stage, LEFT_WHEEL_JOINT)
_setup_wheel_velocity_drive(stage, RIGHT_WHEEL_JOINT)
simulation_app.update()

# ── 5c. 간단한 미로(벽) 배치 ────────────────────────────────────────────
# LiDAR가 실제로 무엇인가 맞혀야 /scan 출력이 비어보이지 않습니다.
# 4m × 4m 외벽 + 내부 장애물 두 개로 간단한 미로를 만듭니다.
# 모든 벽은 FixedCuboid(static collider, 비이동) — Pinky가 부딪혀도 안 움직임.
def add_wall(prim_path: str, center, size, color=(0.55, 0.55, 0.6)):
    """center=(x,y,z), size=(sx,sy,sz) 의 정적 벽(콜리전 + 시각화) 추가."""
    FixedCuboid(
        prim_path=prim_path,
        name=prim_path.split("/")[-1],
        position=np.array(center, dtype=np.float32),
        scale=np.array(size, dtype=np.float32),
        color=np.array(color, dtype=np.float32),
    )


WALL_H = 0.5      # 벽 높이 [m]
WALL_T = 0.1      # 벽 두께 [m]
ROOM = 4.0        # 외벽 한 변 길이 [m]
# 외벽 4면 (방의 중심은 원점)
add_wall("/World/walls/north", (0.0,  ROOM / 2, WALL_H / 2), (ROOM + WALL_T, WALL_T, WALL_H))
add_wall("/World/walls/south", (0.0, -ROOM / 2, WALL_H / 2), (ROOM + WALL_T, WALL_T, WALL_H))
add_wall("/World/walls/east",  ( ROOM / 2, 0.0, WALL_H / 2), (WALL_T, ROOM + WALL_T, WALL_H))
add_wall("/World/walls/west",  (-ROOM / 2, 0.0, WALL_H / 2), (WALL_T, ROOM + WALL_T, WALL_H))
# 내부 장애물 — 미로 느낌
add_wall("/World/walls/inner_a", ( 0.6, 0.5, WALL_H / 2), (1.4, WALL_T, WALL_H))
add_wall("/World/walls/inner_b", (-0.8, -0.4, WALL_H / 2), (WALL_T, 1.4, WALL_H))

print(f"[INFO] {ROOM}m × {ROOM}m 미로 배치 완료 (외벽 4 + 내부 2)")
simulation_app.update()

# ── 6. RTX 2D LiDAR 부착 (rplidar_link 아래) ────────────────────────────
# IsaacSim의 RTX 센서는 카메라 계열 GPU 센서로, ROS2 LaserScan 게시를 위한
# 전용 헬퍼(rep.writers의 "RtxLidarROS2PublishLaserScan")가 제공됩니다.
# 33강의 RayCaster(가벼운 CPU 레이캐스팅)와 달리 LaserScan 메시지를 직접 만들어줍니다.
_, lidar_sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path=LIDAR_PRIM,
    parent=None,
    config="Example_Rotary_2D",           # 평면 2D LiDAR — sensor_msgs/LaserScan 호환
    translation=(0.0, 0.0, 0.0),          # rplidar_link 원점 기준
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
)

# RTX 센서는 카메라처럼 별도 render product가 필요합니다.
hydra_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="PinkyLidar")

# /scan 퍼블리셔 (LaserScan)
scan_writer = rep.writers.get("RtxLidar" + "ROS2PublishLaserScan")
scan_writer.initialize(topicName=args_cli.scan_topic, frameId=LIDAR_FRAME)
scan_writer.attach([hydra_texture])

# (참고) 디버그 시각화 — 뷰포트에 포인트 클라우드 점이 보임
debug_writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
debug_writer.attach([hydra_texture])

simulation_app.update()

# ── 7. ROS2 OmniGraph 구축 ──────────────────────────────────────────────
# OmniGraph(=Action Graph)는 IsaacSim의 데이터플로우 그래프 시스템.
# 매 시뮬레이션 tick마다 OnPlaybackTick이 발화하고, 연결된 ROS2 노드들이
# 그 tick에 맞춰 토픽을 게시/구독합니다.
GRAPH_PATH = "/ActionGraph"

# ── 그래프 노드/연결/값 리스트 빌드 (with_rsp 모드면 정적 TF 제외) ──────
# 정적 TF(base_footprint→base_link, base_link→rplidar_link)는 호스트의
# robot_state_publisher가 URDF에서 그대로 발행하므로, --with_rsp 모드에서는
# IsaacSim 쪽 두 노드를 통째로 빼서 TF "multiple parents" 충돌을 막습니다.
# (동적 odom→base_footprint는 IsaacSim만 알 수 있어 항상 IsaacSim이 게시.)
_use_static_tf = not args_cli.with_rsp

_create_nodes = [
    # 트리거 + sim time
    ("OnTick", "omni.graph.action.OnPlaybackTick"),
    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
    # ROS2 컨텍스트 (모든 ROS2 노드의 공통 소켓)
    ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
    # 퍼블리셔 (항상 게시)
    ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ("PubTfOdomFp", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
    ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
    # 구독자 + 차동 구동 + 관절 적용
    ("SubTwist", "isaacsim.ros2.bridge.ROS2SubscribeTwist"),
    ("BreakLin", "omni.graph.nodes.BreakVector3"),
    ("BreakAng", "omni.graph.nodes.BreakVector3"),
    ("DiffDrive", "isaacsim.robot.wheeled_robots.DifferentialController"),
    ("ArticController", "isaacsim.core.nodes.IsaacArticulationController"),
]
if _use_static_tf:
    _create_nodes += [
        ("PubTfFpBase", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
        ("PubTfBaseLidar", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ]

_connect = [
    # /clock
    ("OnTick.outputs:tick", "PubClock.inputs:execIn"),
    ("Ros2Context.outputs:context", "PubClock.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
    # /joint_states
    ("OnTick.outputs:tick", "PubJointState.inputs:execIn"),
    ("Ros2Context.outputs:context", "PubJointState.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PubJointState.inputs:timeStamp"),
    # ComputeOdom → PubOdom + PubTfOdomFp (동적 odom→base_footprint)
    ("OnTick.outputs:tick", "ComputeOdom.inputs:execIn"),
    ("ComputeOdom.outputs:execOut", "PubOdom.inputs:execIn"),
    ("ComputeOdom.outputs:execOut", "PubTfOdomFp.inputs:execIn"),
    ("Ros2Context.outputs:context", "PubOdom.inputs:context"),
    ("Ros2Context.outputs:context", "PubTfOdomFp.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PubOdom.inputs:timeStamp"),
    ("ReadSimTime.outputs:simulationTime", "PubTfOdomFp.inputs:timeStamp"),
    ("ComputeOdom.outputs:position", "PubOdom.inputs:position"),
    ("ComputeOdom.outputs:orientation", "PubOdom.inputs:orientation"),
    ("ComputeOdom.outputs:linearVelocity", "PubOdom.inputs:linearVelocity"),
    ("ComputeOdom.outputs:angularVelocity", "PubOdom.inputs:angularVelocity"),
    ("ComputeOdom.outputs:position", "PubTfOdomFp.inputs:translation"),
    ("ComputeOdom.outputs:orientation", "PubTfOdomFp.inputs:rotation"),
    # /cmd_vel → BreakVector3 → DiffDrive → ArticulationController
    ("OnTick.outputs:tick", "SubTwist.inputs:execIn"),
    ("Ros2Context.outputs:context", "SubTwist.inputs:context"),
    ("SubTwist.outputs:linearVelocity", "BreakLin.inputs:tuple"),
    ("SubTwist.outputs:angularVelocity", "BreakAng.inputs:tuple"),
    ("SubTwist.outputs:execOut", "DiffDrive.inputs:execIn"),
    ("BreakLin.outputs:x", "DiffDrive.inputs:linearVelocity"),
    ("BreakAng.outputs:z", "DiffDrive.inputs:angularVelocity"),
    ("DiffDrive.outputs:velocityCommand", "ArticController.inputs:velocityCommand"),
    ("OnTick.outputs:tick", "ArticController.inputs:execIn"),
]
if _use_static_tf:
    _connect += [
        # base_footprint → base_link 정적 TF (URDF z=0.028)
        ("OnTick.outputs:tick", "PubTfFpBase.inputs:execIn"),
        ("Ros2Context.outputs:context", "PubTfFpBase.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PubTfFpBase.inputs:timeStamp"),
        # base_link → rplidar_link 정적 TF (URDF 합성 오프셋)
        ("OnTick.outputs:tick", "PubTfBaseLidar.inputs:execIn"),
        ("Ros2Context.outputs:context", "PubTfBaseLidar.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PubTfBaseLidar.inputs:timeStamp"),
    ]

_set_values = [
    # /clock
    ("PubClock.inputs:topicName", "clock"),
    ("PubClock.inputs:nodeNamespace", args_cli.robot_namespace),
    # /joint_states (articulation root를 타깃)
    ("PubJointState.inputs:topicName", "joint_states"),
    ("PubJointState.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubJointState.inputs:targetPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    # /tf: odom → base_footprint (동적, IsaacSim만 알 수 있는 정보)
    ("PubTfOdomFp.inputs:topicName", "tf"),
    ("PubTfOdomFp.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubTfOdomFp.inputs:parentFrameId", "odom"),
    ("PubTfOdomFp.inputs:childFrameId", "base_footprint"),
    ("PubTfOdomFp.inputs:staticPublisher", False),
    # /odom 메시지 (ComputeOdom은 articulation root=base_footprint의 포즈 계산)
    ("PubOdom.inputs:topicName", "odom"),
    ("PubOdom.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubOdom.inputs:odomFrameId", "odom"),
    ("PubOdom.inputs:chassisFrameId", "base_footprint"),
    ("ComputeOdom.inputs:chassisPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    # /cmd_vel
    ("SubTwist.inputs:topicName", args_cli.cmd_topic),
    ("SubTwist.inputs:nodeNamespace", args_cli.robot_namespace),
    # DifferentialController — 출력 velocityCommand = [left_rad_s, right_rad_s]
    ("DiffDrive.inputs:wheelRadius", WHEEL_RADIUS),
    ("DiffDrive.inputs:wheelDistance", WHEEL_DISTANCE),
    ("DiffDrive.inputs:maxLinearSpeed", 0.5),
    ("DiffDrive.inputs:maxAngularSpeed", 2.0),
    ("DiffDrive.inputs:maxWheelSpeed", 20.0),
    # ArticulationController — jointNames 순서가 DiffDrive 출력([left, right])과 일치
    ("ArticController.inputs:targetPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    ("ArticController.inputs:jointNames", [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]),
]
if _use_static_tf:
    _set_values += [
        # /tf_static: base_footprint → base_link (URDF base_link_fixed_joint xyz=0,0,0.028)
        ("PubTfFpBase.inputs:topicName", "tf_static"),
        ("PubTfFpBase.inputs:nodeNamespace", args_cli.robot_namespace),
        ("PubTfFpBase.inputs:parentFrameId", "base_footprint"),
        ("PubTfFpBase.inputs:childFrameId", "base_link"),
        ("PubTfFpBase.inputs:translation", [0.0, 0.0, 0.028]),
        ("PubTfFpBase.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),  # (x,y,z,w) identity
        ("PubTfFpBase.inputs:staticPublisher", True),
        # /tf_static: base_link → rplidar_link (URDF 두 fixed joint 합성)
        #   base_link --(rplidar_mount_fixed_joint xyz=-0.017,0,0.067)--> rplidar_mount
        #   rplidar_mount --(rplidar_link_fixed_joint xyz=0,0,0.030 rpy=0,0,π)--> rplidar_link
        ("PubTfBaseLidar.inputs:topicName", "tf_static"),
        ("PubTfBaseLidar.inputs:nodeNamespace", args_cli.robot_namespace),
        ("PubTfBaseLidar.inputs:parentFrameId", "base_link"),
        ("PubTfBaseLidar.inputs:childFrameId", LIDAR_FRAME),  # = "rplidar_link"
        ("PubTfBaseLidar.inputs:translation", [-0.017, 0.0, 0.097]),
        ("PubTfBaseLidar.inputs:rotation", [0.0, 0.0, 1.0, 0.0]),  # Z축 180°
        ("PubTfBaseLidar.inputs:staticPublisher", True),
    ]

og.Controller.edit(
    {"graph_path": GRAPH_PATH, "evaluator_name": "execution"},
    {
        og.Controller.Keys.CREATE_NODES: _create_nodes,
        og.Controller.Keys.CONNECT: _connect,
        og.Controller.Keys.SET_VALUES: _set_values,
    },
)

simulation_app.update()
simulation_app.update()

# ── 8. 시뮬레이션 시작 ──────────────────────────────────────────────────
print("=" * 70)
print("[INFO] Pinky Pro × ROS2 Jazzy 브릿지 준비 완료")
print(f"  로봇 wrapper:     {ROBOT_PRIM}")
print(f"  Articulation 루트: {ART_ROOT}")
print(f"  LiDAR prim:       {LIDAR_PRIM}")
print(f"  바퀴 반경/축거리: {WHEEL_RADIUS}m / {WHEEL_DISTANCE}m")
print(f"  ROS2 네임스페이스: '{args_cli.robot_namespace or '(none)'}'")
print(f"  TF 모드:          {'독립 (정적 TF도 IsaacSim 게시)' if _use_static_tf else 'with-rsp (정적 TF는 robot_state_publisher가 담당)'}")
print()
static_tf_str = "/tf_static " if _use_static_tf else ""
print("  게시 토픽:  /clock /tf {}/joint_states /odom /{}".format(static_tf_str, args_cli.scan_topic))
print("  구독 토픽:  /{}".format(args_cli.cmd_topic))
print()
print("  호스트에서 (별도 셸):")
print("    source /opt/ros/jazzy/setup.bash")
print("    ros2 topic list")
print("    ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -p use_sim_time:=true")
print("    rviz2  # Fixed Frame: odom, 추가: LaserScan(/scan), TF, Odometry(/odom)")
if not _use_static_tf:
    print()
    print("  --with_rsp 모드에서는 호스트에서 robot_state_publisher도 띄우세요:")
    print("    (sed로 mesh의 절대 경로를 file:// URI로 변환 — RViz가 메시를 찾을 수 있도록)")
    print("    ros2 run robot_state_publisher robot_state_publisher \\")
    print("      --ros-args -p use_sim_time:=true \\")
    print("      -p robot_description:=\"$(sed 's|filename=\\\"/|filename=\\\"file:///|g' \\")
    print("        ~/isaac/isaacsim_tutorials/30_urdf_preparation/pinky_pro.urdf)\"")
print("=" * 70)

world.reset()
# play()를 명시적으로 호출해야 timeline이 시작되고, RTX LiDAR 렌더 파이프라인과
# OmniGraph tick이 실제로 발화합니다. reset()만으로는 timeline이 정지 상태라
# /scan 라이터가 호출될 일이 없어 토픽이 등록조차 안 됩니다 (가장 흔한 함정).
world.play()

# ── 9. 메인 루프 ────────────────────────────────────────────────────────
# 외부 ROS2 노드(teleop, rviz, nav2 등)는 별도 프로세스에서 자유롭게 붙고 떨어집니다.
# 본 스크립트는 그저 시뮬레이션을 돌리고 OmniGraph가 매 tick마다 토픽을 처리하도록 유지합니다.
try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    print("\n[INFO] Ctrl-C: 시뮬레이션 종료")
finally:
    simulation_app.close()
    print("[DONE] 34강 종료")
