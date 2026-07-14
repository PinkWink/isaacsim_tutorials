"""
34_1_slam_nav.py - Pinky Pro SLAM + Nav2 자율주행용 IsaacSim 환경

34번(ROS2 브릿지)의 확장판입니다. 브릿지 구성(토픽/TF/OmniGraph)은 34번과
동일하지만, SLAM과 자율주행 실습에 맞게 세 가지가 다릅니다:

  1. LiDAR: Example_Rotary_2D → RPLIDAR_S2E
     Example_Rotary_2D는 최소 감지 거리 1.0m + 빔이 -2° 아래로 기울어 있어
     좁은 미로에서는 벽이 안 보이거나 바닥을 맞힙니다. RPLIDAR_S2E는
     0.05~30m, 10Hz, 수평 빔 — 실제 Pinky의 RPLIDAR 계열이기도 합니다.
  2. 월드: 4×4m 방 → 8×8m 미로 (중앙 섬 방 + 링 복도 + 4분할 방 + 기둥)
     링 복도를 한 바퀴 돌면 SLAM 루프 클로저(loop closure)를 관찰할 수 있습니다.
  3. 로봇 스폰: 남서쪽 방 (-2.7, -2.7)에서 시작

같은 폴더의 ROS2 패키지 pinky_slam_nav와 함께 사용합니다 (빌드는 README 참고):

  [흐름 1 — 2단계: 지도 작성 → 저장 → 자율주행]
    T1(sim):   source /opt/ros/jazzy/setup.bash
               source env_isaaclab/bin/activate
               python 34_1_slam_nav.py
    T2(slam):  source /opt/ros/jazzy/setup.bash && source install/setup.bash
               ros2 launch pinky_slam_nav slam.launch.py
    T3(주행):  ros2 run teleop_twist_keyboard teleop_twist_keyboard \
                 --ros-args -p use_sim_time:=true
               → 링 복도를 한 바퀴 돌아 지도 완성 (루프 클로저 확인)
    T4(저장):  ros2 run nav2_map_server map_saver_cli \
                 -f <절대경로>/pinky_slam_nav/maps/maze --ros-args -p use_sim_time:=true
    T2 종료 후:
               ros2 launch pinky_slam_nav nav.launch.py \
                 map:=<절대경로>/pinky_slam_nav/maps/maze.yaml
               → RViz에서 2D Pose Estimate로 초기 위치 지정 → Nav2 Goal 클릭

  [흐름 2 — 동시: SLAM하며 바로 자율주행]
    T1(sim):   위와 동일
    T2:        ros2 launch pinky_slam_nav slam_nav.launch.py
               → 지도 없이 바로 RViz에서 Nav2 Goal 클릭

실행:  ※ ROS2 sourcing이 IsaacSim 실행보다 먼저여야 합니다 (34번과 동일).
      ※ GUI 권장. headless로 돌릴 때는 RTX LiDAR 렌더링을 위해
        --headless --enable_cameras 두 플래그가 모두 필요합니다.
"""

# ── 1. AppLauncher (다른 모든 import보다 먼저) ────────────────────────────
import argparse
import os
import sys

# ── 0. 환경 사전 점검 ────────────────────────────────────────────────────
# IsaacSim ROS2 브릿지의 jazzy 번들 lib에는 libament_index_cpp.so가 빠져 있어
# /opt/ros/jazzy/setup.bash를 먼저 source하지 않으면 브릿지 startup이 실패합니다.
_ros_distro = os.environ.get("ROS_DISTRO", "")
if _ros_distro != "jazzy":
    sep = "=" * 70
    msg = (
        f"\n{sep}\n"
        f"[ERROR] ROS_DISTRO 환경변수가 'jazzy'가 아닙니다 (현재: '{_ros_distro}').\n"
        f"        실행 전에 다음 순서로 환경을 준비하세요:\n"
        f"            source /opt/ros/jazzy/setup.bash\n"
        f"            source env_isaaclab/bin/activate    # 그 다음에 venv\n"
        f"            python lectures/34_ros2_bridge/34_1_slam_nav.py\n"
        f"{sep}\n"
    )
    print(msg, file=sys.stderr)
    sys.exit(2)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="34_1 - Pinky Pro SLAM + Nav2 환경")
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
    help="호스트에서 robot_state_publisher를 띄울 때 사용 (34번과 동일 동작).",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. ROS2 브릿지 활성화 ────────────────────────────────────────────────
from isaacsim.core.utils.extensions import enable_extension

if not enable_extension("isaacsim.ros2.bridge"):
    print("[ERROR] isaacsim.ros2.bridge 익스텐션을 활성화할 수 없습니다.", file=sys.stderr)
    simulation_app.close()
    sys.exit(1)
# DifferentialController OmniGraph 노드는 이 익스텐션 소속 —
# GUI에서는 기본 로드되지만 headless에서는 명시적으로 켜야 합니다.
enable_extension("isaacsim.robot.wheeled_robots")
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
from isaacsim.core.prims import SingleXFormPrim
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
LIDAR_FRAME = "rplidar_link"

# 로봇 스폰 위치: 남서쪽 방 (미로 스펙 참고)
SPAWN_POS = (-2.7, -2.7, 0.0)

# ── 5. 월드 + 로봇 + 지면 ────────────────────────────────────────────────
world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
world.scene.add_default_ground_plane()

add_reference_to_stage(USD_PATH, ROBOT_PRIM)
simulation_app.update()

# 스폰 포즈 지정 — world.reset() 전에 wrapper prim을 옮겨 놓으면
# IsaacComputeOdometry가 이 위치를 odom 원점(0,0)으로 삼습니다.
# 즉 SLAM 지도의 원점 = 이 스폰 위치가 됩니다.
SingleXFormPrim(ROBOT_PRIM).set_world_pose(
    np.array(SPAWN_POS, dtype=np.float32), np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
)
simulation_app.update()

# ── 5b. Articulation root / 링크 prim 동적 탐색 (34번과 동일) ────────────
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

# ── 5b-2. 휠 조인트에 velocity drive 명시 설정 (34번과 동일) ─────────────
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

# ── 5c. 8×8m SLAM 미로 배치 ─────────────────────────────────────────────
# 구조 (위에서 본 모습):
#   - 외벽 8×8m
#   - 중앙 섬 방 2.8×2.8m (북쪽에 1.2m 문)
#   - 섬 주위 링 복도 (폭 약 2.6m) — 한 바퀴 돌면 SLAM 루프 클로저 발생!
#   - 링 복도를 4개 스포크 벽으로 4분할 (각 스포크에 1.0m 문)
#   - NE/NW 방에 0.5m 기둥 — 스캔 매칭용 특징(feature)
#
#        N (+y)
#   ┌───────┬───────┐
#   │  NW ■ │ ■ NE  │     ■=기둥, ─/│=벽, 문은 스포크 중간과
#   ├─┐   ┌─┴─┐   ┌─┤     섬 북쪽에 있음
#   │ │   │섬 │   │ │
#   │ │   └───┘   │ │
#   ├─┘           └─┤
#   │  SW      SE   │
#   └───────┴───────┘
#   (SW 방이 로봇 스폰 위치)
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
ARENA = 8.0       # 외벽 한 변 길이 [m]

# (경로 이름, cx, cy, sx, sy) — z와 높이는 공통
MAZE_WALLS = [
    # 외벽 4면
    ("outer_n",  0.0,  4.0, ARENA + 2 * WALL_T, WALL_T),
    ("outer_s",  0.0, -4.0, ARENA + 2 * WALL_T, WALL_T),
    ("outer_e",  4.0,  0.0, WALL_T, ARENA + 2 * WALL_T),
    ("outer_w", -4.0,  0.0, WALL_T, ARENA + 2 * WALL_T),
    # 중앙 섬 방 (북쪽 벽은 짧게 만들어 x∈[0.2, 1.4] 구간이 1.2m 문)
    ("island_s",  0.0, -1.4, 2.8, WALL_T),
    ("island_w", -1.4,  0.0, WALL_T, 2.8),
    ("island_e",  1.4,  0.0, WALL_T, 2.8),
    ("island_n", -0.6,  1.4, 1.6, WALL_T),
    # 링 복도 4분할 스포크 (외벽~중간까지만 뻗어 각 1.0m 문이 생김)
    ("spoke_w", -3.2,  0.0, 1.6, WALL_T),
    ("spoke_s",  0.0, -3.2, WALL_T, 1.6),
    ("spoke_e",  3.2,  0.0, 1.6, WALL_T),
    ("spoke_n",  0.0,  3.2, WALL_T, 1.6),
    # 스캔 특징용 기둥
    ("pillar_ne",  2.6,  2.6, 0.5, 0.5),
    ("pillar_nw", -2.6,  2.6, 0.5, 0.5),
]

for name, cx, cy, sx, sy in MAZE_WALLS:
    add_wall(f"/World/maze/{name}", (cx, cy, WALL_H / 2), (sx, sy, WALL_H))

print(f"[INFO] {ARENA}m × {ARENA}m SLAM 미로 배치 완료 "
      f"(벽 {len(MAZE_WALLS)}개, 최소 통로 폭 1.0m)")
simulation_app.update()

# ── 6. RTX 2D LiDAR 부착 (rplidar_link 아래) ────────────────────────────
# 34번의 Example_Rotary_2D 대신 RPLIDAR_S2E 사용:
#   Example_Rotary_2D — nearRange 1.0m, 빔 -2° 기울어짐 → 좁은 미로에서 SLAM 불가
#   RPLIDAR_S2E       — 0.05~30m, 10Hz, 수평 빔 (실제 Pinky의 RPLIDAR 계열)
_, lidar_sensor = omni.kit.commands.execute(
    "IsaacSensorCreateRtxLidar",
    path=LIDAR_PRIM,
    parent=None,
    config="RPLIDAR_S2E",
    translation=(0.0, 0.0, 0.0),          # rplidar_link 원점 기준
    orientation=Gf.Quatd(1.0, 0.0, 0.0, 0.0),  # (w, x, y, z)
)

hydra_texture = rep.create.render_product(lidar_sensor.GetPath(), [1, 1], name="PinkyLidar")

# /scan 퍼블리셔 (LaserScan)
scan_writer = rep.writers.get("RtxLidar" + "ROS2PublishLaserScan")
scan_writer.initialize(topicName=args_cli.scan_topic, frameId=LIDAR_FRAME)
scan_writer.attach([hydra_texture])

# (참고) 디버그 시각화 — 뷰포트에 포인트 클라우드 점이 보임
debug_writer = rep.writers.get("RtxLidar" + "DebugDrawPointCloud")
debug_writer.attach([hydra_texture])

simulation_app.update()

# ── 7. ROS2 OmniGraph 구축 (34번과 동일) ─────────────────────────────────
GRAPH_PATH = "/ActionGraph"

_use_static_tf = not args_cli.with_rsp

_create_nodes = [
    ("OnTick", "omni.graph.action.OnPlaybackTick"),
    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
    ("Ros2Context", "isaacsim.ros2.bridge.ROS2Context"),
    ("PubClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
    ("PubJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
    ("PubTfOdomFp", "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
    ("PubOdom", "isaacsim.ros2.bridge.ROS2PublishOdometry"),
    ("ComputeOdom", "isaacsim.core.nodes.IsaacComputeOdometry"),
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
    ("OnTick.outputs:tick", "PubClock.inputs:execIn"),
    ("Ros2Context.outputs:context", "PubClock.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PubClock.inputs:timeStamp"),
    ("OnTick.outputs:tick", "PubJointState.inputs:execIn"),
    ("Ros2Context.outputs:context", "PubJointState.inputs:context"),
    ("ReadSimTime.outputs:simulationTime", "PubJointState.inputs:timeStamp"),
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
        ("OnTick.outputs:tick", "PubTfFpBase.inputs:execIn"),
        ("Ros2Context.outputs:context", "PubTfFpBase.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PubTfFpBase.inputs:timeStamp"),
        ("OnTick.outputs:tick", "PubTfBaseLidar.inputs:execIn"),
        ("Ros2Context.outputs:context", "PubTfBaseLidar.inputs:context"),
        ("ReadSimTime.outputs:simulationTime", "PubTfBaseLidar.inputs:timeStamp"),
    ]

_set_values = [
    ("PubClock.inputs:topicName", "clock"),
    ("PubClock.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubJointState.inputs:topicName", "joint_states"),
    ("PubJointState.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubJointState.inputs:targetPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    ("PubTfOdomFp.inputs:topicName", "tf"),
    ("PubTfOdomFp.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubTfOdomFp.inputs:parentFrameId", "odom"),
    ("PubTfOdomFp.inputs:childFrameId", "base_footprint"),
    ("PubTfOdomFp.inputs:staticPublisher", False),
    ("PubOdom.inputs:topicName", "odom"),
    ("PubOdom.inputs:nodeNamespace", args_cli.robot_namespace),
    ("PubOdom.inputs:odomFrameId", "odom"),
    ("PubOdom.inputs:chassisFrameId", "base_footprint"),
    ("ComputeOdom.inputs:chassisPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    ("SubTwist.inputs:topicName", args_cli.cmd_topic),
    ("SubTwist.inputs:nodeNamespace", args_cli.robot_namespace),
    ("DiffDrive.inputs:wheelRadius", WHEEL_RADIUS),
    ("DiffDrive.inputs:wheelDistance", WHEEL_DISTANCE),
    ("DiffDrive.inputs:maxLinearSpeed", 0.5),
    ("DiffDrive.inputs:maxAngularSpeed", 2.0),
    ("DiffDrive.inputs:maxWheelSpeed", 20.0),
    ("ArticController.inputs:targetPrim", [usdrt.Sdf.Path(ART_ROOT)]),
    ("ArticController.inputs:jointNames", [LEFT_WHEEL_JOINT, RIGHT_WHEEL_JOINT]),
]
if _use_static_tf:
    _set_values += [
        ("PubTfFpBase.inputs:topicName", "tf_static"),
        ("PubTfFpBase.inputs:nodeNamespace", args_cli.robot_namespace),
        ("PubTfFpBase.inputs:parentFrameId", "base_footprint"),
        ("PubTfFpBase.inputs:childFrameId", "base_link"),
        ("PubTfFpBase.inputs:translation", [0.0, 0.0, 0.028]),
        ("PubTfFpBase.inputs:rotation", [0.0, 0.0, 0.0, 1.0]),  # (x,y,z,w) identity
        ("PubTfFpBase.inputs:staticPublisher", True),
        ("PubTfBaseLidar.inputs:topicName", "tf_static"),
        ("PubTfBaseLidar.inputs:nodeNamespace", args_cli.robot_namespace),
        ("PubTfBaseLidar.inputs:parentFrameId", "base_link"),
        ("PubTfBaseLidar.inputs:childFrameId", LIDAR_FRAME),
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
print("[INFO] Pinky Pro SLAM + Nav2 환경 준비 완료")
print(f"  로봇 스폰:        {SPAWN_POS[:2]} (남서쪽 방) — SLAM 지도 원점")
print(f"  Articulation 루트: {ART_ROOT}")
print(f"  LiDAR:            RPLIDAR_S2E @ {LIDAR_PRIM}")
print(f"  미로:             {ARENA}m × {ARENA}m (링 복도 + 5개 방)")
print()
static_tf_str = "/tf_static " if _use_static_tf else ""
print("  게시 토픽:  /clock /tf {}/joint_states /odom /{}".format(static_tf_str, args_cli.scan_topic))
print("  구독 토픽:  /{}".format(args_cli.cmd_topic))
print()
print("  다음 단계 (별도 셸, pinky_slam_nav 패키지 빌드 후):")
print("    [2단계 흐름] ros2 launch pinky_slam_nav slam.launch.py")
print("                 + teleop으로 링 복도 일주 → map_saver_cli → nav.launch.py")
print("    [동시 흐름]  ros2 launch pinky_slam_nav slam_nav.launch.py")
print("  자세한 순서는 34_ros2_bridge/README.md의 '34-1' 섹션 참고")
print("=" * 70)

world.reset()
# play()를 명시적으로 호출해야 timeline이 시작되고, RTX LiDAR 렌더 파이프라인과
# OmniGraph tick이 실제로 발화합니다 (34번과 동일한 함정 주의).
world.play()

# ── 9. 메인 루프 ────────────────────────────────────────────────────────
try:
    while simulation_app.is_running():
        world.step(render=True)
except KeyboardInterrupt:
    print("\n[INFO] Ctrl-C: 시뮬레이션 종료")
finally:
    simulation_app.close()
    print("[DONE] 34-1강 종료")
