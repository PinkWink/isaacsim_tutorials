"""
33_pinky_sensors.py - Pinky Pro 센서 추가 (LiDAR + 카메라)

변환된 Pinky Pro USD에 Isaac Sim 센서를 부착합니다:
  - RayCaster (LiDAR): rplidar_link에 2D LiDAR 센서 부착
  - Camera: front_camera_link에 RGB 카메라 부착
  - 센서 데이터를 읽어 콘솔에 출력
  - 차동 구동으로 이동하며 센서 데이터 변화 관찰
  - 장애물을 배치하여 LiDAR 감지 테스트

실행:
    source env_isaaclab/bin/activate
    python lectures/33_pinky_sensors/33_pinky_sensors.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="33 - Pinky Pro 센서 (LiDAR + Camera)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # 카메라 센서 사용을 위해 필수

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, CameraCfg
from isaaclab.sensors.ray_caster.patterns import LidarPatternCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

# ── 3. 설정 ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "31_urdf_to_usd", "usd_output", "pinky_pro.usd"
))

assert os.path.exists(USD_PATH), f"USD 없음: {USD_PATH}\n먼저 31강을 실행하세요."

WHEEL_RADIUS = 0.028
WHEEL_BASE = 0.0811

PINKY_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=USD_PATH),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.05)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)


# ══════════════════════════════════════════════════════════════════════════
# Scene (로봇 + 센서 + 장애물)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class PinkySensorSceneCfg(InteractiveSceneCfg):
    """Pinky Pro + LiDAR + Camera + 장애물."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.8, 0.8, 0.85)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 1.0, 0.95)),
    )

    # ── Pinky Pro 로봇 ──────────────────────────────────────────────────
    robot: ArticulationCfg = PINKY_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── LiDAR 센서 (RayCaster) ──────────────────────────────────────────
    # merge_fixed_joints=True에 의해 base_link가 base_footprint에 병합됨
    # → 실제 존재하는 prim은 base_footprint
    # offset으로 rplidar_link 위치를 지정 (base_footprint 기준)
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_footprint",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        pattern_cfg=LidarPatternCfg(
            channels=1,                           # 2D LiDAR (1채널)
            vertical_fov_range=(0.0, 0.0),         # 수평면만
            horizontal_fov_range=(-180.0, 180.0),   # 360도
            horizontal_res=2.0,                     # 2도 간격 → 180개 포인트
        ),
        offset=RayCasterCfg.OffsetCfg(
            pos=(-0.017, 0.0, 0.125),             # rplidar 위치: base_footprint 기준
        ),
        max_distance=8.0,
        debug_vis=False,                           # 히트 없을 때 에러 방지
    )

    # ── 카메라 센서 ──────────────────────────────────────────────────────
    # merge_fixed_joints로 front_camera_link가 사라졌으므로
    # 별도 prim 경로에 카메라를 spawn하여 생성
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FrontCamera",
        offset=CameraCfg.OffsetCfg(
            pos=(0.035, 0.0, 0.065),              # front_camera_link 위치 (base 기준)
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=2.0,
            horizontal_aperture=3.6,
            clipping_range=(0.01, 5.0),
        ),
        width=320,
        height=240,
        data_types=["rgb"],
        update_period=0.1,
    )

    # ── 장애물 (LiDAR 테스트용) ──────────────────────────────────────────
    obstacle_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_1",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.1, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.075)),
    )
    obstacle_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Obstacle_2",
        spawn=sim_utils.CuboidCfg(
            size=(0.1, 0.3, 0.15),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.3, 0.075)),
    )


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v, omega):
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    lidar = scene["lidar"]
    camera = scene["camera"]
    sim_dt = sim.get_physics_dt()

    # 관절 탐색
    left_ids, _ = robot.find_joints("l_wheel.*")
    right_ids, _ = robot.find_joints("r_wheel.*")
    wheel_ids = left_ids + right_ids

    # 모션 패턴
    PATTERNS = [
        ("전진", 0.08, 0.0, 300),
        ("좌회전", 0.0, 2.0, 200),
        ("전진", 0.08, 0.0, 300),
        ("우회전", 0.0, -2.0, 200),
        ("정지", 0.0, 0.0, 200),
    ]
    total_steps = sum(p[3] for p in PATTERNS)
    ep_step = 0

    print("=" * 70)
    print("[INFO] Pinky Pro + 센서 시뮬레이션 시작")
    print(f"  LiDAR: {lidar.num_rays} rays, max_dist={8.0}m")
    print(f"  Camera: 320x240 RGB")
    print("=" * 70)

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # 리셋
        if ep_step % (total_steps + 200) == 0:
            ep_step = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] += 0.05
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            robot.write_joint_state_to_sim(
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            scene.reset()
            print("\n[리셋]")

        # 패턴 결정
        v, omega = 0.0, 0.0
        cumulative = 0
        for name, pv, pw, dur in PATTERNS:
            if cumulative <= ep_step < cumulative + dur:
                v, omega = pv, pw
                if ep_step == cumulative:
                    print(f"  [Phase] {name}")
                break
            cumulative += dur

        # 바퀴 제어
        wl, wr = unicycle_to_wheel_vel(v, omega)
        wheel_vel = torch.zeros(robot.num_instances, len(wheel_ids), device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ep_step += 1

        # 센서 데이터 출력 (100스텝마다)
        if ep_step % 100 == 0:
            pos = robot.data.root_pos_w[0]

            # LiDAR 데이터
            lidar_data = lidar.data.ray_hits_w  # (N, num_rays, 3)
            distances = torch.norm(
                lidar_data[0] - robot.data.root_pos_w[0].unsqueeze(0), dim=-1
            )
            valid = distances < 8.0
            min_dist = distances[valid].min().item() if valid.any() else 8.0
            num_hits = valid.sum().item()

            # 카메라 데이터
            cam_shape = "N/A"
            if camera.data.output.get("rgb") is not None:
                cam_shape = str(tuple(camera.data.output["rgb"].shape))

            print(
                f"[Step {ep_step:5d}] "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}) | "
                f"LiDAR: {num_hits} hits, min={min_dist:.2f}m | "
                f"Cam: {cam_shape}"
            )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 0.5], target=[0.0, 0.0, 0.05])

    scene_cfg = PinkySensorSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] 씬 생성 완료 (로봇 + LiDAR + Camera + 장애물)")

    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 33강 완료")
