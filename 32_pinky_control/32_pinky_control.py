"""
32_pinky_control.py - Pinky Pro 차동 구동 제어 및 경로 주행

31강에서 변환한 Pinky Pro USD를 사용하여 차동 구동 제어를 구현합니다:
  - l_wheel_joint, r_wheel_joint에 velocity target 적용
  - 유니사이클 모델 (v, ω) → 바퀴 각속도 변환
  - P-제어기 기반 Waypoint 내비게이션
  - 경로 시각화 (파란=계획, 빨간=실제)

실행:
    source env_isaaclab/bin/activate
    python lectures/32_pinky_control/32_pinky_control.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="32 - Pinky Pro 차동 구동 제어")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat

# ── 3. Pinky Pro USD 설정 ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
USD_PATH = os.path.abspath(os.path.join(
    SCRIPT_DIR, "..", "31_urdf_to_usd", "usd_output", "pinky_pro.usd"
))

assert os.path.exists(USD_PATH), (
    f"USD 파일 없음: {USD_PATH}\n먼저 31강을 실행하세요."
)

# Pinky Pro 물리 파라미터 (URDF에서 확인)
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

# Waypoint 경로 (1m × 1m 사각형)
WAYPOINTS = [
    [0.5, 0.0],
    [0.5, 0.5],
    [0.0, 0.5],
    [0.0, 0.0],
]


# ══════════════════════════════════════════════════════════════════════════
# Scene
# ══════════════════════════════════════════════════════════════════════════


@configclass
class PinkyControlSceneCfg(InteractiveSceneCfg):
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
    robot: ArticulationCfg = PINKY_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ══════════════════════════════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v, omega):
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


def create_path_markers(origin, waypoints):
    """Waypoint 마커(녹색) + 경로 라인(파란 점) 생성."""
    ox, oy = origin[0].item(), origin[1].item()

    # Waypoint 마커
    wp_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Waypoints",
        markers={"wp": sim_utils.SphereCfg(
            radius=0.02,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.9, 0.2)),
        )},
    )
    wp_markers = VisualizationMarkers(wp_cfg)
    positions = torch.tensor([[ox + w[0], oy + w[1], 0.01] for w in waypoints])
    wp_markers.visualize(translations=positions)

    # 경로 점선 (파란색)
    path_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PathDots",
        markers={"dot": sim_utils.SphereCfg(
            radius=0.005,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.5, 1.0)),
        )},
    )
    path_markers = VisualizationMarkers(path_cfg)

    dots = []
    for i in range(len(waypoints)):
        j = (i + 1) % len(waypoints)
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[j]
        for t_val in [k / 20.0 for k in range(20)]:
            dots.append([ox + x1 + (x2-x1)*t_val, oy + y1 + (y2-y1)*t_val, 0.003])
    path_markers.visualize(translations=torch.tensor(dots))

    return wp_markers, path_markers


class TrailVisualizer:
    """실제 경로 (빨간 점)."""
    MAX_POINTS = 500
    INTERVAL = 5

    def __init__(self):
        self.positions = torch.zeros(self.MAX_POINTS, 3)
        self.positions[:, 2] = -1.0
        self.count = 0
        self.step = 0
        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/Trail",
            markers={"dot": sim_utils.SphereCfg(
                radius=0.004,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.1)),
            )},
        )
        self.marker = VisualizationMarkers(cfg)
        self.marker.visualize(translations=self.positions)

    def update(self, pos_w):
        self.step += 1
        if self.step % self.INTERVAL != 0 or self.count >= self.MAX_POINTS:
            return
        self.positions[self.count] = torch.tensor([pos_w[0].item(), pos_w[1].item(), 0.003])
        self.count += 1
        self.marker.visualize(translations=self.positions)

    def reset(self):
        self.positions[:, 2] = -1.0
        self.count = 0
        self.step = 0
        self.marker.visualize(translations=self.positions)


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # 관절 탐색
    left_ids, _ = robot.find_joints("l_wheel.*")
    right_ids, _ = robot.find_joints("r_wheel.*")
    wheel_ids = left_ids + right_ids
    print(f"[INFO] Wheel IDs: L={left_ids}, R={right_ids}")

    # 시각화
    origin = scene.env_origins[0]
    create_path_markers(origin, WAYPOINTS)
    trail = TrailVisualizer()

    # Waypoint / P-제어
    waypoints = torch.tensor(WAYPOINTS, device=sim.device)
    num_wp = waypoints.shape[0]
    current_wp = torch.zeros(1, dtype=torch.long, device=sim.device)
    lap_count = 0
    step_count = 0

    Kp_lin, Kp_ang = 1.5, 6.0
    max_v, max_w = 0.15, 4.0
    arrival_th = 0.03

    print(f"[INFO] Waypoints: {WAYPOINTS}")
    print("=" * 70)

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # 리셋
        if step_count == 0:
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
            current_wp[:] = 0
            lap_count = 0
            trail.reset()
            print("[리셋] 내비게이션 시작")

        # 로봇 상태
        pos_w = robot.data.root_pos_w[0, :2]
        origin_xy = scene.env_origins[0, :2]
        pos_local = pos_w - origin_xy
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        current_yaw = yaw[0]

        # P-제어
        target = waypoints[current_wp[0]]
        dx = target[0] - pos_local[0]
        dy = target[1] - pos_local[1]
        dist = torch.sqrt(dx**2 + dy**2)
        desired_yaw = torch.atan2(dy, dx)
        h_err = torch.atan2(torch.sin(desired_yaw - current_yaw), torch.cos(desired_yaw - current_yaw))

        v = torch.clamp(Kp_lin * dist, max=max_v)
        omega = torch.clamp(Kp_ang * h_err, -max_w, max_w)
        v = v * torch.clamp(1.0 - torch.abs(h_err) / math.pi, min=0.1)

        # 바퀴 속도
        wl, wr = unicycle_to_wheel_vel(v.item(), omega.item())
        wheel_vel = torch.zeros(robot.num_instances, len(wheel_ids), device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        trail.update(robot.data.root_pos_w[0])

        # WP 도달 판정
        if dist.item() < arrival_th:
            old_wp = current_wp[0].item()
            current_wp[0] = (current_wp[0] + 1) % num_wp
            new_wp = current_wp[0].item()
            if new_wp == 0:
                lap_count += 1
                print(f"  WP {old_wp} → 1바퀴 완료! (총 {lap_count}바퀴)")
            else:
                print(f"  WP {old_wp} → WP {new_wp}")

        if step_count % 100 == 0:
            print(
                f"[Step {step_count:5d}] pos=({pos_local[0]:.3f}, {pos_local[1]:.3f}) "
                f"WP={current_wp[0].item()} dist={dist.item():.3f}m"
            )


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.25, 0.25, 0.8], target=[0.25, 0.25, 0.0])

    scene_cfg = PinkyControlSceneCfg(num_envs=1, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] 씬 생성 완료")

    sim.reset()
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 32강 완료")
