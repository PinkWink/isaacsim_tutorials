"""
19_obstacle_avoidance.py - JetBot 장애물 회피 자율 주행

JetBot이 장애물을 감지하고 회피하면서 목표 지점으로 이동합니다:
  - 장애물을 kinematic RigidObject(박스)로 스폰
  - 로봇-장애물 간 거리를 계산하여 회피 판정
  - 장애물 없으면: 목표를 향해 P-제어로 전진 (17번과 동일)
  - 장애물 감지 시: 반대 방향으로 회전하며 감속 (회피 모드)
  - 유니사이클 → 차동 구동 변환 후 바퀴에 속도 적용
  - 목표 도달 시 자동 리셋

실행:
    source env_isaaclab/bin/activate
    cd lectures/19_obstacle_avoidance
    python 19_obstacle_avoidance.py

    # GUI 없이 실행
    python 19_obstacle_avoidance.py --headless
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="19 - JetBot 장애물 회피")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat

# ── 3. JetBot 설정 ──────────────────────────────────────────────────────

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

WHEEL_RADIUS = 0.03
WHEEL_BASE = 0.1125

# ── 4. 장애물/목표 설정 ──────────────────────────────────────────────────

# 장애물 위치 (환경 로컬 좌표, JetBot 스케일에 맞춤)
OBSTACLE_POSITIONS = [
    (0.4, 0.1),
    (0.8, -0.15),
    (1.2, 0.2),
    (1.6, -0.1),
]

GOAL_POSITION = (2.0, 0.0)
OBSTACLE_SIZE = (0.1, 0.1, 0.15)

# 회피 파라미터
AVOIDANCE_THRESHOLD = 0.3     # 장애물 감지 거리 (m)
GOAL_TOLERANCE = 0.08         # 목표 도달 거리 (m)
MAX_LINEAR_VEL = 0.25         # 최대 전진 속도 (m/s)
SLOW_LINEAR_VEL = 0.08        # 회피 시 감속 전진 (m/s)
MAX_ANGULAR_VEL = 3.0         # 최대 회전 속도 (rad/s)
Kp_HEADING = 4.0              # 방향 P-제어 게인
Kp_AVOID = 5.0                # 회피 회전 게인


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg
# ══════════════════════════════════════════════════════════════════════════


def _make_obstacle_cfg(idx: int, color: tuple) -> RigidObjectCfg:
    """장애물 RigidObjectCfg 생성 헬퍼."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Obstacle_{idx+1}",
        spawn=sim_utils.CuboidCfg(
            size=OBSTACLE_SIZE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(OBSTACLE_POSITIONS[idx][0], OBSTACLE_POSITIONS[idx][1], OBSTACLE_SIZE[2] / 2)
        ),
    )


@configclass
class ObstacleSceneCfg(InteractiveSceneCfg):
    """장애물 회피 씬 (JetBot 1대 + 장애물 4개 + 목표)."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    obstacle_1: RigidObjectCfg = _make_obstacle_cfg(0, (0.9, 0.2, 0.2))
    obstacle_2: RigidObjectCfg = _make_obstacle_cfg(1, (0.9, 0.5, 0.1))
    obstacle_3: RigidObjectCfg = _make_obstacle_cfg(2, (0.9, 0.9, 0.1))
    obstacle_4: RigidObjectCfg = _make_obstacle_cfg(3, (0.2, 0.4, 0.9))


# ══════════════════════════════════════════════════════════════════════════
# 차동 구동 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v, omega):
    """유니사이클 (v, ω) → 좌/우 바퀴 각속도."""
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


def normalize_angle(angle):
    """각도를 [-π, π]로 정규화."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


# ══════════════════════════════════════════════════════════════════════════
# 시각화 헬퍼
# ══════════════════════════════════════════════════════════════════════════


def create_goal_marker(origin):
    """목표 마커 (녹색 구체) 생성."""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/GoalMarker",
        markers={
            "goal": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.1, 0.9, 0.1),
                    emissive_color=(0.0, 0.3, 0.0),
                ),
            ),
        },
    )
    marker = VisualizationMarkers(cfg)
    gx = origin[0].item() + GOAL_POSITION[0]
    gy = origin[1].item() + GOAL_POSITION[1]
    marker.visualize(translations=torch.tensor([[gx, gy, 0.05]]))
    return marker


def create_planned_path(origin):
    """계획 경로 시각화 (파란색 점선): 출발점 → 목표까지 직선."""
    num_dots = 40
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PlannedPath",
        markers={
            "dot": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.3, 0.5, 1.0),  # 파란색
                ),
            ),
        },
    )
    marker = VisualizationMarkers(cfg)

    ox, oy = origin[0].item(), origin[1].item()
    positions = torch.zeros(num_dots, 3)
    for i in range(num_dots):
        t = i / (num_dots - 1)
        positions[i] = torch.tensor([
            ox + GOAL_POSITION[0] * t,
            oy + GOAL_POSITION[1] * t,
            0.005,
        ])
    marker.visualize(translations=positions)
    return marker


class TrailVisualizer:
    """로봇이 실제로 지나간 경로를 빨간 점으로 시각화합니다."""

    MAX_POINTS = 500
    RECORD_INTERVAL = 5  # 5스텝마다 1개 기록

    def __init__(self, origin):
        self.origin = origin
        self.positions = torch.zeros(self.MAX_POINTS, 3)
        # 초기에는 모든 점을 바닥 아래에 숨김
        self.positions[:, 2] = -1.0
        self.count = 0
        self.step = 0

        cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/ActualTrail",
            markers={
                "dot": sim_utils.SphereCfg(
                    radius=0.008,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.2, 0.1),  # 빨간색
                    ),
                ),
            },
        )
        self.marker = VisualizationMarkers(cfg)
        self.marker.visualize(translations=self.positions)

    def update(self, robot_pos_w):
        """로봇 월드 위치를 기록하고 마커를 업데이트합니다."""
        self.step += 1
        if self.step % self.RECORD_INTERVAL != 0:
            return
        if self.count >= self.MAX_POINTS:
            return

        self.positions[self.count] = torch.tensor([
            robot_pos_w[0].item(),
            robot_pos_w[1].item(),
            0.005,
        ])
        self.count += 1
        self.marker.visualize(translations=self.positions)

    def reset(self):
        """경로 기록을 초기화합니다."""
        self.positions[:, 2] = -1.0
        self.count = 0
        self.step = 0
        self.marker.visualize(translations=self.positions)


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """장애물 회피 + 목표 추적 알고리즘으로 JetBot을 제어합니다."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # ── 관절 ID ──────────────────────────────────────────────────────────
    left_ids, _ = robot.find_joints("left_wheel_joint")
    right_ids, _ = robot.find_joints("right_wheel_joint")
    wheel_ids = left_ids + right_ids

    # ── 장애물/목표 텐서 ─────────────────────────────────────────────────
    obs_local = torch.tensor(OBSTACLE_POSITIONS, device=sim.device)  # (4, 2)
    goal_local = torch.tensor(GOAL_POSITION, device=sim.device)      # (2,)

    # ── 시각화 ───────────────────────────────────────────────────────────
    origin = scene.env_origins[0]
    goal_marker = create_goal_marker(origin)
    planned_path = create_planned_path(origin)      # 파란색: 계획 경로
    trail = TrailVisualizer(origin)                  # 빨간색: 실제 경로

    # ── 상태 변수 ────────────────────────────────────────────────────────
    step_count = 0
    reached_goal = False

    print(f"[INFO] dt={sim_dt:.4f}s, 장애물 {len(OBSTACLE_POSITIONS)}개")
    print(f"[INFO] 목표: {GOAL_POSITION}, 회피 거리: {AVOIDANCE_THRESHOLD}m")
    print(f"[INFO] 경로 시각화: 파란색=계획경로, 빨간색=실제경로")
    print("=" * 70)

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 리셋 ────────────────────────────────────────────────────
        if step_count == 0 or (reached_goal and step_count % 200 == 0):
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            reached_goal = False
            trail.reset()
            print(f"\n[Step {step_count:5d}] 리셋 — 목표를 향해 출발!")

        # ── (2) 로봇 상태 읽기 ───────────────────────────────────────────
        pos_w = robot.data.root_pos_w[0, :2]
        origin_xy = scene.env_origins[0, :2]
        pos_local = pos_w - origin_xy  # (2,)

        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        current_yaw = yaw[0]  # scalar

        # ── (3) 목표까지 거리/방향 ───────────────────────────────────────
        goal_diff = goal_local - pos_local
        goal_dist = torch.norm(goal_diff)
        goal_angle = torch.atan2(goal_diff[1], goal_diff[0])

        if goal_dist.item() < GOAL_TOLERANCE:
            reached_goal = True
            print(f"  *** 목표 도달! (거리: {goal_dist.item():.3f}m) ***")

        # ── (4) 가장 가까운 장애물 탐색 ──────────────────────────────────
        obs_diffs = obs_local - pos_local.unsqueeze(0)  # (4, 2)
        obs_dists = torch.norm(obs_diffs, dim=1)         # (4,)
        min_dist, min_idx = obs_dists.min(dim=0)

        # ── (5) 회피/추적 알고리즘 ───────────────────────────────────────
        if reached_goal:
            v_cmd, omega_cmd = 0.0, 0.0
            action = "DONE"

        elif min_dist.item() < AVOIDANCE_THRESHOLD:
            # ★ 회피 모드: 장애물 반대 방향으로 회전 + 감속
            closest_dir = obs_diffs[min_idx]
            avoid_angle = torch.atan2(-closest_dir[1], -closest_dir[0])
            angle_err = normalize_angle(avoid_angle - current_yaw)

            omega_cmd = torch.clamp(
                Kp_AVOID * angle_err, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL
            ).item()

            speed_ratio = max(min_dist.item() / AVOIDANCE_THRESHOLD, 0.1)
            v_cmd = SLOW_LINEAR_VEL * speed_ratio
            action = f"AVOID (d={min_dist.item():.2f}m)"

        else:
            # ★ 목표 추적 모드: P-제어
            heading_err = normalize_angle(goal_angle - current_yaw)

            omega_cmd = torch.clamp(
                Kp_HEADING * heading_err, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL
            ).item()

            alignment = max(torch.cos(heading_err).item(), 0.1)
            v_cmd = MAX_LINEAR_VEL * alignment
            action = f"NAV (d_goal={goal_dist.item():.2f}m)"

        # ── (6) 유니사이클 → 바퀴 속도 적용 ─────────────────────────────
        wl, wr = unicycle_to_wheel_vel(v_cmd, omega_cmd)
        wheel_vel = torch.zeros(robot.num_instances, 2, device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        # ── (7) 시뮬레이션 스텝 ──────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # ── (7.5) 실제 경로 기록 (빨간 점) ──────────────────────────────
        trail.update(robot.data.root_pos_w[0])

        # ── (8) 상태 출력 (50 스텝마다) ──────────────────────────────────
        if step_count % 50 == 0:
            print(
                f"[Step {step_count:5d}] "
                f"pos=({pos_local[0]:.2f}, {pos_local[1]:.2f}) "
                f"yaw={math.degrees(current_yaw.item()):+.0f}° | "
                f"obs={min_dist.item():.2f}m | {action}"
            )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.0, -1.5, 2.0], target=[1.0, 0.0, 0.0])

    scene_cfg = ObstacleSceneCfg(num_envs=1, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
