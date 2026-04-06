"""
17_mobile_navigation.py - JetBot P-제어기 기반 Waypoint 내비게이션

JetBot 1대를 P-제어기로 제어하여 2m×2m 사각형 Waypoint 경로를
자율적으로 따라가는 내비게이션을 구현합니다:
  - GUI에 Waypoint 마커(구체)와 경로 라인(실린더)을 시각화
  - atan2로 목표 방향 계산, heading error 기반 P-제어
  - 유니사이클 → 차동 구동 변환
  - Waypoint 도달 시 다음 Waypoint로 자동 전환 (순환)

실행:
    source env_isaaclab/bin/activate
    cd lectures/17_mobile_navigation
    python 17_mobile_navigation.py
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="17 - JetBot Waypoint 내비게이션")
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

# ── 4. Waypoint 경로 정의 (2m × 2m 사각형) ──────────────────────────────
WAYPOINTS = [
    [2.0, 0.0],    # WP 0: 전방 2m
    [2.0, 2.0],    # WP 1: 대각선
    [0.0, 2.0],    # WP 2: 좌측
    [0.0, 0.0],    # WP 3: 원점 복귀
]


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg (로봇 1대)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class NavSceneCfg(InteractiveSceneCfg):
    """내비게이션 씬 (환경 1개)."""

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


# ══════════════════════════════════════════════════════════════════════════
# 경로 시각화
# ══════════════════════════════════════════════════════════════════════════


def create_path_visualization(waypoints_list: list, origin: torch.Tensor):
    """Waypoint 마커(구체)와 경로 라인(실린더)을 Isaac Sim GUI에 생성합니다.

    Args:
        waypoints_list: [[x,y], ...] 형태의 Waypoint 좌표 리스트
        origin: 환경 원점 (3,) 텐서
    """
    num_wp = len(waypoints_list)
    ox, oy, oz = origin[0].item(), origin[1].item(), origin[2].item()

    # ── Waypoint 마커 (녹색 구체) ────────────────────────────────────────
    wp_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/WaypointMarkers",
        markers={
            "wp": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.2, 0.9, 0.2),  # 녹색
                ),
            ),
        },
    )
    wp_markers = VisualizationMarkers(wp_marker_cfg)

    # 각 WP 위치에 구체 배치
    wp_positions = torch.zeros(num_wp, 3)
    for i, (wx, wy) in enumerate(waypoints_list):
        wp_positions[i] = torch.tensor([ox + wx, oy + wy, 0.05])

    wp_markers.visualize(translations=wp_positions)

    # ── 경로 라인 (주황색 실린더) ────────────────────────────────────────
    line_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/PathLines",
        markers={
            "line": sim_utils.CylinderCfg(
                radius=0.008,
                height=1.0,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.5, 0.0),  # 주황색
                ),
            ),
        },
    )
    line_markers = VisualizationMarkers(line_marker_cfg)

    # 각 WP 쌍 사이에 실린더를 배치 (위치 = 중점, 회전 = 방향, 높이 = 거리)
    line_positions = torch.zeros(num_wp, 3)
    line_orientations = torch.zeros(num_wp, 4)
    line_scales = torch.zeros(num_wp, 3)

    for i in range(num_wp):
        j = (i + 1) % num_wp
        x1, y1 = waypoints_list[i]
        x2, y2 = waypoints_list[j]

        # 중점
        mx = ox + (x1 + x2) / 2.0
        my = oy + (y1 + y2) / 2.0
        line_positions[i] = torch.tensor([mx, my, 0.005])

        # 길이
        dx = x2 - x1
        dy = y2 - y1
        length = math.sqrt(dx**2 + dy**2)
        line_scales[i] = torch.tensor([1.0, 1.0, length])

        # 방향: 실린더는 기본적으로 Z축 방향이므로 XY 평면으로 눕혀야 함
        # Z축 → XY 방향으로 회전
        angle = math.atan2(dy, dx)
        # 쿼터니언: Y축 90도 회전 + Z축 angle 회전
        # 먼저 Y축으로 90도 눕힌 후, Z축으로 angle 회전
        # q = qz(angle) * qy(pi/2)
        cy, sy = math.cos(angle / 2), math.sin(angle / 2)
        qz = torch.tensor([cy, 0.0, 0.0, sy])  # Z축 회전
        qy = torch.tensor([math.cos(math.pi / 4), 0.0, math.sin(math.pi / 4), 0.0])  # Y축 90도

        # 쿼터니언 곱: q = qz * qy
        w1, x1q, y1q, z1q = qz
        w2, x2q, y2q, z2q = qy
        line_orientations[i] = torch.tensor([
            w1*w2 - x1q*x2q - y1q*y2q - z1q*z2q,
            w1*x2q + x1q*w2 + y1q*z2q - z1q*y2q,
            w1*y2q - x1q*z2q + y1q*w2 + z1q*x2q,
            w1*z2q + x1q*y2q - y1q*x2q + z1q*w2,
        ])

    line_markers.visualize(
        translations=line_positions,
        orientations=line_orientations,
        scales=line_scales,
    )

    # ── 현재 목표 마커 (빨간 구체, 매 스텝 업데이트용) ────────────────────
    target_marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/TargetMarker",
        markers={
            "target": sim_utils.SphereCfg(
                radius=0.08,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.1, 0.1),  # 빨간색
                    emissive_color=(0.5, 0.0, 0.0),
                ),
            ),
        },
    )
    target_marker = VisualizationMarkers(target_marker_cfg)

    return wp_markers, line_markers, target_marker


# ══════════════════════════════════════════════════════════════════════════
# 차동 구동 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v, omega):
    """유니사이클 (v, ω) → 좌/우 바퀴 각속도 (rad/s). 텐서 지원."""
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """P-제어기로 Waypoint를 따라가는 내비게이션. GUI에 경로 시각화."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # ── 관절 ID 조회 ─────────────────────────────────────────────────────
    left_ids, _ = robot.find_joints("left_wheel_joint")
    right_ids, _ = robot.find_joints("right_wheel_joint")
    wheel_ids = left_ids + right_ids

    # ── Waypoint 텐서 ────────────────────────────────────────────────────
    waypoints = torch.tensor(WAYPOINTS, device=sim.device)
    num_wp = waypoints.shape[0]

    # ── 경로 시각화 생성 ─────────────────────────────────────────────────
    origin = scene.env_origins[0]
    wp_markers, line_markers, target_marker = create_path_visualization(
        WAYPOINTS, origin
    )
    print("[INFO] 경로 시각화 완료 (녹색=WP, 주황=경로, 빨강=현재 목표)")

    # ── P-제어기 게인 ────────────────────────────────────────────────────
    Kp_linear = 1.5
    Kp_angular = 4.0
    max_linear_vel = 0.3
    max_angular_vel = 3.0
    arrival_threshold = 0.08

    # ── 상태 변수 ────────────────────────────────────────────────────────
    current_wp = torch.zeros(1, dtype=torch.long, device=sim.device)
    lap_count = 0
    step_count = 0

    print(f"[INFO] dt={sim_dt:.4f}s")
    print(f"[INFO] Waypoints (2m×2m): {WAYPOINTS}")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 초기 리셋 ────────────────────────────────────────────────
        if step_count == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            current_wp[:] = 0
            lap_count = 0
            print("[INFO] 리셋 완료 — 내비게이션 시작")

        # ── (2) 로봇 상태 읽기 ───────────────────────────────────────────
        pos_w = robot.data.root_pos_w[:, :2]
        origin_xy = scene.env_origins[:, :2]
        pos_local = pos_w - origin_xy

        roll, pitch, yaw = euler_xyz_from_quat(robot.data.root_quat_w)
        current_yaw = yaw

        # ── (3) P-제어기 ─────────────────────────────────────────────────
        target = waypoints[current_wp]

        dx = target[:, 0] - pos_local[:, 0]
        dy = target[:, 1] - pos_local[:, 1]
        distance = torch.sqrt(dx**2 + dy**2)

        desired_yaw = torch.atan2(dy, dx)
        heading_err = torch.atan2(
            torch.sin(desired_yaw - current_yaw),
            torch.cos(desired_yaw - current_yaw),
        )

        v = torch.clamp(Kp_linear * distance, max=max_linear_vel)
        omega = torch.clamp(Kp_angular * heading_err, -max_angular_vel, max_angular_vel)
        v = v * torch.clamp(1.0 - torch.abs(heading_err) / math.pi, min=0.1)

        # ── (4) 바퀴 속도 적용 ───────────────────────────────────────────
        wl, wr = unicycle_to_wheel_vel(v, omega)
        wheel_vel = torch.zeros(robot.num_instances, 2, device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        # ── (5) 시뮬레이션 스텝 ──────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # ── (6) 현재 목표 마커 업데이트 (빨간 구체 이동) ─────────────────
        wp_idx = current_wp[0].item()
        tx = origin[0].item() + waypoints[wp_idx, 0].item()
        ty = origin[1].item() + waypoints[wp_idx, 1].item()
        target_marker.visualize(
            translations=torch.tensor([[tx, ty, 0.08]]),
        )

        # ── (7) Waypoint 도달 판정 ───────────────────────────────────────
        arrived = distance < arrival_threshold
        if arrived.any():
            old_wp = current_wp[0].item()
            current_wp[0] = (current_wp[0] + 1) % num_wp
            new_wp = current_wp[0].item()

            if new_wp == 0:
                lap_count += 1
                print(f"  WP {old_wp} 도달! → 1바퀴 완료 (총 {lap_count}바퀴)")
            else:
                print(f"  WP {old_wp} 도달! → WP {new_wp} ({WAYPOINTS[new_wp]})")

        # ── (8) 상태 출력 (50 스텝마다) ──────────────────────────────────
        if step_count % 50 == 0:
            print(
                f"[Step {step_count:5d}] "
                f"pos=({pos_local[0,0]:.2f}, {pos_local[0,1]:.2f}) | "
                f"yaw={math.degrees(current_yaw[0].item()):+.0f}° | "
                f"WP={wp_idx} dist={distance[0]:.3f}m | "
                f"v={v[0]:.2f} ω={omega[0]:.2f}"
            )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # 위에서 내려다보는 카메라 (2m×2m 경로 전체가 보이도록)
    sim.set_camera_view(eye=[1.0, 1.0, 5.0], target=[1.0, 1.0, 0.0])

    # 로봇 1대만 생성
    scene_cfg = NavSceneCfg(num_envs=1, env_spacing=6.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - 로봇 1대")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
