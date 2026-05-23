"""
06_2_ee_marker_visualization.py - End-Effector 위치 마커 시각화 (rviz2 Marker 스타일)

06_1 의 중력 보상 비교 위에, ROS 2 rviz2 의 visualization_msgs/Marker 처럼
로봇팔 끝단(End-Effector) 위치를 Isaac Sim viewport 에 마커로 표시합니다.

각 로봇마다 두 종류의 EE 마커를 표시합니다:
  - 노란 큐브  (Target):  현재 조인트 지령(target) 으로 FK 계산한 끝단 위치
  - 빨간/초록 구 (Actual): 실제 조인트 각도에서 PhysX 가 계산한 EE 위치
                          - Robot A → 빨간 구 (중력 보상 없음)
                          - Robot B → 초록 구 (중력 보상 있음)

비교 관찰:
  - Robot A: 노란 Target 과 빨간 Actual 이 크게 벌어집니다 (엘보 처짐)
  - Robot B: 노란 Target 과 초록 Actual 이 거의 겹칩니다 (잘 추종)

Target FK 계산 방법 — 'Ghost robot' 트릭:
  보이지 않는 'ghost' Franka 를 z=100 m 상공에 두고 매 step 마다 target joint 로
  강제 세팅(write_joint_state_to_sim) → PhysX 가 내부 FK 로 갱신해주는 panda_hand
  body 위치를 그대로 읽어 사용합니다.

  ※ 참고 — Isaac Sim 에는 정통 FK 솔버도 있습니다:
    isaacsim.robot_motion.motion_generation.lula.kinematics.LulaKinematicsSolver
      → solver.compute_forward_kinematics(frame_name, joint_positions)
    이 방식은 시뮬레이션과 무관하게 수학적으로 EE 를 계산해서 더 정통적이지만
    robot_description.yaml + URDF 경로 설정이 필요합니다. 본 예제에서는 IsaacLab
    API 만으로 학습 곡선을 낮추려고 ghost robot 패턴을 선택했습니다.

  Ghost 의 EE 월드 좌표에서 ghost origin (0,0,100) 을 빼면 로봇 로컬 좌표 target EE,
  각 로봇 origin 을 더해 두 로봇 옆에 동일한 target 마커를 표시.

실행:
    source env_isaaclab/bin/activate
    cd lectures/06_robot_joint_control
    python 06_2_ee_marker_visualization.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="06-2 - EE Marker Visualization")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import os
import torch
from collections import deque

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sim import SimulationCfg, SimulationContext

from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip

# ── 3. Matplotlib ────────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use("TkAgg")
    INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE = False
import matplotlib.pyplot as plt


# 로봇 origin 위치 (월드 좌표)
ROBOT_A_ORIGIN = torch.tensor([0.0, 0.0, 0.0])  # 보상 없음
ROBOT_B_ORIGIN = torch.tensor([2.0, 0.0, 0.0])  # 보상 있음
GHOST_ORIGIN  = torch.tensor([0.0, 0.0, 100.0])  # offscreen 상공 — 시각적으로 안 보임

# EE 바디 이름 (Franka)
EE_BODY_NAME = "panda_hand"


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성
# ══════════════════════════════════════════════════════════════════════════

def design_scene():
    """바닥, 조명, Franka 로봇 3대(A, B, Ghost)를 배치합니다.

    Ghost 로봇은 z=100 m 상공에 두어 카메라에 잡히지 않게 합니다.
    매 step 마다 target joint 로 강제 세팅되어 'FK from command' 계산용으로만 사용.
    """
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)

    # 로봇 A: 중력 보상 없음 (원점)
    sim_utils.create_prim("/World/OriginA", "Xform", translation=ROBOT_A_ORIGIN.tolist())
    cfg_a = FRANKA_PANDA_CFG.copy()
    cfg_a.prim_path = "/World/OriginA/Robot"
    robot_a = Articulation(cfg=cfg_a)

    # 로봇 B: 중력 보상 있음 (x=2.0m 옆)
    sim_utils.create_prim("/World/OriginB", "Xform", translation=ROBOT_B_ORIGIN.tolist())
    cfg_b = FRANKA_PANDA_CFG.copy()
    cfg_b.prim_path = "/World/OriginB/Robot"
    robot_b = Articulation(cfg=cfg_b)

    # Ghost 로봇: target FK 전용. 상공에 두어 시각적으로 안 보이게.
    sim_utils.create_prim("/World/OriginGhost", "Xform", translation=GHOST_ORIGIN.tolist())
    cfg_ghost = FRANKA_PANDA_CFG.copy()
    cfg_ghost.prim_path = "/World/OriginGhost/Robot"
    robot_ghost = Articulation(cfg=cfg_ghost)

    return robot_a, robot_b, robot_ghost


def build_ee_markers():
    """EE 위치 시각화 마커 — rviz2 Marker 처럼 세 가지 prototype 으로 구성."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/World/Visuals/EEMarkers",
        markers={
            # 0: target (FK from command) — 노란색 큐브
            "target": sim_utils.CuboidCfg(
                size=(0.06, 0.06, 0.06),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.9, 0.1),
                    emissive_color=(0.4, 0.35, 0.0),  # 살짝 빛나게
                ),
            ),
            # 1: actual_a (Robot A 실제 EE, 보상 없음) — 빨간 구
            "actual_a": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.15, 0.15),
                    emissive_color=(0.4, 0.05, 0.05),
                ),
            ),
            # 2: actual_b (Robot B 실제 EE, 보상 있음) — 초록 구
            "actual_b": sim_utils.SphereCfg(
                radius=0.045,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.15, 1.0, 0.3),
                    emissive_color=(0.05, 0.4, 0.1),
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)
    # 두 로봇이 다 보이는 시점
    sim.set_camera_view(eye=[3.5, 3.5, 2.5], target=[1.0, 0.0, 0.5])

    robot_a, robot_b, robot_ghost = design_scene()
    ee_markers = build_ee_markers()
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot_a.update(sim_dt)
    robot_b.update(sim_dt)
    robot_ghost.update(sim_dt)

    # EE body index — Franka 의 panda_hand
    ee_idx = robot_a.data.body_names.index(EE_BODY_NAME)
    print(f"[INFO] EE body '{EE_BODY_NAME}' index = {ee_idx}")
    print("[INFO] Robot A: no gravity compensation (Actual = 빨간 구)")
    print("[INFO] Robot B: with gravity compensation (Actual = 초록 구)")
    print("[INFO] Target (FK from command) = 노란 큐브  (두 로봇 옆에 각각 표시)")

    # 디바이스 일치를 위해 origin 텐서를 sim device 로 이동
    robot_a_origin = ROBOT_A_ORIGIN.to(sim.device)
    robot_b_origin = ROBOT_B_ORIGIN.to(sim.device)
    ghost_origin   = GHOST_ORIGIN.to(sim.device)

    # ── 제어 파라미터 ────────────────────────────────────────────────────
    J0_AMP, J0_FREQ = 0.5, 0.3
    J3_AMP, J3_FREQ = 0.3, 0.5
    J3_CENTER = -2.0

    TOTAL_TIME = 15.0
    TOTAL_STEPS = int(TOTAL_TIME / sim_dt)

    # ── 중력 보상 API 확인 ───────────────────────────────────────────────
    has_gravity_api = hasattr(robot_b.root_physx_view, "get_gravity_compensation_forces") or \
                      hasattr(robot_b.root_physx_view, "get_generalized_gravity_forces")
    if has_gravity_api:
        print("[INFO] PhysX gravity compensation API available")

    # ── 데이터 버퍼 (전체 시간 보존) ─────────────────────────────────────
    WINDOW_SEC = 5.0
    PLOT_INTERVAL = 3

    t_buf = deque()
    # EE 위치 거리 (target ↔ actual)
    ee_err_a = deque()  # Robot A target - actual 거리 (m)
    ee_err_b = deque()  # Robot B target - actual 거리 (m)

    # ── 잔상 트레일 버퍼 ────────────────────────────────────────────────
    # 각 EE 마커의 과거 위치를 deque 에 저장 → 매 프레임 전체를 그려서 잔상 효과.
    # 오래된 점은 scale 을 작게 줘서 자연스럽게 페이드아웃 되도록 합니다.
    TRAIL_LEN = 40        # 잔상 점 개수
    TRAIL_SAMPLE = 2      # 매 N step 마다 trail 에 추가 (1=매 step)
    MAX_SCALE = 1.0       # 가장 최근 점의 크기
    MIN_SCALE = 0.15      # 가장 오래된 점의 크기

    trail_target_a = deque(maxlen=TRAIL_LEN)
    trail_target_b = deque(maxlen=TRAIL_LEN)
    trail_actual_a = deque(maxlen=TRAIL_LEN)
    trail_actual_b = deque(maxlen=TRAIL_LEN)

    # ── matplotlib: 두 로봇의 EE 추종 오차 시각화 ────────────────────────
    if INTERACTIVE:
        plt.ion()

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    fig.suptitle("EE Position Tracking Error — Gravity Compensation Comparison", fontsize=12)
    ln_err_a, = ax.plot([], [], "-", color="tab:red",   linewidth=1.5, label="Robot A (no comp)")
    ln_err_b, = ax.plot([], [], "-", color="tab:green", linewidth=1.5, label="Robot B (with comp)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("‖target_EE − actual_EE‖ (m)")
    ax.set_ylim(0.0, 0.4)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if INTERACTIVE:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # ── 로봇 리셋 ────────────────────────────────────────────────────────
    for robot in [robot_a, robot_b, robot_ghost]:
        default_pos = robot.data.default_joint_pos.clone()
        default_vel = torch.zeros_like(default_pos)
        robot.write_joint_state_to_sim(default_pos, default_vel)
        robot.reset()
        robot.update(sim_dt)

    print(f"\n[INFO] Running {TOTAL_STEPS} steps ({TOTAL_TIME}s)...")
    print("=" * 60)

    zero_vel = torch.zeros_like(robot_ghost.data.default_joint_pos)

    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        t = step * sim_dt

        # ── 동일한 사인파 목표 (Robot A, B, Ghost 모두 같은 target) ──────
        target_pos = robot_a.data.default_joint_pos.clone()
        target_pos[:, 0] = J0_AMP * math.sin(2.0 * math.pi * J0_FREQ * t)
        target_pos[:, 3] = J3_CENTER + J3_AMP * math.sin(2.0 * math.pi * J3_FREQ * t)

        # ── Robot A: 위치 제어만 ─────────────────────────────────────────
        robot_a.set_joint_position_target(target_pos)
        robot_a.write_data_to_sim()

        # ── Robot B: 위치 제어 + 중력 보상 ───────────────────────────────
        robot_b.set_joint_position_target(target_pos)
        if has_gravity_api:
            if hasattr(robot_b.root_physx_view, "get_gravity_compensation_forces"):
                gravity_torques = robot_b.root_physx_view.get_gravity_compensation_forces()
            else:
                gravity_torques = robot_b.root_physx_view.get_generalized_gravity_forces()
            robot_b.set_joint_effort_target(gravity_torques)
        robot_b.write_data_to_sim()

        # ── Ghost: target joint 로 강제 세팅 → FK 계산 ───────────────────
        # write_joint_state_to_sim 은 PhysX 상태를 직접 덮어쓰므로 중력 영향 없이
        # 매 프레임 target 자세를 유지합니다.
        robot_ghost.write_joint_state_to_sim(target_pos, zero_vel)
        robot_ghost.write_data_to_sim()

        # ── 물리 스텝 ───────────────────────────────────────────────────
        sim.step()
        robot_a.update(sim_dt)
        robot_b.update(sim_dt)
        robot_ghost.update(sim_dt)

        # ── EE 위치 (월드 좌표) 추출 ─────────────────────────────────────
        # body_state_w shape: (num_envs=1, num_bodies, 13) — [pos(3), quat(4), lin_vel(3), ang_vel(3)]
        ee_actual_a = robot_a.data.body_state_w[0, ee_idx, :3]      # (3,)
        ee_actual_b = robot_b.data.body_state_w[0, ee_idx, :3]      # (3,)
        ee_target_ghost = robot_ghost.data.body_state_w[0, ee_idx, :3]  # ghost 의 EE 위치

        # Ghost 의 origin offset 을 빼서 로봇 로컬 좌표계의 target EE 위치로 변환
        ee_target_local = ee_target_ghost - ghost_origin

        # 각 로봇의 origin 에 맞춰 target marker 위치 계산
        ee_target_for_a = ee_target_local + robot_a_origin
        ee_target_for_b = ee_target_local + robot_b_origin

        # ── 마커 + 잔상 갱신 ────────────────────────────────────────────
        # TRAIL_SAMPLE step 마다 현재 위치를 각 trail 에 추가
        if step % TRAIL_SAMPLE == 0:
            trail_target_a.append(ee_target_for_a.detach().clone())
            trail_target_b.append(ee_target_for_b.detach().clone())
            trail_actual_a.append(ee_actual_a.detach().clone())
            trail_actual_b.append(ee_actual_b.detach().clone())

        # 4개 trail 을 하나의 마커 배열로 합칩니다.
        # 각 trail 의 i 번째 점은 oldest→newest 순서 → scale 도 작은→큰 순으로.
        positions = []
        marker_idx_list = []
        scales_list = []
        for trail, mi in [
            (trail_target_a, 0),  # target (노란 큐브)
            (trail_target_b, 0),
            (trail_actual_a, 1),  # actual A (빨간 구)
            (trail_actual_b, 2),  # actual B (초록 구)
        ]:
            n = len(trail)
            if n == 0:
                continue
            for j, pos in enumerate(trail):
                positions.append(pos)
                marker_idx_list.append(mi)
                # j=0 가장 오래된 점 → MIN_SCALE, j=n-1 최신 → MAX_SCALE
                frac = (j + 1) / n
                s = MIN_SCALE + (MAX_SCALE - MIN_SCALE) * frac
                scales_list.append([s, s, s])

        if positions:
            translations = torch.stack(positions, dim=0)
            marker_indices = torch.tensor(marker_idx_list, device=sim.device)
            scales = torch.tensor(scales_list, device=sim.device)
            ee_markers.visualize(
                translations=translations,
                marker_indices=marker_indices,
                scales=scales,
            )

        # ── EE 추종 오차 기록 ────────────────────────────────────────────
        err_a = float(torch.linalg.norm(ee_target_for_a - ee_actual_a))
        err_b = float(torch.linalg.norm(ee_target_for_b - ee_actual_b))

        t_buf.append(t)
        ee_err_a.append(err_a)
        ee_err_b.append(err_b)

        # ── 그래프 갱신 ──────────────────────────────────────────────────
        if INTERACTIVE and step % PLOT_INTERVAL == 0 and len(t_buf) > 1:
            t_list = list(t_buf)
            x_min = max(0, t - WINDOW_SEC)
            x_max = max(WINDOW_SEC, t)
            ln_err_a.set_data(t_list, list(ee_err_a))
            ln_err_b.set_data(t_list, list(ee_err_b))
            ax.set_xlim(x_min, x_max)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # ── 콘솔 출력 ───────────────────────────────────────────────────
        if (step + 1) % 200 == 0:
            print(
                f"  [Step {step + 1:4d} | t={t:.1f}s] "
                f"EE error — no comp: {err_a*1000:6.1f} mm, with comp: {err_b*1000:6.1f} mm"
            )

    # ── 최종 그래프 저장 ─────────────────────────────────────────────────
    print(f"\n[INFO] Simulation complete — {TOTAL_STEPS} steps")

    if len(t_buf) > 1:
        ax.set_xlim(0, TOTAL_TIME)
        if INTERACTIVE:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "06_2_ee_marker_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Graph saved: {save_path}")

    if INTERACTIVE:
        plt.ioff()
        print("[INFO] Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)

    simulation_app.close()
    print("[DONE] Simulation ended")


if __name__ == "__main__":
    main()
