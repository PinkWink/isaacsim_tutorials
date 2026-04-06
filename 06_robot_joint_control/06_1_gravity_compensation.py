"""
06_1_gravity_compensation.py - 중력 보상을 적용한 위치 제어

06강의 위치 제어에서 Joint 3(Elbow)에 중력에 의한 오프셋이 발생하는 문제를
중력 보상(Gravity Compensation)으로 해결합니다.

PhysX의 get_generalized_gravity_forces()로 각 관절에 작용하는 중력 토크를 계산하고,
이를 effort_target에 더해 중력을 상쇄합니다.

비교 실험:
  - 좌측 subplot: 중력 보상 없음 (06강과 동일)
  - 우측 subplot: 중력 보상 적용

실행:
    source env_isaaclab/bin/activate
    cd lectures/06_robot_joint_control
    python 06_1_gravity_compensation.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="06-1 - Gravity Compensation")
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


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성: 로봇 2대 (보상 없음 / 보상 있음)
# ══════════════════════════════════════════════════════════════════════════


def design_scene():
    """바닥, 조명, Franka 로봇 2대를 나란히 배치합니다."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)

    # 로봇 A: 중력 보상 없음 (원점)
    sim_utils.create_prim("/World/OriginA", "Xform", translation=[0.0, 0.0, 0.0])
    cfg_a = FRANKA_PANDA_CFG.copy()
    cfg_a.prim_path = "/World/OriginA/Robot"
    robot_a = Articulation(cfg=cfg_a)

    # 로봇 B: 중력 보상 있음 (x=2.0m 옆)
    sim_utils.create_prim("/World/OriginB", "Xform", translation=[2.0, 0.0, 0.0])
    cfg_b = FRANKA_PANDA_CFG.copy()
    cfg_b.prim_path = "/World/OriginB/Robot"
    robot_b = Articulation(cfg=cfg_b)

    return robot_a, robot_b


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """중력 보상 유/무를 비교합니다."""

    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)
    sim.set_camera_view(eye=[3.0, 3.0, 2.5], target=[1.0, 0.0, 0.5])

    robot_a, robot_b = design_scene()
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot_a.update(sim_dt)
    robot_b.update(sim_dt)

    print("[INFO] sim.reset() complete")
    print(f"[INFO] Robot A: no gravity compensation")
    print(f"[INFO] Robot B: with gravity compensation")

    # ── 제어 파라미터 ────────────────────────────────────────────────────
    J0_AMP, J0_FREQ = 0.5, 0.3
    J3_AMP, J3_FREQ = 0.3, 0.5
    J3_CENTER = -2.0

    TOTAL_TIME = 15.0
    TOTAL_STEPS = int(TOTAL_TIME / sim_dt)

    # ── 중력 보상 API 확인 ───────────────────────────────────────────────
    has_gravity_api = hasattr(robot_b.root_physx_view, "get_generalized_gravity_forces")
    if has_gravity_api:
        print("[INFO] PhysX gravity compensation API available")
    else:
        print("[WARNING] get_generalized_gravity_forces() not found, trying alternative...")
        # 사용 가능한 메서드 출력
        methods = [m for m in dir(robot_b.root_physx_view) if "grav" in m.lower() or "force" in m.lower()]
        print(f"  Available methods: {methods}")

    # ── 데이터 버퍼 ──────────────────────────────────────────────────────
    WINDOW_SEC = 5.0
    WINDOW_N = int(WINDOW_SEC / sim_dt)
    PLOT_INTERVAL = 3

    t_buf = deque(maxlen=WINDOW_N)
    # Robot A (no compensation)
    tgt_j3 = deque(maxlen=WINDOW_N)
    act_a_j3 = deque(maxlen=WINDOW_N)
    # Robot B (with compensation)
    act_b_j3 = deque(maxlen=WINDOW_N)

    # ── matplotlib 설정 ──────────────────────────────────────────────────
    if INTERACTIVE:
        plt.ion()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle("Joint 3 (Elbow) Position Control — Gravity Compensation Comparison", fontsize=12)

    # Left: no compensation
    ln_tgt_a, = ax1.plot([], [], "--", color="tab:blue", linewidth=2, label="Target")
    ln_act_a, = ax1.plot([], [], "-", color="tab:red", linewidth=1.2, label="Actual")
    ax1.set_title("Without Gravity Compensation")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Joint3 Position (rad)")
    ax1.set_ylim(J3_CENTER - J3_AMP * 2.5, J3_CENTER + J3_AMP * 2.5)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: with compensation
    ln_tgt_b, = ax2.plot([], [], "--", color="tab:blue", linewidth=2, label="Target")
    ln_act_b, = ax2.plot([], [], "-", color="tab:green", linewidth=1.2, label="Actual (compensated)")
    ax2.set_title("With Gravity Compensation")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if INTERACTIVE:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # ── 로봇 리셋 ────────────────────────────────────────────────────────
    for robot in [robot_a, robot_b]:
        default_pos = robot.data.default_joint_pos.clone()
        default_vel = torch.zeros_like(default_pos)
        robot.write_joint_state_to_sim(default_pos, default_vel)
        robot.reset()
        robot.update(sim_dt)

    # ══════════════════════════════════════════════════════════════════════
    #  시뮬레이션 루프
    # ══════════════════════════════════════════════════════════════════════
    print(f"\n[INFO] Running {TOTAL_STEPS} steps ({TOTAL_TIME}s)...")
    print("=" * 60)

    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        t = step * sim_dt

        # ── 동일한 사인파 목표 ───────────────────────────────────────────
        target_pos_a = robot_a.data.default_joint_pos.clone()
        target_pos_a[:, 0] = J0_AMP * math.sin(2.0 * math.pi * J0_FREQ * t)
        target_pos_a[:, 3] = J3_CENTER + J3_AMP * math.sin(2.0 * math.pi * J3_FREQ * t)

        target_pos_b = robot_b.data.default_joint_pos.clone()
        target_pos_b[:, 0] = J0_AMP * math.sin(2.0 * math.pi * J0_FREQ * t)
        target_pos_b[:, 3] = J3_CENTER + J3_AMP * math.sin(2.0 * math.pi * J3_FREQ * t)

        # ── Robot A: 위치 제어만 (중력 보상 없음) ────────────────────────
        robot_a.set_joint_position_target(target_pos_a)
        robot_a.write_data_to_sim()

        # ── Robot B: 위치 제어 + 중력 보상 ───────────────────────────────
        robot_b.set_joint_position_target(target_pos_b)

        if has_gravity_api:
            # PhysX에서 각 관절의 중력 토크를 계산
            gravity_torques = robot_b.root_physx_view.get_generalized_gravity_forces()
            # 중력 토크를 effort에 더해 상쇄
            robot_b.set_joint_effort_target(gravity_torques)

        robot_b.write_data_to_sim()

        # ── 물리 스텝 ───────────────────────────────────────────────────
        sim.step()
        robot_a.update(sim_dt)
        robot_b.update(sim_dt)

        # ── 데이터 기록 ──────────────────────────────────────────────────
        t_buf.append(t)
        tgt_j3.append(target_pos_a[0, 3].item())
        act_a_j3.append(robot_a.data.joint_pos[0, 3].item())
        act_b_j3.append(robot_b.data.joint_pos[0, 3].item())

        # ── 그래프 갱신 ──────────────────────────────────────────────────
        if INTERACTIVE and step % PLOT_INTERVAL == 0 and len(t_buf) > 1:
            t_list = list(t_buf)
            x_min = max(0, t - WINDOW_SEC)
            x_max = max(WINDOW_SEC, t)

            ln_tgt_a.set_data(t_list, list(tgt_j3))
            ln_act_a.set_data(t_list, list(act_a_j3))
            ax1.set_xlim(x_min, x_max)

            ln_tgt_b.set_data(t_list, list(tgt_j3))
            ln_act_b.set_data(t_list, list(act_b_j3))
            ax2.set_xlim(x_min, x_max)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # ── 콘솔 출력 ───────────────────────────────────────────────────
        if (step + 1) % 200 == 0:
            err_a = abs(target_pos_a[0, 3].item() - robot_a.data.joint_pos[0, 3].item())
            err_b = abs(target_pos_b[0, 3].item() - robot_b.data.joint_pos[0, 3].item())
            print(
                f"  [Step {step + 1:4d} | t={t:.1f}s] "
                f"J3 error — no comp: {err_a:.4f} rad, with comp: {err_b:.4f} rad"
            )

    # ── 최종 그래프 저장 ─────────────────────────────────────────────────
    print(f"\n[INFO] Simulation complete — {TOTAL_STEPS} steps")

    if len(t_buf) > 1:
        ax1.set_xlim(0, TOTAL_TIME)
        ax2.set_xlim(0, TOTAL_TIME)
        if INTERACTIVE:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_1_gravity_comp_result.png")
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
