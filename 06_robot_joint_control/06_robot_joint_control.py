"""
06_robot_joint_control.py - 로봇 관절 위치 제어

Franka Panda 로봇의 joint0(베이스)과 joint3(엘보)에 사인파 위치 명령을 주고,
목표값과 실제값을 실시간 matplotlib 그래프로 비교합니다.

그래프는 약 5초 폭의 스크롤 윈도우로 실시간 업데이트됩니다.

실행:
    source env_isaaclab/bin/activate
    cd lectures/06_robot_joint_control
    python 06_robot_joint_control.py

    # GUI 없이 실행 (그래프는 파일로 저장)
    python 06_robot_joint_control.py --headless
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="06 - 로봇 관절 위치 제어")
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

# ── 3. Matplotlib 설정 (IsaacSim 초기화 이후) ────────────────────────────
# IsaacSim GUI와 별도의 Tk 윈도우에서 그래프를 표시합니다.
import matplotlib
try:
    matplotlib.use("TkAgg")
    INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE = False
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성
# ══════════════════════════════════════════════════════════════════════════


def design_scene() -> Articulation:
    """바닥, 조명, Franka Panda 로봇을 배치합니다."""
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.95, 0.95, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)

    franka_cfg: ArticulationCfg = FRANKA_PANDA_CFG.copy()
    franka_cfg.prim_path = "/World/Origin/Robot"
    return Articulation(cfg=franka_cfg)


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Joint0, Joint3에 사인파 위치 명령을 주고 실시간 그래프로 비교합니다."""

    # ── SimulationContext ─────────────────────────────────────────────────
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.5])

    robot = design_scene()
    sim.reset()

    sim_dt = sim.get_physics_dt()
    robot.update(sim_dt)

    print("[INFO] sim.reset() complete")
    print(f"[INFO] Joint names: {robot.data.joint_names}")
    print(f"[INFO] Num joints:  {robot.num_joints}")
    print(f"[INFO] Physics dt:  {sim_dt:.6f}s")

    # ── 제어 파라미터 ────────────────────────────────────────────────────
    # Joint 0 (panda_joint1, base rotation): sine, amp=0.5 rad, freq=0.3 Hz
    # Joint 3 (panda_joint4, elbow):         sine, amp=0.3 rad, freq=0.5 Hz
    #   panda_joint4 limits: -3.0545 ~ -0.0698 rad
    #   default=-2.810은 하한에 너무 가까우므로 center를 -2.0으로 설정
    J0_AMP = 0.5        # rad
    J0_FREQ = 0.3       # Hz
    J3_AMP = 0.3        # rad
    J3_FREQ = 0.5       # Hz
    j3_default = -2.0   # 관절 리밋 중앙 부근 (default -2.810은 하한에 너무 가까움)

    TOTAL_TIME = 15.0   # 총 시뮬레이션 시간 (초)
    TOTAL_STEPS = int(TOTAL_TIME / sim_dt)

    print(f"\n[INFO] Position Control: Joint0 & Joint3 (sine wave)")
    print(f"  Joint0: amp={J0_AMP} rad, freq={J0_FREQ} Hz")
    print(f"  Joint3: amp={J3_AMP} rad, freq={J3_FREQ} Hz, center={j3_default:.3f} rad")
    print(f"  Duration: {TOTAL_TIME}s ({TOTAL_STEPS} steps)")
    print("=" * 60)

    # ── 그래프 윈도우 설정 ────────────────────────────────────────────────
    WINDOW_SEC = 5.0     # 스크롤 윈도우 폭 (초)
    WINDOW_N = int(WINDOW_SEC / sim_dt)
    PLOT_INTERVAL = 3    # N step마다 그래프 갱신

    # 데이터 버퍼 — 전체 시간을 보존 (라이브 스크롤은 아래 set_xlim 으로 처리).
    # maxlen 으로 자르면 저장된 PNG 에서 마지막 WINDOW_SEC 구간만 보이게 됩니다.
    t_buf = deque()
    tgt_j0 = deque()
    act_j0 = deque()
    tgt_j3 = deque()
    act_j3 = deque()

    # ── matplotlib 실시간 그래프 초기화 ──────────────────────────────────
    if INTERACTIVE:
        plt.ion()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5))
    fig.suptitle("Position Control — Target (dashed) vs Actual (solid)", fontsize=13)

    # Joint 0 subplot
    ln_tgt0, = ax1.plot([], [], "--", color="tab:blue", linewidth=2, label="Target")
    ln_act0, = ax1.plot([], [], "-", color="tab:red", linewidth=1.2, label="Actual")
    ax1.set_ylabel("Joint0 Position (rad)")
    ax1.set_ylim(-J0_AMP * 1.5, J0_AMP * 1.5)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Joint 0 (Base Rotation) — 0.3 Hz sine wave")

    # Joint 3 subplot
    ln_tgt3, = ax2.plot([], [], "--", color="tab:blue", linewidth=2, label="Target")
    ln_act3, = ax2.plot([], [], "-", color="tab:red", linewidth=1.2, label="Actual")
    ax2.set_ylabel("Joint3 Position (rad)")
    ax2.set_ylim(j3_default - J3_AMP * 1.8, j3_default + J3_AMP * 1.8)
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Joint 3 (Elbow) — 0.5 Hz sine wave")

    plt.tight_layout()

    if INTERACTIVE:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # ── 로봇 초기 자세 리셋 ──────────────────────────────────────────────
    default_pos = robot.data.default_joint_pos.clone()
    default_vel = torch.zeros_like(default_pos)
    robot.write_joint_state_to_sim(default_pos, default_vel)
    robot.reset()
    robot.update(sim_dt)

    # ══════════════════════════════════════════════════════════════════════
    #  시뮬레이션 루프 (1회, TOTAL_STEPS)
    # ══════════════════════════════════════════════════════════════════════
    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        # ── (1) 사인파 위치 목표 생성 ────────────────────────────────────
        t = step * sim_dt
        target_pos = robot.data.default_joint_pos.clone()
        target_pos[:, 0] = J0_AMP * math.sin(2.0 * math.pi * J0_FREQ * t)
        target_pos[:, 3] = j3_default + J3_AMP * math.sin(2.0 * math.pi * J3_FREQ * t)

        # ── (2) 위치 명령 적용 ───────────────────────────────────────────
        robot.set_joint_position_target(target_pos)
        robot.write_data_to_sim()

        # ── (3) 물리 스텝 ────────────────────────────────────────────────
        sim.step()
        robot.update(sim_dt)

        # ── (4) 데이터 기록 ──────────────────────────────────────────────
        t_buf.append(t)
        tgt_j0.append(target_pos[0, 0].item())
        act_j0.append(robot.data.joint_pos[0, 0].item())
        tgt_j3.append(target_pos[0, 3].item())
        act_j3.append(robot.data.joint_pos[0, 3].item())

        # ── (5) 실시간 그래프 갱신 ───────────────────────────────────────
        if INTERACTIVE and step % PLOT_INTERVAL == 0 and len(t_buf) > 1:
            t_list = list(t_buf)
            ln_tgt0.set_data(t_list, list(tgt_j0))
            ln_act0.set_data(t_list, list(act_j0))
            ln_tgt3.set_data(t_list, list(tgt_j3))
            ln_act3.set_data(t_list, list(act_j3))

            # 스크롤: x축을 현재 시각 기준 ±WINDOW_SEC 폭으로 유지
            x_min = max(0, t - WINDOW_SEC)
            x_max = max(WINDOW_SEC, t)
            ax1.set_xlim(x_min, x_max)
            ax2.set_xlim(x_min, x_max)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # ── (6) 콘솔 출력 (100 step마다) ─────────────────────────────────
        if (step + 1) % 100 == 0:
            err_j0 = abs(target_pos[0, 0].item() - robot.data.joint_pos[0, 0].item())
            err_j3 = abs(target_pos[0, 3].item() - robot.data.joint_pos[0, 3].item())
            print(
                f"  [Step {step + 1:4d} | t={t:.2f}s] "
                f"J0: tgt={target_pos[0, 0].item():+.3f} act={robot.data.joint_pos[0, 0].item():+.3f} err={err_j0:.4f} | "
                f"J3: tgt={target_pos[0, 3].item():+.3f} act={robot.data.joint_pos[0, 3].item():+.3f} err={err_j3:.4f}"
            )

    print(f"\n[INFO] Simulation complete — {TOTAL_STEPS} steps ({TOTAL_TIME}s)")

    # ── 최종 그래프 저장 (전체 구간) ─────────────────────────────────────
    # 스크롤 윈도우를 전체 범위로 확장하여 최종 이미지 저장
    if len(t_buf) > 1:
        ax1.set_xlim(0, TOTAL_TIME)
        ax2.set_xlim(0, TOTAL_TIME)
        if INTERACTIVE:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "06_control_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Final graph saved: {save_path}")

    if INTERACTIVE:
        plt.ioff()
        print("[INFO] Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)
        print("[INFO] Headless mode — graph saved to file only.")

    simulation_app.close()
    print("[DONE] Simulation ended")


if __name__ == "__main__":
    main()
