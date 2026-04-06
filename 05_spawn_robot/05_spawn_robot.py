"""
05_spawn_robot.py - 로봇 아티큘레이션 스폰하기

IsaacLab에서 로봇(Articulation)을 불러와 씬에 배치하는 방법을 배웁니다:
  - 사전 정의된 로봇 설정(FRANKA_PANDA_CFG) 사용
  - Xform 원점을 이용한 로봇 배치
  - Articulation 객체 생성 및 관절 정보 조회
  - 랜덤 토크를 가해 로봇 움직이기

실행:
    source env_isaaclab/bin/activate
    cd lectures/05_spawn_robot
    python 05_spawn_robot.py

    # GUI 없이 실행
    python 05_spawn_robot.py --headless
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="05 - 로봇 스폰 튜토리얼")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationCfg, SimulationContext

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# isaaclab_assets 패키지에는 다양한 로봇의 ArticulationCfg가 미리 정의되어 있습니다.
# FRANKA_PANDA_CFG: 프랑카 판다 (7+2 관절) - 산업용 로봇팔
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성 함수
# ══════════════════════════════════════════════════════════════════════════


def design_scene() -> Articulation:
    """씬에 바닥, 조명, Franka Panda 로봇을 배치합니다."""

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # ── 돔 조명 (DomeLight) ──────────────────────────────────────────────
    # DomeLight는 전방향 환경광을 제공합니다 (사실적인 조명)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    # ── Xform 원점 생성 ──────────────────────────────────────────────────
    # 로봇의 원점(Origin)을 별도의 Xform 프림으로 만듭니다.
    # Xform은 USD에서 좌표계(transform)를 나타내는 기본 프림 타입입니다.
    sim_utils.create_prim("/World/Origin1", "Xform", translation=[0.0, 0.0, 0.0])

    # ── Franka Panda 로봇 스폰 ───────────────────────────────────────────
    # .copy()로 원본 설정을 복사한 뒤 prim_path를 지정합니다.
    # prim_path는 USD 스테이지에서 로봇이 위치할 경로입니다.
    # Franka는 7자유도 로봇팔 + 2개의 그리퍼 핑거 관절을 가집니다.
    franka_cfg = FRANKA_PANDA_CFG.copy()
    franka_cfg.prim_path = "/World/Origin1/Robot"
    franka = Articulation(cfg=franka_cfg)

    return franka


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Franka Panda 로봇을 스폰하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)

    # 카메라 시점: 로봇이 보이도록 설정
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.5])

    # ── 씬 구성 ──────────────────────────────────────────────────────────
    franka = design_scene()

    # ── sim.reset() - 물리 엔진 초기화 ────────────────────────────────────
    # reset() 이후에 로봇의 관절 정보가 활성화됩니다.
    # 반드시 step() 전에 호출해야 합니다!
    sim.reset()
    print("[INFO] sim.reset() 완료 - 물리 엔진 초기화 완료")

    # ── 관절 정보 출력 ────────────────────────────────────────────────────
    # joint_names: 로봇이 가진 관절들의 이름 리스트
    # num_joints: 관절의 총 개수
    print("=" * 60)
    print("[Franka Panda 로봇 정보]")
    print(f"  관절 이름: {franka.joint_names}")
    print(f"  관절 수:   {franka.num_joints}")
    print("=" * 60)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    step_count = 0

    while simulation_app.is_running():
        # 시뮬레이션 정지/일시정지 처리
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 500 step마다 에피소드 리셋 ────────────────────────────────
        if step_count % 500 == 0:
            root_state = franka.data.default_root_state.clone()
            franka.write_root_pose_to_sim(root_state[:, :7])
            franka.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = (
                franka.data.default_joint_pos.clone(),
                franka.data.default_joint_vel.clone(),
            )
            franka.write_joint_state_to_sim(joint_pos, joint_vel)
            franka.reset()

            print(f"\n[INFO] 에피소드 리셋 (step={step_count})")

        # ── (2) 랜덤 토크 적용 ────────────────────────────────────────────
        # randn_like로 관절 개수에 맞는 랜덤 힘을 생성하여 적용합니다.
        # Franka: 9개 관절에 랜덤 토크
        efforts = torch.randn_like(franka.data.joint_pos) * 5.0
        franka.set_joint_effort_target(efforts)
        franka.write_data_to_sim()

        # ── (3) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()
        step_count += 1

        # ── (4) 로봇 버퍼 업데이트 ───────────────────────────────────────
        # update()를 호출해야 data.joint_pos 등의 값이 최신 상태로 갱신됩니다.
        franka.update(sim_dt)

        # ── (5) 200 step마다 관절 위치 출력 ───────────────────────────────
        if step_count % 200 == 0:
            print(f"\n[Step {step_count}]")
            print(f"  Franka joint pos: {franka.data.joint_pos}")

    # ── 종료 ──────────────────────────────────────────────────────────────
    simulation_app.close()
    print(f"\n[DONE] 시뮬레이션 종료 - 총 {step_count} steps 실행")


if __name__ == "__main__":
    main()
