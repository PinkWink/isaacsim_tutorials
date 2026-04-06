"""
04_simulation_loop.py - 시뮬레이션 루프 상세 이해

IsaacLab 시뮬레이션의 핵심 루프 구조를 배웁니다:
  - sim.reset()으로 초기화
  - sim.step()으로 물리 + 렌더링 진행
  - dt(시간 간격)를 이용한 시뮬레이션 시간 추적
  - 에피소드 리셋 개념 (주기적으로 초기 상태 복원)

실행:
    source env_isaaclab/bin/activate
    python 04_simulation_loop.py --num_envs 1

    # GUI 없이 실행
    python 04_simulation_loop.py --headless
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="04 - 시뮬레이션 루프 튜토리얼")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext

# ══════════════════════════════════════════════════════════════════════════
# 씬 구성 함수
# ══════════════════════════════════════════════════════════════════════════

# 구(sphere)의 초기 높이 – 이 높이에서 떨어뜨립니다
SPHERE_START_HEIGHT = 2.0
SPHERE_RADIUS = 0.15
SPHERE_PRIM_PATH = "/World/Sphere"


def design_scene() -> None:
    """씬에 바닥, 조명, 강체 구(sphere)를 배치합니다."""

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # 물리 충돌이 가능한 기본 바닥을 생성합니다.
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # ── 원거리 조명 (DistantLight) ───────────────────────────────────────
    # 태양광처럼 평행 광선을 내는 조명입니다.
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    # ── 강체 구 (Rigid Body Sphere) ──────────────────────────────────────
    # rigid_props 를 설정하면 물리 시뮬레이션의 영향을 받는 강체가 됩니다.
    # restitution(반발계수)을 높여서 바운스되도록 설정합니다.
    sphere_cfg = sim_utils.SphereCfg(
        radius=SPHERE_RADIUS,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.6, 1.0),  # 파란색
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.5,
            dynamic_friction=0.5,
            restitution=0.8,  # 반발계수: 1.0이면 완전 탄성충돌
        ),
    )
    # translation 인자로 초기 위치 지정 (x, y, z)
    sphere_cfg.func(SPHERE_PRIM_PATH, sphere_cfg, translation=(0.0, 0.0, SPHERE_START_HEIGHT))


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """시뮬레이션 루프를 실행합니다."""

    # ── 3. SimulationContext 생성 ─────────────────────────────────────────
    # dt(물리 타임스텝): 시뮬레이션이 한 step마다 진행하는 시간 (초)
    #   - dt가 작을수록 정확하지만 계산량 증가
    #   - dt가 클수록 빠르지만 정확도 감소, 물체가 뚫고 지나갈 수 있음
    #   - 기본값: 1/60초 ≈ 0.0167초
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)

    # 카메라 시점 설정 (eye 위치, target 위치)
    sim.set_camera_view(eye=[3.0, 3.0, 2.5], target=[0.0, 0.0, 0.5])

    # ── 4. 씬 디자인 ─────────────────────────────────────────────────────
    design_scene()

    # ── 5. sim.reset() - 시뮬레이션 초기화 ────────────────────────────────
    # reset()은 물리 엔진을 초기화하고, USD 스테이지의 모든 프림을
    # PhysX 시뮬레이션에 등록합니다.
    # ※ 반드시 step() 호출 전에 한 번 실행해야 합니다!
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 시작 준비 완료")

    # ── 6. 시간 추적 변수 설정 ────────────────────────────────────────────
    # sim_dt: 물리 엔진이 한 스텝마다 진행하는 시간 (초 단위)
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0      # 누적 시뮬레이션 시간
    step_count = 0       # 총 스텝 카운터
    episode_step = 0     # 에피소드 내 스텝 카운터

    print(f"[INFO] Physics dt = {sim_dt:.6f} 초  (1 step = {sim_dt*1000:.2f} ms)")
    print(f"[INFO] 100 step마다 시간 출력, 500 step마다 에피소드 리셋")
    print("=" * 70)

    # ── 구(sphere) 위치를 읽기 위한 뷰 생성 ──────────────────────────────
    # XformPrimView를 사용하면 GPU/Fabric 백엔드에서도 효율적으로
    # 프림의 월드 좌표를 읽고 쓸 수 있습니다.
    sphere_view = sim_utils.XformPrimView(SPHERE_PRIM_PATH, device=sim.device)

    # ── 7. 시뮬레이션 루프 ────────────────────────────────────────────────
    #
    # 루프 구조:
    #   while 앱이 실행 중:
    #       1) sim.step()        → 물리 연산 + 렌더링 1프레임 진행
    #       2) sim_time += dt    → 시뮬레이션 시간 누적
    #       3) 상태 읽기          → 구의 위치 등을 읽음
    #       4) 조건 확인          → N 스텝마다 에피소드 리셋
    #
    while simulation_app.is_running():
        # 시뮬레이션이 정지되었으면 루프 종료
        if sim.is_stopped():
            break
        # 시뮬레이션이 일시정지 상태면 렌더링만 수행
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 물리 스텝 실행 ────────────────────────────────────────────
        # sim.step()은 내부적으로:
        #   a) PhysX가 dt만큼 물리 연산 수행 (힘, 충돌, 중력 등)
        #   b) 렌더링 업데이트 (render=True일 때)
        sim.step()

        # ── (2) 시간 업데이트 ─────────────────────────────────────────────
        sim_time += sim_dt
        step_count += 1
        episode_step += 1

        # ── (3) 상태 읽기 - 구의 현재 위치 ───────────────────────────────
        # get_world_poses()는 (positions, orientations)를 반환합니다.
        #   positions  : shape (N, 3) - [x, y, z]
        #   orientations: shape (N, 4) - [w, x, y, z] (쿼터니언)
        positions, orientations = sphere_view.get_world_poses()
        sphere_pos = positions[0]  # 첫 번째(유일한) 구의 위치

        # ── (4) 100 step마다 상태 출력 ────────────────────────────────────
        if episode_step % 100 == 0:
            print(
                f"  [Step {step_count:5d}] "
                f"sim_time = {sim_time:7.3f}s | "
                f"episode_step = {episode_step:4d} | "
                f"sphere z = {sphere_pos[2].item():6.3f}m"
            )

        # ── (5) 500 step마다 에피소드 리셋 ────────────────────────────────
        # 에피소드(episode): RL에서 한 번의 시도 단위
        # 일정 스텝 후 초기 상태로 되돌려 새로운 에피소드를 시작합니다.
        if episode_step >= 500:
            print(f"\n{'='*70}")
            print(f"  [에피소드 리셋] step_count={step_count}, sim_time={sim_time:.3f}s")
            print(f"  리셋 전 구 위치: z = {sphere_pos[2].item():.4f}m")

            # 구의 위치를 초기 높이로 복원
            # set_world_poses()로 위치와 방향을 직접 설정할 수 있습니다.
            reset_pos = torch.tensor([[0.0, 0.0, SPHERE_START_HEIGHT]], device=sim.device)
            reset_ori = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=sim.device)  # 단위 쿼터니언
            sphere_view.set_world_poses(positions=reset_pos, orientations=reset_ori)

            # 에피소드 카운터만 리셋 (전체 시뮬레이션 시간은 유지)
            episode_step = 0
            print(f"  리셋 후 구 위치: z = {SPHERE_START_HEIGHT:.4f}m")
            print(f"{'='*70}\n")

    # ── 8. 종료 ───────────────────────────────────────────────────────────
    simulation_app.close()
    print(f"\n[DONE] 시뮬레이션 종료 - 총 {step_count} steps, {sim_time:.3f}초 경과")


if __name__ == "__main__":
    main()
