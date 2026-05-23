"""
04_simulation_loop.py - 시뮬레이션 루프 + 탄성 비교 실험

IsaacLab 시뮬레이션의 핵심 루프 구조를 배우면서, 동시에 세 개의 공을
서로 다른 탄성 계수(restitution) 로 떨어뜨려 차이를 관찰합니다.

  - sim.reset()으로 초기화
  - sim.step()으로 물리 + 렌더링 진행
  - dt(시간 간격)를 이용한 시뮬레이션 시간 추적
  - 에피소드 리셋 (주기적으로 초기 상태 복원)

세 공의 구성:
  - 🔴 빨간공  (x = -1.0): restitution = 0.2  → 거의 안 튐
  - 🟡 노란공  (x =  0.0): restitution = 0.5  → 보통 정도 튐
  - 🔵 파란공  (x = +1.0): restitution = 0.9  → 잘 튐

실행:
    source env_isaaclab/bin/activate
    python 04_simulation_loop.py

    # GUI 없이 실행
    python 04_simulation_loop.py --headless
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="04 - 시뮬레이션 루프 + 탄성 비교 실험")
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
# 실험 파라미터
# ══════════════════════════════════════════════════════════════════════════

SPHERE_START_HEIGHT = 2.0   # 떨어뜨리는 높이 (m)
SPHERE_RADIUS = 0.15

# 세 공의 설정 — (이름, prim 경로, x 좌표, restitution, 색상 RGB)
#   restitution: 0.0 ~ 1.0. 클수록 잘 튑니다.
#   바닥의 restitution=1.0 이므로, 실제 충돌 시 반발은 공의 값에 가깝게 동작합니다.
BALLS = [
    ("빨간공", "/World/Ball_red",    -1.0, 0.2, (1.0, 0.2, 0.2)),
    ("노란공", "/World/Ball_yellow",  0.0, 0.5, (1.0, 0.9, 0.1)),
    ("파란공", "/World/Ball_blue",   +1.0, 0.9, (0.2, 0.5, 1.0)),
]

# 공 prim 경로를 한 번에 잡기 위한 정규식 (XformPrimView 가 받음)
BALL_PATTERN = "/World/Ball_.*"


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성 함수
# ══════════════════════════════════════════════════════════════════════════

def design_scene() -> None:
    """씬에 바닥, 조명, 세 개의 강체 구를 배치합니다."""

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # 바닥의 restitution=1.0 (완전 탄성) 이라서 충돌 시 반발은 공의 값이 그대로 드러납니다.
    # (PhysX 의 기본 combine mode 는 average — 바닥이 1.0이면 공 값에 가깝게 보정됨)
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(restitution=1.0),
    )
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # ── 원거리 조명 ──────────────────────────────────────────────────────
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    # ── 세 개의 강체 구 ─────────────────────────────────────────────────
    # 같은 높이(z=SPHERE_START_HEIGHT)에 다른 x 좌표로 배치 → 동시에 낙하
    # 각 공은 다른 restitution → 바운스 양상이 다르게 보임
    for name, prim_path, x_pos, restitution, color in BALLS:
        sphere_cfg = sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=restitution,
            ),
        )
        sphere_cfg.func(
            prim_path,
            sphere_cfg,
            translation=(x_pos, 0.0, SPHERE_START_HEIGHT),
        )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """시뮬레이션 루프를 실행합니다."""

    # ── 3. SimulationContext 생성 ─────────────────────────────────────────
    # dt(물리 타임스텝): 시뮬레이션이 한 step마다 진행하는 시간 (초)
    #   - dt가 작을수록 정확하지만 계산량 증가
    #   - 기본값: 1/60초 ≈ 0.0167초
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(cfg=sim_cfg)

    # 카메라 시점 — 세 공을 한 화면에 담도록 약간 멀리 배치
    sim.set_camera_view(eye=[4.0, 4.0, 2.5], target=[0.0, 0.0, 0.5])

    # ── 4. 씬 디자인 ─────────────────────────────────────────────────────
    design_scene()

    # ── 5. sim.reset() ───────────────────────────────────────────────────
    # reset()은 물리 엔진을 초기화하고, USD 스테이지의 모든 프림을
    # PhysX 시뮬레이션에 등록합니다. step() 전에 반드시 호출해야 합니다.
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 시작 준비 완료")

    # ── 6. 시간 추적 변수 ────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    episode_step = 0

    print(f"[INFO] Physics dt = {sim_dt:.6f} 초  (1 step = {sim_dt*1000:.2f} ms)")
    print(f"[INFO] 100 step마다 z 좌표 출력, 500 step마다 에피소드 리셋")
    print("[INFO] 실험 구성:")
    for name, _, x_pos, restitution, _ in BALLS:
        print(f"        {name}  x={x_pos:+.1f}  restitution={restitution:.2f}")
    print("=" * 78)

    # ── 세 공을 한 번에 다루는 view ─────────────────────────────────────
    # 정규식으로 매칭되는 모든 prim 을 한 view 로 관리합니다 → (3, 3) shape.
    balls_view = sim_utils.XformPrimView(BALL_PATTERN, device=sim.device)

    # 리셋 시 사용할 초기 위치/방향 — 한 번만 만들어두고 재사용
    reset_positions = torch.tensor(
        [[x, 0.0, SPHERE_START_HEIGHT] for _, _, x, _, _ in BALLS],
        device=sim.device,
    )
    reset_orientations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * len(BALLS),  # 단위 쿼터니언
        device=sim.device,
    )

    # ── 7. 시뮬레이션 루프 ────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션이 정지되었으면 루프 종료
        if sim.is_stopped():
            break
        # 일시정지 상태면 렌더링만 수행
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 물리 스텝 ────────────────────────────────────────────────
        sim.step()

        # ── (2) 시간 업데이트 ────────────────────────────────────────────
        sim_time += sim_dt
        step_count += 1
        episode_step += 1

        # ── (3) 세 공의 현재 위치 읽기 ───────────────────────────────────
        # positions shape: (3, 3) — [ball_idx, xyz]
        positions, _ = balls_view.get_world_poses()

        # ── (4) 100 step마다 z 좌표 출력 ─────────────────────────────────
        if episode_step % 100 == 0:
            z_values = positions[:, 2].cpu().numpy()
            z_str = " | ".join(
                f"{name} z={z:.3f}m"
                for (name, *_), z in zip(BALLS, z_values)
            )
            print(
                f"  [Step {step_count:5d}] "
                f"t={sim_time:6.3f}s  ep={episode_step:4d}  {z_str}"
            )

        # ── (5) 500 step마다 에피소드 리셋 ───────────────────────────────
        # 모든 공을 동시에 초기 위치로 되돌립니다.
        if episode_step >= 500:
            print(f"\n{'='*78}")
            print(f"  [에피소드 리셋] step_count={step_count}, sim_time={sim_time:.3f}s")
            for (name, *_), z in zip(BALLS, positions[:, 2].cpu().numpy()):
                print(f"    리셋 전 {name} z = {z:.4f}m")

            balls_view.set_world_poses(
                positions=reset_positions, orientations=reset_orientations
            )

            episode_step = 0
            print(f"    모든 공을 z = {SPHERE_START_HEIGHT:.4f}m 로 복원")
            print(f"{'='*78}\n")

    # ── 8. 종료 ───────────────────────────────────────────────────────────
    simulation_app.close()
    print(f"\n[DONE] 시뮬레이션 종료 - 총 {step_count} steps, {sim_time:.3f}초 경과")


if __name__ == "__main__":
    main()
