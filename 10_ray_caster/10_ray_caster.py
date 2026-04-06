"""
10_ray_caster.py - RayCaster 센서로 거리 측정 (LiDAR 유사)

RayCaster는 지정된 패턴으로 광선(ray)을 쏘아 메시와의 교차점을 계산하는 센서입니다.
실제 로봇의 LiDAR, 높이 맵(height map) 측정 등에 사용됩니다:
  - RayCasterCfg로 센서 설정 (패턴, 대상 메시, 시각화)
  - GridPatternCfg로 격자 형태의 광선 패턴 정의
  - ray_hits_w 데이터로 광선이 맞은 월드 좌표 획득
  - 여러 환경에서 동시에 동작하는 병렬 ray casting

실행:
    source env_isaaclab/bin/activate
    cd lectures/10_ray_caster
    python 10_ray_caster.py --num_envs 2

    # GUI 없이 실행
    python 10_ray_caster.py --headless --num_envs 2
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="10 - RayCaster 센서 튜토리얼")
parser.add_argument("--num_envs", type=int, default=2, help="생성할 환경(env) 개수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""AppLauncher 초기화 이후에 나머지 모듈을 import 합니다."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.utils.math import random_orientation


# ══════════════════════════════════════════════════════════════════════════
# 씬 설계
# ══════════════════════════════════════════════════════════════════════════


def design_scene(num_envs: int) -> dict:
    """씬을 구성합니다.

    GroundPlane, 조명, 그리고 여러 환경에 강체 공(ball)을 배치합니다.
    RayCaster는 각 공에 부착되어, 공 위치에서 아래 방향으로 광선을 쏩니다.

    Args:
        num_envs: 생성할 환경 개수.

    Returns:
        origins: 각 환경의 원점 좌표 텐서.
    """

    # ── 2-1) Ground Plane (지면) ──────────────────────────────────────────
    # RayCaster가 광선을 쏘아 교차점을 찾을 대상 메시입니다.
    # mesh_prim_paths에서 이 경로를 지정합니다.
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/defaultGroundPlane", cfg_ground)

    # ── 2-2) Distant Light (원거리 조명) ──────────────────────────────────
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.95, 0.95, 1.0),
    )
    cfg_light.func("/World/DistantLight", cfg_light, translation=(0.0, 0.0, 10.0))

    # ── 2-3) 환경 원점(origin) 생성 ──────────────────────────────────────
    # 각 환경을 일정 간격으로 배치합니다.
    # 예: num_envs=2이면 (0,0,0), (3,0,0)
    env_spacing = 3.0
    origins = []
    for i in range(num_envs):
        origin = (i * env_spacing, 0.0, 0.0)
        origins.append(origin)

        # ── 2-4) 각 환경에 Origin Xform 생성 ─────────────────────────────
        # RayCaster의 prim_path 와일드카드(/World/Origin.*/ball)가 매칭될 수 있도록
        # /World/Origin0, /World/Origin1, ... 프림을 생성합니다.
        prim_path = f"/World/Origin{i}"
        prim_utils = sim_utils  # sim_utils에 spawn_xform 등 없으므로 USD 직접 사용
        # Xform 프림 생성
        import isaacsim.core.utils.prims as prim_utils_core

        prim_utils_core.create_prim(prim_path, "Xform", translation=origin)

        # ── 2-5) 강체 공(ball) 생성 ───────────────────────────────────────
        # RayCaster가 부착될 강체입니다.
        # RayCaster는 이 공의 위치를 추적하며 광선을 발사합니다.
        cfg_ball = sim_utils.SphereCfg(
            radius=0.15,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,  # 중력 비활성화 (공중에 유지)
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.6, 1.0),  # 파란색 공
            ),
        )
        cfg_ball.func(
            f"{prim_path}/ball",
            cfg_ball,
            translation=(0.0, 0.0, 2.0),  # origin 기준 상대 좌표 (높이 2m)
        )

    # origins를 텐서로 변환
    origins_tensor = torch.tensor(origins, dtype=torch.float32)

    return origins_tensor


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(
    sim: SimulationContext,
    ray_caster: RayCaster,
    num_envs: int,
    origins: torch.Tensor,
) -> None:
    """RayCaster 센서를 사용한 시뮬레이션 루프를 실행합니다.

    Args:
        sim: SimulationContext 인스턴스.
        ray_caster: RayCaster 센서 인스턴스.
        num_envs: 환경 개수.
        origins: 각 환경의 원점 좌표.
    """
    sim_dt = sim.get_physics_dt()
    step_count = 0

    # ── RayCaster 기본 정보 출력 ──────────────────────────────────────────
    # RayCaster 센서가 생성한 광선(ray) 정보를 확인합니다.
    print("=" * 70)
    print("[RayCaster 센서 정보]")
    print(f"  광선(ray) 개수: {ray_caster.num_rays}")
    print(f"  센서 개수 (환경 개수): {num_envs}")
    print(f"  ray_hits_w shape: ({num_envs}, {ray_caster.num_rays}, 3)")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션이 정지되었으면 루프 종료
        if sim.is_stopped():
            break
        # 일시정지 상태면 렌더링만 수행
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 250 step마다 공 위치 랜덤 리셋 ───────────────────────────
        # 공의 높이를 랜덤으로 변경하여 RayCaster의 거리 측정 변화를 관찰합니다.
        if step_count % 250 == 0:
            print(f"\n[INFO] 공 위치 랜덤 리셋 (step={step_count})")

            # 각 환경의 공을 랜덤 높이(1.0 ~ 3.0m)에 배치
            for i in range(num_envs):
                prim_path = f"/World/Origin{i}/ball"
                # 랜덤 높이 생성
                random_height = 1.0 + torch.rand(1).item() * 2.0  # 1.0 ~ 3.0m
                random_x_offset = (torch.rand(1).item() - 0.5) * 1.0  # -0.5 ~ 0.5m
                random_y_offset = (torch.rand(1).item() - 0.5) * 1.0

                new_pos = (
                    origins[i, 0].item() + random_x_offset,
                    origins[i, 1].item() + random_y_offset,
                    random_height,
                )

                # USD를 통해 공 위치 업데이트
                from pxr import UsdGeom

                stage = sim_utils.get_current_stage()
                prim = stage.GetPrimAtPath(prim_path)
                if prim.IsValid():
                    xform = UsdGeom.Xformable(prim)
                    xform.ClearXformOpOrder()
                    xform.AddTranslateOp().Set(new_pos)

                print(f"  env {i}: ball 위치 = ({new_pos[0]:.2f}, {new_pos[1]:.2f}, {new_pos[2]:.2f})")

        # ── (2) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()
        step_count += 1

        # ── (3) RayCaster 업데이트 ────────────────────────────────────────
        # RayCaster의 내부 버퍼를 최신 물리 상태로 갱신합니다.
        # force_recompute=True로 매 스텝마다 광선을 다시 계산합니다.
        ray_caster.update(dt=sim_dt, force_recompute=True)

        # ── (4) 100 step마다 RayCaster 데이터 출력 ────────────────────────
        if step_count % 100 == 0:
            # ray_hits_w: 광선이 메시와 교차한 월드 좌표
            # shape: (num_envs, num_rays, 3) - [x, y, z]
            ray_hits = ray_caster.data.ray_hits_w

            print(f"\n[Step {step_count}] RayCaster 데이터:")
            print(f"  ray_hits_w shape: {ray_hits.shape}")

            for env_idx in range(num_envs):
                hits = ray_hits[env_idx]  # shape: (num_rays, 3)

                # 유효한 히트만 필터링 (z > -1e5: 유효하지 않은 히트는 매우 큰 음수)
                valid_mask = hits[:, 2] > -1e4
                valid_hits = hits[valid_mask]

                if valid_hits.numel() > 0:
                    # 히트 포인트의 z 좌표 통계
                    max_z = valid_hits[:, 2].max().item()
                    min_z = valid_hits[:, 2].min().item()
                    mean_z = valid_hits[:, 2].mean().item()

                    print(f"  env {env_idx}: 유효 히트 {valid_hits.shape[0]}개, "
                          f"z범위=[{min_z:.3f}, {max_z:.3f}], z평균={mean_z:.3f}")
                else:
                    print(f"  env {env_idx}: 유효 히트 없음 (광선이 메시에 도달하지 못함)")

            # 센서 위치 출력
            print(f"  센서 위치 (pos_w): {ray_caster.data.pos_w}")
            print(f"  총 광선 수: {ray_caster.num_rays}")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """RayCaster 센서 시뮬레이션을 설정하고 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점: 환경 전체가 보이도록 설정
    sim.set_camera_view(eye=[5.0, 5.0, 5.0], target=[1.0, 0.0, 1.0])

    # ── 씬 구성 ──────────────────────────────────────────────────────────
    num_envs = args_cli.num_envs
    origins = design_scene(num_envs)

    # ── RayCaster 센서 생성 ──────────────────────────────────────────────
    # RayCasterCfg: 광선 센서의 설정을 정의합니다.
    #
    # - prim_path: 센서가 부착될 프림 경로.
    #   "/World/Origin.*/ball"은 모든 환경의 ball을 매칭합니다.
    #   Origin0/ball, Origin1/ball, ... 에 각각 센서가 부착됩니다.
    #
    # - mesh_prim_paths: 광선이 교차할 대상 메시 목록.
    #   GroundPlane 메시를 지정하여 바닥면과의 교차점을 계산합니다.
    #
    # - pattern_cfg: 광선 패턴 설정.
    #   GridPatternCfg는 2D 격자 형태의 광선 패턴을 생성합니다.
    #   resolution=0.1m 간격, size=(2.0, 2.0)m 범위의 격자.
    #   각 격자 점에서 아래 방향(0,0,-1)으로 광선을 발사합니다.
    #
    # - debug_vis: True이면 광선 히트 포인트를 시각적으로 표시합니다.
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Origin.*/ball",
        mesh_prim_paths=["/World/defaultGroundPlane"],
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.1,
            size=(2.0, 2.0),
        ),
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    # ── sim.reset() ───────────────────────────────────────────────────────
    # 씬 생성 후 반드시 호출하여 물리 엔진을 초기화합니다.
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 준비 완료")
    print(f"[INFO] 환경 개수: {num_envs}")
    print(f"[INFO] 환경 원점: {origins}")

    # ── 시뮬레이션 실행 ───────────────────────────────────────────────────
    run_simulator(sim, ray_caster, num_envs, origins)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
