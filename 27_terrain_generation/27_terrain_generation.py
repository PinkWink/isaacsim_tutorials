"""
27_terrain_generation.py - 다양한 지형 위의 ANYmal 보행 로봇

IsaacLab의 절차적 지형 생성(Procedural Terrain Generation) API로
다양한 지형(평탄, 계단, 랜덤 그리드, 파도)을 생성하고,
ANYmal-C 4족 보행 로봇을 배치하여 지형별 동작 차이를 관찰합니다:
  - TerrainGeneratorCfg: 전체 지형 그리드 설정 (크기, 행/열, 커리큘럼)
  - TerrainImporterCfg: 생성된 지형을 씬에 임포트
  - Sub-terrain 유형: Flat, PyramidStairs, RandomGrid, Wave
  - ANYmal-C를 지형 위에 배치하여 보행 가능성을 시각적으로 확인

실행:
    source env_isaaclab/bin/activate
    cd lectures/27_terrain_generation
    python 27_terrain_generation.py

    # GUI 없이 실행
    python 27_terrain_generation.py --headless

    # 환경(로봇) 수 변경 (기본: 16)
    python 27_terrain_generation.py --num_envs 64
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="27 - 지형 생성 + ANYmal 튜토리얼")
parser.add_argument("--num_envs", type=int, default=16, help="생성할 환경(로봇) 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ── 3. 지형 생성 관련 import ─────────────────────────────────────────────
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import (
    HfWaveTerrainCfg,
)

# ── 4. ANYmal-C 로봇 설정 ───────────────────────────────────────────────
# 4족 보행 로봇: 계단, 울퉁불퉁한 지형 등에서 보행 능력을 관찰하기에 적합
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  지형 생성 설정 (TerrainGeneratorCfg)
# ══════════════════════════════════════════════════════════════════════════

TUTORIAL_TERRAIN_CFG = TerrainGeneratorCfg(
    # ── 기본 그리드 설정 ────────────────────────────────────────────────
    size=(8.0, 8.0),            # 각 서브 지형 크기: 8m × 8m
    num_rows=4,                 # 4행 (난이도 레벨 4단계)
    num_cols=4,                 # 4열 (각 지형 유형이 배치됨)
    border_width=5.0,           # 전체 지형 주변 테두리 너비 (m)

    # ── 커리큘럼 설정 ───────────────────────────────────────────────────
    # curriculum=True: 행(row)이 증가할수록 난이도가 올라갑니다.
    # 예) row 0: difficulty=0.0 (쉬움), row 3: difficulty=1.0 (어려움)
    curriculum=True,
    difficulty_range=(0.0, 1.0),

    # ── 해상도 설정 (height_field 지형에 적용) ─────────────────────────
    horizontal_scale=0.1,       # X/Y 축 해상도: 0.1m (10cm 격자)
    vertical_scale=0.005,       # Z 축 해상도: 0.005m (5mm 단위)
    slope_threshold=0.75,       # 이 값 이상의 경사면은 수직으로 보정

    # ── 시각화 설정 ─────────────────────────────────────────────────────
    color_scheme="height",      # 높이에 따라 색상 표시

    # ── 서브 지형 정의 ──────────────────────────────────────────────────
    sub_terrains={
        # 1) 평평한 바닥 — 기준 성능 측정, ANYmal이 정상 보행하는 모습
        "flat": MeshPlaneTerrainCfg(
            proportion=0.25,
        ),

        # 2) 피라미드 계단 — 계단 오르내리기 능력 테스트
        # step_height_range: difficulty에 따라 5cm(쉬움) ~ 20cm(어려움)
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.5,
        ),

        # 3) 랜덤 그리드 — 불규칙 지면 위 보행 안정성 테스트
        "random_grid": MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.02, 0.15),
            platform_width=2.0,
        ),

        # 4) 파도 지형 — 연속적인 경사/기복 위 보행 테스트
        "wave": HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.02, 0.15),
            num_waves=3,
            border_width=0.25,
        ),
    },
)


# ══════════════════════════════════════════════════════════════════════════
#  씬 설정 (InteractiveSceneCfg)
# ══════════════════════════════════════════════════════════════════════════

@configclass
class TerrainDemoSceneCfg(InteractiveSceneCfg):
    """지형 생성 + ANYmal-C 데모 씬.

    ground 대신 terrain을 사용하여 절차적 지형을 배치합니다.
    ANYmal-C 로봇이 각 지형 위에 배치되어 보행 가능성을 시각적으로 확인합니다.
    """

    # ── 지형 (TerrainImporterCfg) ────────────────────────────────────────
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TUTORIAL_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.3, 0.3),
        ),
        debug_vis=True,
    )

    # ── ANYmal-C 4족 보행 로봇 ───────────────────────────────────────────
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )

    # ── 조명 ─────────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            color=(0.9, 0.9, 0.9),
            intensity=1000.0,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """다양한 지형 위에 ANYmal-C를 배치하고 시뮬레이션합니다."""

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(30.0, 30.0, 25.0), target=(0.0, 0.0, 0.0))

    scene_cfg = TerrainDemoSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=8.0,
    )
    scene = InteractiveScene(scene_cfg)

    # ── 지형 정보 출력 ───────────────────────────────────────────────────
    terrain_cfg = TUTORIAL_TERRAIN_CFG
    total_size_x = terrain_cfg.size[0] * terrain_cfg.num_rows + 2 * terrain_cfg.border_width
    total_size_y = terrain_cfg.size[1] * terrain_cfg.num_cols + 2 * terrain_cfg.border_width
    total_cells = terrain_cfg.num_rows * terrain_cfg.num_cols

    print("=" * 70)
    print("[INFO] 지형 생성 완료!")
    print(f"  서브 지형 크기:   {terrain_cfg.size[0]}m x {terrain_cfg.size[1]}m")
    print(f"  그리드 배열:      {terrain_cfg.num_rows}행 x {terrain_cfg.num_cols}열 = {total_cells}셀")
    print(f"  전체 지형 크기:   {total_size_x:.1f}m x {total_size_y:.1f}m")
    print(f"  커리큘럼 모드:    {terrain_cfg.curriculum}")
    print(f"  로봇:             ANYmal-C (4족 보행)")
    print(f"  환경(로봇) 수:    {args_cli.num_envs}")
    print("  서브 지형 목록:")
    for name, sub_cfg in terrain_cfg.sub_terrains.items():
        print(f"    - {name:20s} (비율: {sub_cfg.proportion:.0%}, 클래스: {type(sub_cfg).__name__})")
    print("=" * 70)
    print("\n[INFO] 관찰 포인트:")
    print("  - 평탄 지형: ANYmal이 안정적으로 서 있는 모습")
    print("  - 계단 지형: 높은 계단에서 균형을 잃는 모습 (학습 안 된 상태)")
    print("  - 랜덤 그리드: 불규칙 지면에서의 발 접촉 변화")
    print("  - 파도 지형: 경사면에서 미끄러지는 현상")
    print("  → 26강에서 학습한 보행 정책을 적용하면 이 지형들을 극복할 수 있습니다!")

    # ── 시뮬레이션 시작 ──────────────────────────────────────────────────
    sim.reset()

    step_count = 0
    sim_dt = sim_cfg.dt

    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt=sim_dt)
        step_count += 1

        # 주기적 상태 출력
        if step_count % 500 == 0:
            robot = scene["robot"]
            root_pos = robot.data.root_pos_w
            heights = root_pos[:, 2]
            print(
                f"[Step {step_count:5d}] "
                f"로봇 높이 — 평균: {heights.mean().item():.3f}m, "
                f"최소: {heights.min().item():.3f}m, "
                f"최대: {heights.max().item():.3f}m | "
                f"로봇 수: {root_pos.shape[0]}"
            )

        if step_count >= 3000:
            print(f"\n[INFO] {step_count} 스텝 완료, 시뮬레이션 종료")
            break


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 지형 생성 데모 종료")
