"""
02_spawn_primitives.py - 기본 도형(Primitive) 스포닝 튜토리얼

이 스크립트는 IsaacLab을 사용하여 USD Stage 위에 다양한 기본 도형을
스폰(생성)하는 방법을 보여줍니다.

사용법:
    # GUI 모드로 실행
    python 02_spawn_primitives.py

    # Headless 모드로 실행 (GUI 없이)
    python 02_spawn_primitives.py --headless

    # 특정 GPU 디바이스 지정
    python 02_spawn_primitives.py --device cuda:0
"""

# ──────────────────────────────────────────────────────────────
# 1단계: argparse + AppLauncher 설정
#   - IsaacLab에서는 반드시 AppLauncher를 통해 시뮬레이터를 실행해야 합니다.
#   - AppLauncher.add_app_launcher_args()는 --headless, --device 등
#     IsaacSim에 필요한 공통 인자를 자동으로 추가해 줍니다.
#   - 주의: AppLauncher 생성 전에 다른 isaaclab/omni 모듈을 import하면 안 됩니다.
# ──────────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

# argparse 파서 생성 및 IsaacLab 공통 인자 추가
parser = argparse.ArgumentParser(description="Tutorial 02 - 기본 도형(Primitive) 스포닝")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# AppLauncher 인스턴스 생성 → 내부적으로 SimulationApp이 초기화됩니다.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

# ──────────────────────────────────────────────────────────────
# 2단계: IsaacLab 모듈 임포트
#   - AppLauncher 생성 이후에야 isaaclab.sim 등의 모듈을 import할 수 있습니다.
#   - 이 순서를 지키지 않으면 USD 런타임 에러가 발생합니다.
# ──────────────────────────────────────────────────────────────
import isaaclab.sim as sim_utils


def design_scene():
    """씬(Scene) 구성 함수

    시뮬레이션 공간에 바닥, 조명, 그리고 다양한 기본 도형들을 배치합니다.
    각 도형은 서로 다른 색상의 PreviewSurface 머티리얼을 적용하여
    시각적으로 구분할 수 있도록 합니다.
    """

    # ──────────────────────────────────────────────────────────
    # 3단계: Ground Plane 추가
    #   - GroundPlaneCfg를 사용하여 무한 평면 바닥을 생성합니다.
    #   - USD 경로 "/World/defaultGroundPlane"에 프림(Prim)이 생성됩니다.
    # ──────────────────────────────────────────────────────────
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    print("[INFO] Ground Plane 생성 완료")

    # ──────────────────────────────────────────────────────────
    # 4단계: Distant Light 추가
    #   - DistantLightCfg로 태양광과 유사한 조명을 추가합니다.
    #   - intensity: 빛의 강도 (3000.0 정도면 적당)
    #   - color: RGB 값 (0~1 범위), 여기서는 따뜻한 흰색
    # ──────────────────────────────────────────────────────────
    light_cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    print("[INFO] Distant Light 생성 완료")

    # ──────────────────────────────────────────────────────────
    # 5단계: Xform 프림 생성 (그룹 컨테이너)
    #   - Xform은 USD에서 트랜스폼(위치/회전/스케일)을 가지는 빈 그룹 노드입니다.
    #   - 여러 도형을 하나의 Xform 아래에 묶으면 계층적으로 관리할 수 있습니다.
    #   - "/World/Objects" 아래에 모든 도형을 배치합니다.
    # ──────────────────────────────────────────────────────────
    sim_utils.create_prim("/World/Objects", "Xform")
    print("[INFO] Xform 그룹 '/World/Objects' 생성 완료")

    # ──────────────────────────────────────────────────────────
    # 6단계: 빨간색 원뿔 (Red Cone) 스폰
    #   - ConeCfg: 원뿔 형상 설정
    #     - radius: 밑면 반지름 (0.15m)
    #     - height: 높이 (0.5m)
    #   - PreviewSurfaceCfg: USD 기본 머티리얼 (색상, 반사도 등)
    #     - diffuse_color: 확산 반사 색상 (R, G, B)
    #   - 위치: 원점 왼쪽 (-1.0, 0.0, 0.25)
    # ──────────────────────────────────────────────────────────
    cone_cfg = sim_utils.ConeCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    )
    cone_cfg.func("/World/Objects/Cone", cone_cfg, translation=(-1.0, 0.0, 0.25))
    print("[INFO] 빨간색 원뿔(Cone) 생성 완료 - 위치: (-1.0, 0.0, 0.25)")

    # ──────────────────────────────────────────────────────────
    # 7단계: 파란색 정육면체 (Blue Cube) 스폰
    #   - CuboidCfg: 직육면체 형상 설정
    #     - size: (가로, 세로, 높이) 크기 (0.3m x 0.3m x 0.3m)
    #   - 위치: 원점 앞쪽 (-0.33, 0.0, 0.15)
    # ──────────────────────────────────────────────────────────
    cube_cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
    )
    cube_cfg.func("/World/Objects/Cube", cube_cfg, translation=(-0.33, 0.0, 0.15))
    print("[INFO] 파란색 정육면체(Cube) 생성 완료 - 위치: (-0.33, 0.0, 0.15)")

    # ──────────────────────────────────────────────────────────
    # 8단계: 초록색 구 (Green Sphere) 스폰
    #   - SphereCfg: 구 형상 설정
    #     - radius: 반지름 (0.2m)
    #   - 위치: 원점 오른쪽 (0.33, 0.0, 0.2)
    # ──────────────────────────────────────────────────────────
    sphere_cfg = sim_utils.SphereCfg(
        radius=0.2,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
    )
    sphere_cfg.func("/World/Objects/Sphere", sphere_cfg, translation=(0.33, 0.0, 0.2))
    print("[INFO] 초록색 구(Sphere) 생성 완료 - 위치: (0.33, 0.0, 0.2)")

    # ──────────────────────────────────────────────────────────
    # 9단계: 노란색 원기둥 (Yellow Cylinder) 스폰
    #   - CylinderCfg: 원기둥 형상 설정
    #     - radius: 반지름 (0.15m)
    #     - height: 높이 (0.5m)
    #   - 위치: 원점 오른쪽 (1.0, 0.0, 0.25)
    # ──────────────────────────────────────────────────────────
    cylinder_cfg = sim_utils.CylinderCfg(
        radius=0.15,
        height=0.5,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
    )
    cylinder_cfg.func("/World/Objects/Cylinder", cylinder_cfg, translation=(1.0, 0.0, 0.25))
    print("[INFO] 노란색 원기둥(Cylinder) 생성 완료 - 위치: (1.0, 0.0, 0.25)")


def main():
    """메인 함수: 시뮬레이션 설정, 씬 구성, 시뮬레이션 루프 실행"""

    # ──────────────────────────────────────────────────────────
    # 10단계: SimulationContext 생성
    #   - SimulationCfg: 시뮬레이션의 물리 설정을 담는 설정 객체
    #     - dt: 시뮬레이션 타임스텝 (0.01초 = 100Hz)
    #     - device: 연산 디바이스 ("cuda:0" 또는 "cpu")
    #   - SimulationContext: PhysX 물리 엔진과의 인터페이스
    # ──────────────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    print(f"[INFO] SimulationContext 생성 완료 (dt={sim_cfg.dt}, device={sim_cfg.device})")

    # ──────────────────────────────────────────────────────────
    # 11단계: 카메라 뷰 설정
    #   - eye: 카메라 위치 - 도형들을 비스듬히 내려다보는 각도
    #   - target: 카메라가 바라보는 지점 - 도형들의 중심 부근
    # ──────────────────────────────────────────────────────────
    sim.set_camera_view(eye=[2.0, 0.0, 2.5], target=[-0.5, 0.0, 0.5])
    print("[INFO] 카메라 뷰 설정 완료")

    # ──────────────────────────────────────────────────────────
    # 12단계: 씬 구성
    #   - 시뮬레이션을 시작(reset)하기 전에 모든 오브젝트를 배치해야 합니다.
    # ──────────────────────────────────────────────────────────
    design_scene()

    # ──────────────────────────────────────────────────────────
    # 13단계: 시뮬레이션 초기화 (reset)
    #   - sim.reset()은 물리 엔진을 초기화하고 시뮬레이션을 시작 가능 상태로 만듭니다.
    #   - 이 호출 이후부터 sim.step()으로 시뮬레이션을 진행할 수 있습니다.
    # ──────────────────────────────────────────────────────────
    sim.reset()
    print("[INFO] 시뮬레이션 초기화(reset) 완료")
    print("[INFO] 시뮬레이션 루프를 시작합니다...")
    print("=" * 60)

    # ──────────────────────────────────────────────────────────
    # 14단계: 시뮬레이션 루프
    #   - simulation_app.is_running(): 시뮬레이터가 활성 상태인지 확인
    #     (GUI 창을 닫거나 Ctrl+C 시 False 반환)
    #   - sim.step(): 한 타임스텝(dt)만큼 물리 시뮬레이션을 진행
    #   - 도형들은 물리 없는 시각적 오브젝트이므로 정지 상태를 유지합니다.
    # ──────────────────────────────────────────────────────────
    step_count = 0

    while simulation_app.is_running():
        # 물리 시뮬레이션 1스텝 진행
        sim.step()
        step_count += 1

        # 500 스텝마다 시뮬레이션 시간 출력
        if step_count % 500 == 0:
            sim_time = sim.current_time
            print(f"  [Step {step_count:>6d}] 시뮬레이션 시간: {sim_time:.2f}초")

    # ──────────────────────────────────────────────────────────
    # 루프 종료 후 안내 메시지
    # ──────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"[INFO] 시뮬레이션 종료 (총 {step_count} 스텝 실행)")


if __name__ == "__main__":
    try:
        main()
    finally:
        # ──────────────────────────────────────────────────────
        # 15단계: 시뮬레이터 종료
        #   - simulation_app.close()를 반드시 호출하여 리소스를 정리합니다.
        #   - try/finally를 사용하여 예외 발생 시에도 종료를 보장합니다.
        # ──────────────────────────────────────────────────────
        simulation_app.close()
        print("[INFO] 시뮬레이터가 정상적으로 종료되었습니다.")
