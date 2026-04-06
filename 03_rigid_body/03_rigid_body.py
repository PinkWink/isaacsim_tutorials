# 03_rigid_body.py
# IsaacLab 강체(Rigid Body) 물리 시뮬레이션 튜토리얼
#
# 이 스크립트는 IsaacLab에서 강체 물리를 설정하는 방법을 보여줍니다.
# Visual-only 오브젝트와 Rigid Body 오브젝트의 차이를 비교합니다.
#
# 실행 방법:
#   source env_isaaclab/bin/activate
#   python 03_rigid_body.py
#   python 03_rigid_body.py --headless  (GUI 없이 실행)

# ──────────────────────────────────────────────
# 1) argparse + AppLauncher (IsaacLab 표준 패턴)
#    - IsaacSim 관련 import 전에 반드시 AppLauncher를 먼저 초기화해야 합니다.
# ──────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="IsaacLab 튜토리얼: 강체(Rigid Body) 물리 시뮬레이션"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""AppLauncher 초기화 이후에 나머지 모듈을 import 합니다."""

# ──────────────────────────────────────────────
# 2) IsaacLab / USD 관련 import
# ──────────────────────────────────────────────
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


def design_scene():
    """씬(Scene)을 구성합니다.

    Ground Plane, 조명, 그리고 다양한 오브젝트를 배치합니다.
    Visual-only 오브젝트와 Rigid Body 오브젝트의 차이를 비교할 수 있도록 구성합니다.
    """

    # ──────────────────────────────────────────
    # 2-1) Ground Plane (지면)
    #   - 오브젝트가 떨어질 바닥면을 생성합니다.
    # ──────────────────────────────────────────
    cfg_ground = sim_utils.GroundPlaneCfg()
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # ──────────────────────────────────────────
    # 2-2) Distant Light (원거리 조명)
    #   - 씬 전체를 밝히는 조명을 추가합니다.
    # ──────────────────────────────────────────
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3000.0,
        color=(0.95, 0.95, 1.0),
    )
    cfg_light.func("/World/DistantLight", cfg_light, translation=(0.0, 0.0, 10.0))

    # ──────────────────────────────────────────
    # 3) Visual-Only Sphere (시각 전용 구체)
    #   - rigid_props, mass_props, collision_props가 없습니다.
    #   - 물리 엔진의 영향을 받지 않으므로 공중에 그대로 떠 있습니다.
    #   - 렌더링만 되고, 중력이나 충돌에 반응하지 않습니다.
    # ──────────────────────────────────────────
    cfg_sphere_visual = sim_utils.SphereCfg(
        radius=0.2,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.2),  # 빨간색 - "나는 Visual-only!"
        ),
    )
    # 높이 1.5m에 배치 → 물리가 없으므로 그대로 떠 있음
    cfg_sphere_visual.func(
        "/World/VisualOnlySphere",
        cfg_sphere_visual,
        translation=(-1.0, 0.0, 1.5),
    )

    # ──────────────────────────────────────────
    # 4) Rigid Body Sphere (강체 구체) - 1.0 kg
    #   - rigid_props: PhysX 강체 시뮬레이션을 활성화합니다.
    #   - mass_props: 질량을 설정합니다 (1.0 kg).
    #   - collision_props: 충돌 감지를 활성화합니다.
    #   - 이 세 가지가 모두 있어야 물리적으로 올바르게 동작합니다.
    # ──────────────────────────────────────────
    cfg_sphere_rigid = sim_utils.SphereCfg(
        radius=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.0, 1.0, 0.0),  # 초록색 - "나는 Rigid Body!"
        ),
    )
    # 높이 1.5m에 배치 → 중력에 의해 아래로 떨어짐
    cfg_sphere_rigid.func(
        "/World/RigidBodySphere",
        cfg_sphere_rigid,
        translation=(0.0, 0.0, 1.5),
    )

    # ──────────────────────────────────────────
    # 5) Rigid Body Cube (강체 큐브) - 5.0 kg (더 무거움)
    #   - 질량이 다르면 충돌 시 반응이 달라집니다.
    #   - 이 큐브는 구체보다 5배 무겁습니다.
    #   - 질량은 떨어지는 속도에는 영향을 주지 않지만(진공에서),
    #     충돌 반응과 운동량에 영향을 줍니다.
    # ──────────────────────────────────────────
    cfg_cube_rigid = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.2, 0.4, 1.0),  # 파란색 큐브
        ),
    )
    # 높이 2.0m에 배치 → 더 높은 곳에서 떨어짐
    cfg_cube_rigid.func(
        "/World/RigidBodyCube",
        cfg_cube_rigid,
        translation=(1.0, 0.0, 2.0),
    )

    # ──────────────────────────────────────────
    # 6) Rigid Body Cone (강체 원뿔) - 2.0 kg
    #   - 다양한 형상도 강체로 만들 수 있습니다.
    #   - 원뿔은 떨어진 후 넘어지는 모습을 볼 수 있습니다.
    # ──────────────────────────────────────────
    cfg_cone_rigid = sim_utils.ConeCfg(
        radius=0.2,
        height=0.4,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=2.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(1.0, 0.8, 0.0),  # 노란색 원뿔
        ),
    )
    # 높이 1.8m에 배치
    cfg_cone_rigid.func(
        "/World/RigidBodyCone",
        cfg_cone_rigid,
        translation=(2.0, 0.0, 1.8),
    )


def main():
    """메인 함수: 시뮬레이션을 설정하고 실행합니다."""

    # ──────────────────────────────────────────
    # 7) SimulationContext 생성
    #   - dt: 시뮬레이션 시간 간격 (1/60초)
    #   - 물리 엔진(PhysX)과 렌더링을 관리합니다.
    # ──────────────────────────────────────────
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)

    # 카메라 위치 설정 - 씬 전체를 볼 수 있는 위치
    sim.set_camera_view(
        eye=(5.0, 3.0, 3.0),     # 카메라 위치
        target=(0.5, 0.0, 0.5),  # 카메라가 바라보는 지점
    )

    # 씬 구성 함수 호출
    design_scene()

    # ──────────────────────────────────────────
    # 8) 시뮬레이션 초기화 (reset)
    #   - 물리 엔진을 초기화하고 첫 프레임을 준비합니다.
    #   - design_scene() 이후에 반드시 호출해야 합니다.
    # ──────────────────────────────────────────
    sim.reset()
    print("=" * 60)
    print("[INFO] 시뮬레이션이 시작되었습니다.")
    print("[INFO] 빨간 구체 (Visual-only): 공중에 떠 있음")
    print("[INFO] 초록 구체 (Rigid Body, 1kg): 떨어짐")
    print("[INFO] 파란 큐브 (Rigid Body, 5kg): 떨어짐")
    print("[INFO] 노란 원뿔 (Rigid Body, 2kg): 떨어짐")
    print("=" * 60)

    # ──────────────────────────────────────────
    # 9) 시뮬레이션 루프
    #   - simulation_app이 실행 중인 동안 계속 반복합니다.
    #   - sim.step(): 물리 엔진을 한 스텝 진행합니다.
    #     - 강체 오브젝트: 중력, 충돌 등이 계산됩니다.
    #     - Visual-only 오브젝트: 물리 계산 없이 그대로 유지됩니다.
    # ──────────────────────────────────────────
    step_count = 0
    while simulation_app.is_running():
        # 물리 시뮬레이션 한 스텝 진행
        sim.step()
        step_count += 1

        # 처음 몇 스텝에서 상태 출력 (너무 많이 출력하지 않도록)
        if step_count in (1, 30, 60, 120):
            elapsed_time = step_count * sim_cfg.dt
            print(
                f"[Step {step_count:4d}] 경과 시간: {elapsed_time:.2f}초"
            )

    print(f"[INFO] 시뮬레이션 종료. 총 {step_count} 스텝 실행됨.")


if __name__ == "__main__":
    main()
