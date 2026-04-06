"""
07_scene_design.py - InteractiveScene을 이용한 씬 설계

IsaacLab의 InteractiveScene 인터페이스를 사용하여 씬을 구성하는 방법을 배웁니다:
  - @configclass로 InteractiveSceneCfg 상속 정의
  - {ENV_REGEX_NS}를 이용한 다중 환경 네임스페이싱
  - GroundPlane, DomeLight, Cartpole(Articulation), Table(RigidObject) 배치
  - scene["entity_name"]으로 엔티티 접근
  - scene.write_data_to_sim() / scene.update() 패턴
  - scene.reset()으로 환경 일괄 리셋

실행:
    source env_isaaclab/bin/activate
    python 07_scene_design.py --num_envs 4

    # GUI 없이 실행
    python 07_scene_design.py --headless

    # 환경 수 변경
    python 07_scene_design.py --num_envs 16
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="07 - InteractiveScene 씬 설계 튜토리얼")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# isaaclab_assets 패키지에 미리 정의된 ArticulationCfg를 사용합니다.
from isaaclab_assets import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """InteractiveScene에 등록할 모든 엔티티를 선언합니다.

    @configclass 데코레이터를 사용하여 dataclass 기반으로 정의합니다.
    각 속성(attribute)이 씬에 추가될 엔티티 하나에 대응합니다.

    prim_path에 {ENV_REGEX_NS}를 사용하면, InteractiveScene이 num_envs 개수만큼
    환경을 복제할 때 자동으로 /World/envs/env_0, /World/envs/env_1 등으로 확장됩니다.
    """

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # 바닥은 전역(global)이므로 {ENV_REGEX_NS}를 사용하지 않습니다.
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ── 돔 조명 (DomeLight) ──────────────────────────────────────────────
    # DomeLight도 전역 조명이므로 환경별로 복제하지 않습니다.
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ── Cartpole 로봇 (Articulation) ─────────────────────────────────────
    # CARTPOLE_CFG.replace()로 prim_path만 변경합니다.
    # {ENV_REGEX_NS}는 환경마다 자동으로 고유 경로로 치환됩니다.
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── 테이블 (RigidObject) ─────────────────────────────────────────────
    # Cuboid를 이용해 테이블 형태의 강체(rigid object)를 생성합니다.
    # init_state로 로봇 옆에 배치합니다.
    table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.6, 0.4, 0.4),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.4, 0.2),  # 갈색 (나무 질감)
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.8, 0.0, 0.2)),
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """시뮬레이션 루프를 실행합니다."""

    # ── 씬에서 엔티티 꺼내기 ─────────────────────────────────────────────
    # scene["속성이름"]으로 InteractiveSceneCfg에서 정의한 엔티티에 접근합니다.
    robot = scene["cartpole"]
    table = scene["table"]

    # ── 시간 관련 변수 ───────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    step_count = 0

    print(f"[INFO] Physics dt = {sim_dt:.6f}초  (1 step = {sim_dt * 1000:.2f}ms)")
    print(f"[INFO] 환경 수: {scene.num_envs}")
    print(f"[INFO] 환경 원점(env_origins):\n{scene.env_origins}")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션 정지/일시정지 처리
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 500 step마다 에피소드 리셋 ────────────────────────────────
        # 모든 환경을 동시에 초기 상태로 되돌립니다.
        if step_count % 500 == 0:
            # 카운터 리셋
            step_count = 0

            # ── root state 리셋 ──────────────────────────────────────────
            # default_root_state: shape (num_envs, 13) = [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            # 주의: root state는 시뮬레이션 월드 좌표계 기준이므로
            #        scene.env_origins를 더해 각 환경의 원점으로 오프셋합니다.
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # ── joint state 리셋 (약간의 노이즈 추가) ────────────────────
            joint_pos, joint_vel = (
                robot.data.default_joint_pos.clone(),
                robot.data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # ── 씬 내부 버퍼 초기화 ─────────────────────────────────────
            # scene.reset()은 씬에 포함된 모든 엔티티의 내부 버퍼를 초기화합니다.
            scene.reset()
            print("[INFO] 에피소드 리셋 완료 - 모든 환경 초기화")

        # ── (2) 랜덤 토크 적용 ────────────────────────────────────────────
        # robot.data.joint_pos의 shape = (num_envs, num_joints)
        # 각 환경마다 독립적인 랜덤 토크가 생성됩니다.
        efforts = torch.randn_like(robot.data.joint_pos) * 5.0
        robot.set_joint_effort_target(efforts)

        # ── (3) 씬 데이터를 시뮬레이션에 기록 ─────────────────────────────
        # 모든 엔티티의 명령(action)을 물리 엔진에 전달합니다.
        scene.write_data_to_sim()

        # ── (4) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()

        # ── (5) 스텝 카운터 증가 ──────────────────────────────────────────
        step_count += 1

        # ── (6) 씬 버퍼 업데이트 ──────────────────────────────────────────
        # sim.step() 이후 물리 엔진의 최신 상태를 씬의 내부 텐서에 반영합니다.
        scene.update(sim_dt)

        # ── (7) 200 step마다 상태 출력 ────────────────────────────────────
        if step_count % 200 == 0:
            print(f"\n[Step {step_count}]")
            print(f"  Cartpole joint positions (all envs):\n    {robot.data.joint_pos}")
            print(f"  Table root position (env 0): {table.data.root_pos_w[0]}")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """InteractiveScene을 생성하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점: 여러 환경이 보이도록 약간 멀리 설정
    sim.set_camera_view(eye=[2.5, 0.0, 4.0], target=[0.0, 0.0, 2.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    # num_envs: 복제할 환경 수 (args_cli.num_envs로 전달)
    # env_spacing: 환경 간 간격 (미터 단위)
    scene_cfg = MySceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - {args_cli.num_envs}개 환경")

    # ── sim.reset() - 물리 엔진 초기화 ────────────────────────────────────
    # 반드시 InteractiveScene 생성 이후, step() 호출 전에 실행합니다.
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 시작 준비 완료")

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────────
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
