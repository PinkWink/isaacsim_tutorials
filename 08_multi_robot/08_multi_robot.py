"""
08_multi_robot.py - InteractiveScene으로 다중 로봇 시뮬레이션

InteractiveScene과 {ENV_REGEX_NS}를 사용하여 여러 환경에
서로 다른 로봇(Cartpole, Franka)을 동시에 배치하고 제어합니다:
  - InteractiveSceneCfg + @configclass로 선언적 씬 구성
  - {ENV_REGEX_NS}를 이용한 환경별 자동 복제
  - scene["key"]로 로봇 접근
  - env_ids를 이용한 특정 환경만 선택적 제어
  - 두 로봇에 각각 다른 액션 적용

실행:
    source env_isaaclab/bin/activate
    cd lectures/08_multi_robot
    python 08_multi_robot.py --num_envs 4

    # GUI 없이 실행
    python 08_multi_robot.py --headless --num_envs 4
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="08 - InteractiveScene 다중 로봇 튜토리얼")
parser.add_argument("--num_envs", type=int, default=4, help="생성할 환경(env) 개수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# isaaclab_assets 패키지에는 다양한 로봇의 ArticulationCfg가 미리 정의되어 있습니다.
from isaaclab_assets import CARTPOLE_CFG  # isort:skip
from isaaclab_assets import FRANKA_PANDA_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# 씬 설정 (InteractiveSceneCfg)
# ══════════════════════════════════════════════════════════════════════════
# @configclass 데코레이터를 사용하면 dataclass처럼 필드를 선언하여
# 씬의 구성 요소를 정의할 수 있습니다.
#
# {ENV_REGEX_NS}: 환경 정규식 네임스페이스
#   - InteractiveScene이 num_envs 만큼 환경을 자동 복제할 때,
#     이 패턴이 각 환경의 고유 경로로 치환됩니다.
#   - 예: num_envs=4이면 /World/envs/env_0, /World/envs/env_1, ...
#   - 모든 환경에 동일한 로봇이 생성됩니다.


@configclass
class MultiRobotSceneCfg(InteractiveSceneCfg):
    """두 종류의 로봇(Cartpole, Franka)이 포함된 씬 설정.

    하나의 InteractiveSceneCfg 안에 여러 로봇을 선언하면
    모든 환경에 두 로봇이 함께 생성됩니다.
    """

    # ── 바닥면 ────────────────────────────────────────────────────────────
    # prim_path에 {ENV_REGEX_NS}를 쓰지 않으면 전체 월드에 하나만 생성됩니다.
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ── 돔 조명 ───────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ── Cartpole 로봇 ────────────────────────────────────────────────────
    # {ENV_REGEX_NS}/Cartpole → 각 환경마다 하나의 Cartpole이 생성됩니다.
    # CARTPOLE_CFG.replace()로 prim_path만 변경한 새 설정을 만듭니다.
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Cartpole"
    )

    # ── Franka Panda 로봇 ────────────────────────────────────────────────
    # Franka를 Cartpole 옆에 배치하기 위해 init_state.pos를 오프셋합니다.
    # init_state.pos는 해당 환경의 원점(env_origin) 기준 상대 좌표입니다.
    franka: ArticulationCfg = FRANKA_PANDA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Franka",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.5, 0.0, 0.0),  # Cartpole에서 x축으로 1.5m 떨어진 위치
            joint_pos={
                "panda_joint1": 0.0,
                "panda_joint2": -0.569,
                "panda_joint3": 0.0,
                "panda_joint4": -2.810,
                "panda_joint5": 0.0,
                "panda_joint6": 3.037,
                "panda_joint7": 0.741,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """다중 로봇 시뮬레이션 루프를 실행합니다."""

    # ── scene["key"]로 로봇 접근 ──────────────────────────────────────────
    # InteractiveSceneCfg에서 정의한 필드 이름이 key가 됩니다.
    cartpole = scene["cartpole"]
    franka = scene["franka"]

    sim_dt = sim.get_physics_dt()
    step_count = 0

    # ── 관절 정보 출력 ────────────────────────────────────────────────────
    print("=" * 70)
    print("[Cartpole 로봇 정보]")
    print(f"  관절 이름: {cartpole.joint_names}")
    print(f"  관절 개수: {cartpole.num_joints}")
    print(f"  환경 개수: {cartpole.num_instances}")
    print()
    print("[Franka Panda 로봇 정보]")
    print(f"  관절 이름: {franka.joint_names}")
    print(f"  관절 개수: {franka.num_joints}")
    print(f"  환경 개수: {franka.num_instances}")
    print("=" * 70)

    # ── 환경 인덱스 준비 ──────────────────────────────────────────────────
    # 특정 환경만 제어하고 싶을 때 사용할 인덱스
    # 예: env 0번만 선택
    env_0_ids = torch.tensor([0], dtype=torch.long, device=sim.device)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션이 정지되었으면 루프 종료
        if sim.is_stopped():
            break
        # 일시정지 상태면 렌더링만 수행
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 500 step마다 에피소드 리셋 ────────────────────────────────
        if step_count % 500 == 0:
            # -- Cartpole 리셋 (모든 환경) --
            # default_root_state: shape (num_envs, 13) - [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            root_state = cartpole.data.default_root_state.clone()
            # 각 환경의 원점 오프셋을 더해줘야 올바른 월드 좌표가 됩니다.
            root_state[:, :3] += scene.env_origins
            cartpole.write_root_pose_to_sim(root_state[:, :7])
            cartpole.write_root_velocity_to_sim(root_state[:, 7:])
            # 관절 초기 상태 + 약간의 노이즈
            joint_pos, joint_vel = (
                cartpole.data.default_joint_pos.clone(),
                cartpole.data.default_joint_vel.clone(),
            )
            joint_pos += torch.rand_like(joint_pos) * 0.1
            cartpole.write_joint_state_to_sim(joint_pos, joint_vel)

            # -- Franka 리셋 (모든 환경) --
            root_state = franka.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            franka.write_root_pose_to_sim(root_state[:, :7])
            franka.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos, joint_vel = (
                franka.data.default_joint_pos.clone(),
                franka.data.default_joint_vel.clone(),
            )
            franka.write_joint_state_to_sim(joint_pos, joint_vel)

            # -- 씬 전체 내부 버퍼 리셋 --
            scene.reset()
            print(f"\n[INFO] 에피소드 리셋 (step={step_count})")

        # ── (2) Cartpole에 랜덤 토크 적용 (모든 환경) ─────────────────────
        # shape: (num_envs, num_joints) = (4, 2)
        cartpole_efforts = torch.randn_like(cartpole.data.joint_pos) * 5.0
        cartpole.set_joint_effort_target(cartpole_efforts)

        # ── (3) Franka에 랜덤 토크 적용 (모든 환경) ───────────────────────
        # shape: (num_envs, num_joints) = (4, 9)
        franka_efforts = torch.randn_like(franka.data.joint_pos) * 2.0
        franka.set_joint_effort_target(franka_efforts)

        # ── (4) 특정 환경(env 0)의 Franka에만 큰 토크 적용 ───────────────
        # env_ids 파라미터를 사용하면 특정 환경의 로봇만 제어할 수 있습니다.
        # 아래는 env 0번의 Franka 첫 번째 관절에만 큰 토크를 적용하는 예시입니다.
        if step_count % 100 < 50:
            # env 0번 Franka에 강한 토크를 주어 차별화된 움직임 관찰
            strong_effort = torch.zeros(1, franka.num_joints, device=sim.device)
            strong_effort[0, 0] = 50.0  # panda_joint1에 50Nm 토크
            franka.set_joint_effort_target(strong_effort, env_ids=env_0_ids)

        # ── (5) 씬 데이터를 시뮬레이션에 기록 ─────────────────────────────
        # scene.write_data_to_sim()은 모든 에셋의 write_data_to_sim()을
        # 한꺼번에 호출하는 편의 함수입니다.
        scene.write_data_to_sim()

        # ── (6) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()
        step_count += 1

        # ── (7) 씬 버퍼 업데이트 ─────────────────────────────────────────
        # scene.update()는 모든 에셋의 데이터 버퍼를 최신 상태로 갱신합니다.
        scene.update(sim_dt)

        # ── (8) 200 step마다 상태 출력 ────────────────────────────────────
        if step_count % 200 == 0:
            print(f"\n[Step {step_count}]")
            # Cartpole: 각 환경의 관절 위치 (shape: num_envs x 2)
            print(f"  Cartpole joint pos (all envs):")
            for env_idx in range(cartpole.num_instances):
                pos = cartpole.data.joint_pos[env_idx]
                print(f"    env {env_idx}: slider={pos[0]:.3f}, pole={pos[1]:.3f}")

            # Franka: env 0번의 관절 위치만 출력 (9개 관절)
            franka_pos = franka.data.joint_pos[0]
            print(f"  Franka joint pos (env 0): {franka_pos[:7].tolist()}")
            print(f"  Franka gripper  (env 0): {franka_pos[7:].tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """다중 로봇 InteractiveScene을 설정하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점: 모든 환경이 보이도록 넓은 시야 설정
    sim.set_camera_view(eye=[8.0, 8.0, 5.0], target=[0.0, 0.0, 1.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    # MultiRobotSceneCfg에 정의한 모든 에셋이 num_envs개 환경에 자동 복제됩니다.
    # env_spacing: 환경 간 간격 (미터). 로봇이 겹치지 않도록 충분히 설정합니다.
    scene_cfg = MultiRobotSceneCfg(num_envs=args_cli.num_envs, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)

    # ── sim.reset() ───────────────────────────────────────────────────────
    # InteractiveScene 생성 후 반드시 호출해야 물리 엔진이 초기화됩니다.
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 준비 완료")
    print(f"[INFO] 환경 개수: {args_cli.num_envs}")
    print(f"[INFO] 환경 원점 좌표:\n{scene.env_origins}")

    # ── 시뮬레이션 실행 ───────────────────────────────────────────────────
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
