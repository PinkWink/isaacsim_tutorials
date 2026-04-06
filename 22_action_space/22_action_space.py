"""
22_action_space.py - 액션 공간: 다양한 로봇 제어 방식

IsaacLab이 제공하는 여러 ActionTermCfg를 비교하고, Cartpole에 적용합니다:
  - JointEffortActionCfg:   관절 토크(힘) 직접 제어
  - JointPositionActionCfg: 관절 위치(각도) 목표 설정
  - JointVelocityActionCfg: 관절 속도 목표 설정
  - scale/offset를 이용한 액션 범위 조정
  - 여러 관절을 동시에 제어하는 멀티-조인트 액션

실행:
    source env_isaaclab/bin/activate
    cd lectures/22_action_space
    python 22_action_space.py

    # 액션 모드 선택 (effort / position / velocity)
    python 22_action_space.py --action_mode effort
    python 22_action_space.py --action_mode position
    python 22_action_space.py --action_mode velocity
"""

# ── 1. AppLauncher 패턴 ─────────────────────���────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="22 - 액션 공간 튜토리얼")
parser.add_argument("--num_envs", type=int, default=4, help="생성할 환경(env) 개수")
parser.add_argument(
    "--action_mode",
    type=str,
    default="effort",
    choices=["effort", "position", "velocity"],
    help="액션 제어 모드 선택",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ─────────���─────────────────────────────
"""AppLauncher 초기화 이후에 나머지 모듈을 import 합니다."""

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════��═══════════════════════════════════
# Scene 설정
# ═════════════════��════════════════════════════════════════════════════════


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole 씬."""
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ���════════════════════════════��════════════════════════════════════════════
# 3가지 액션 모드 정의
# ��════════════════��════════════════════════════���═══════════════════════════


@configclass
class EffortActionsCfg:
    """Effort (토크) 제어: 관절에 직접 힘을 가합니다.

    에이전트 출력 [-1, 1] → 실제 힘 [-100, 100]N
    - 장점: 직접적이고 물리적으로 정확
    - 단점: 제어가 어렵고, 시뮬레이션 불안정 가능
    """
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,  # [-1, 1] → [-100, 100]N
    )


@configclass
class PositionActionsCfg:
    """Position (위치) 제어: 관절의 목표 위치를 설정합니다.

    에이전트 출력 [-1, 1] → 목표 위치 [-2, 2]m (+ 기본 위치 0)
    - 장점: 안정적이고 제어가 직관적
    - 단점: 빠른 동작에 지연 발생 가능
    - use_default_offset=True: 기본 관절 위치를 오프셋으로 사용
    """
    joint_position = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=2.0,    # [-1, 1] → ±2m 범위
        use_default_offset=True,  # 기본 위치(0) 기준
    )


@configclass
class VelocityActionsCfg:
    """Velocity (속도) 제어: 관절의 목표 속도를 설정합니다.

    에이전트 출력 [-1, 1] → 목표 속도 [-5, 5]m/s
    - 장점: 부드러운 움직임
    - 단점: 위치 정밀도 떨어짐
    """
    joint_velocity = mdp.JointVelocityActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=5.0,    # [-1, 1] → [-5, 5]m/s
        use_default_offset=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# 공통 Observation / Event / Reward / Termination
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_pos = RewTerm(
        func=mdp.joint_deviation_l1, weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)},
    )


# ═════════════════════════════════════════════════════════════���════════════
# 환경 설정: 액션 모드에 따라 분기
# ════════════════════���═════════════════════════���═══════════════════════════


@configclass
class CartpoleEffortEnvCfg(ManagerBasedRLEnvCfg):
    """Effort 제어 환경."""
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions: EffortActionsCfg = EffortActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


@configclass
class CartpolePositionEnvCfg(ManagerBasedRLEnvCfg):
    """Position 제어 환경."""
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions: PositionActionsCfg = PositionActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


@configclass
class CartpoleVelocityEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity 제어 환경."""
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions: VelocityActionsCfg = VelocityActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ════════════════════════════════════════════════��═════════════════════════


def main() -> None:
    """선택된 액션 모드로 Cartpole 환경을 실행합니다."""

    # 액션 모드에 따라 환경 설정 선택
    action_mode = args_cli.action_mode
    if action_mode == "effort":
        env_cfg = CartpoleEffortEnvCfg()
    elif action_mode == "position":
        env_cfg = CartpolePositionEnvCfg()
    elif action_mode == "velocity":
        env_cfg = CartpoleVelocityEnvCfg()
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")

    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print(f"[INFO] 액션 모드: {action_mode.upper()}")
    print(f"  환경 수:     {env.num_envs}")
    print(f"  관측 공간:   {env.observation_space}")
    print(f"  액션 공간:   {env.action_space}")
    print("=" * 70)

    # ── 환경 리셋 ────────────────────────────────────────────────────────
    obs, info = env.reset()

    total_steps = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    completed_episodes = 0

    # ── 실행 루프 ────────────────────────────────────────────────────────
    while simulation_app.is_running():
        # 각 모드에 맞는 랜덤 액션 생성
        # 모든 모드에서 에이전트 출력은 [-1, 1] 범위
        action_dim = env.action_space.shape[-1]
        action = 2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0

        # 사인파 액션 (주기적 움직임) - 랜덤보다 관찰하기 쉬움
        if total_steps < 150:
            t = total_steps * 0.05
            action = torch.sin(torch.tensor(t)) * torch.ones_like(action)

        obs, rew, terminated, truncated, info = env.step(action)
        episode_rewards += rew
        episode_lengths += 1
        total_steps += 1

        # 에피소드 종료
        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                completed_episodes += 1
                print(
                    f"[에피소드 {completed_episodes}] Env {idx.item()} ({action_mode}) | "
                    f"길이: {int(episode_lengths[idx].item())} | "
                    f"보상: {episode_rewards[idx].item():.2f}"
                )
            episode_rewards[done] = 0.0
            episode_lengths[done] = 0.0

        # 주기적 상태 출력
        if total_steps % 100 == 0:
            print(
                f"[Step {total_steps:5d}] "
                f"action={action[0, 0]:.3f} | "
                f"cart_pos={obs['policy'][0, 0]:.3f} | "
                f"pole_angle={obs['policy'][0, 1]:.3f}"
            )

        if total_steps >= 300:
            break

    print(f"\n[결과] {action_mode.upper()} 모드 | 완료 에피소드: {completed_episodes}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 ��료")
