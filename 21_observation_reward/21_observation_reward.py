"""
21_observation_reward.py - 관측과 보상: 커스텀 ObsTerm & RewardTerm 만들기

ManagerBasedRLEnv에서 커스텀 관측/보상 함수를 작성하는 방법을 배웁니다:
  - ObservationTermCfg에 커스텀 함수 등록
  - RewardTermCfg에 커스텀 보상 함수 등록
  - noise, clip, scale 옵션으로 관측 전처리
  - 보상 shaping 기법: 목표 추적, 에너지 패널티, 생존 보너스
  - Cartpole 환경에 맞춤 관측/보상을 추가하고 효과 비교

실행:
    source env_isaaclab/bin/activate
    cd lectures/21_observation_reward
    python 21_observation_reward.py

    # GUI 없이 실행
    python 21_observation_reward.py --headless

    # 환경 수 변경
    python 21_observation_reward.py --num_envs 8
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="21 - 관측과 보상 커스텀 튜토리얼")
parser.add_argument("--num_envs", type=int, default=4, help="생성할 환경(env) 개수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
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
from isaaclab.utils.math import wrap_to_pi

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# 커스텀 관측 함수 (ObservationTerm)
# ══════════════════════════════════════════════════════════════════════════
# 관측 함수 시그니처: func(env: ManagerBasedEnv, **params) -> torch.Tensor
# 반환 shape: (num_envs, obs_dim)


def pole_angle_sin_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """폴 각도를 sin/cos로 변환하여 반환합니다.

    각도를 직접 관측하면 -π와 +π 경계에서 불연속이 발생합니다.
    sin/cos 변환은 이 문제를 해결하여 연속적인 관측을 제공합니다.

    Returns:
        shape (num_envs, 2): [sin(pole_angle), cos(pole_angle)]
    """
    robot = env.scene["robot"]
    # cart_to_pole 관절의 인덱스 찾기
    joint_ids, _ = robot.find_joints("cart_to_pole")
    pole_angle = robot.data.joint_pos[:, joint_ids[0]]  # (num_envs,)
    sin_angle = torch.sin(pole_angle)
    cos_angle = torch.cos(pole_angle)
    return torch.stack([sin_angle, cos_angle], dim=-1)  # (num_envs, 2)


def cart_position_normalized(
    env: ManagerBasedRLEnv,
    max_pos: float = 3.0,
) -> torch.Tensor:
    """카트 위치를 [-1, 1] 범위로 정규화하여 반환합니다.

    Args:
        max_pos: 정규화 범위 (±max_pos → ±1)

    Returns:
        shape (num_envs, 1): 정규화된 카트 위치
    """
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    cart_pos = robot.data.joint_pos[:, joint_ids[0]]  # (num_envs,)
    normalized = cart_pos / max_pos
    return normalized.unsqueeze(-1)  # (num_envs, 1)


def all_joint_velocities(env: ManagerBasedRLEnv) -> torch.Tensor:
    """모든 관절의 속도를 반환합니다.

    Returns:
        shape (num_envs, num_joints): 모든 관절의 속도
    """
    robot = env.scene["robot"]
    return robot.data.joint_vel  # (num_envs, num_joints)


# ══════════════════════════════════════════════════════════════════════════
# 커스텀 보상 함수 (RewardTerm)
# ══════════════════════════════════════════════════════════════════════════
# 보상 함수 시그니처: func(env: ManagerBasedRLEnv, **params) -> torch.Tensor
# 반환 shape: (num_envs,)


def pole_upright_bonus(env: ManagerBasedRLEnv) -> torch.Tensor:
    """폴이 직립에 가까울수록 높은 보상을 줍니다.

    cos(pole_angle)이 1에 가까울수록 (직립) 높은 보상.
    cos(0°) = 1.0 (최대), cos(90°) = 0.0 (최소)

    Returns:
        shape (num_envs,): 0~1 범위의 보상
    """
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    pole_angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.cos(pole_angle)  # (num_envs,)


def cart_center_reward(
    env: ManagerBasedRLEnv,
    sigma: float = 1.0,
) -> torch.Tensor:
    """카트가 중앙에 가까울수록 높은 보상 (가우시안 커널).

    reward = exp(-cart_pos² / (2 * sigma²))

    Args:
        sigma: 가우시안 폭 (작을수록 중앙에 엄격)

    Returns:
        shape (num_envs,): 0~1 범위의 보상
    """
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    cart_pos = robot.data.joint_pos[:, joint_ids[0]]
    return torch.exp(-cart_pos**2 / (2.0 * sigma**2))


def energy_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """에이전트가 사용한 에너지(힘의 제곱합)에 대한 패널티.

    에너지를 절약하는 행동을 장려합니다.

    Returns:
        shape (num_envs,): 에너지 사용량 (양수, weight를 음수로 설정)
    """
    # applied_action: 실제 적용된 액션 값
    return torch.sum(env.action_manager.action**2, dim=-1)


# ══════════════════════════════════════════════════════════════════════════
# Scene 설정 (20강과 동일)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole 씬 설정."""
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


# ══════════════════════════════════════════════════════════════════════════
# Action 설정 (20강과 동일)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """Cartpole 액션: 카트에 힘 적용."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 커스텀 Observation 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:
    """커스텀 관측 스펙.

    기본 관측(joint_pos_rel, joint_vel_rel)에 추가로:
      - pole_angle_sin_cos: 폴 각도의 sin/cos 변환
      - cart_pos_norm: 카트 위치 정규화 (clip 적용)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크 관측 그룹.

        총 관측 차원: 2(pos) + 2(vel) + 2(sin/cos) + 1(norm_pos) = 7
        """

        # ── 기본 MDP 관측 ────────────────────────────────────────────────
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        # ── 커스텀 관측: 폴 각도의 sin/cos ──────────────────────────────
        # 함수 레퍼런스만 전달하면 됩니다.
        pole_sin_cos = ObsTerm(func=pole_angle_sin_cos)

        # ── 커스텀 관측: 정규화된 카트 위치 ─────────────────────────────
        # params로 추가 인자를 전달할 수 있습니다.
        # clip=(-1.0, 1.0): 출력값을 [-1, 1]로 제한
        cart_pos_norm = ObsTerm(
            func=cart_position_normalized,
            params={"max_pos": 3.0},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ══════════════════════════════════════════════════════════════════════════
# Event 설정 (20강과 동일)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class EventCfg:
    """리셋 이벤트."""
    reset_cart_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole_position = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# 커스텀 Reward 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """커스텀 보상 텀.

    보상 설계 전략:
      1) 생존 보상: 에피소드가 계속되는 것 자체가 좋음
      2) 목표 추적: 폴을 세우고, 카트를 중앙에 유지
      3) 에너지 패널티: 불필요한 힘 사용 억제
      4) 실패 패널티: 종료 시 큰 음수 보상
    """

    # ── 생존 보상 (기본 MDP 함수) ─────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # ── 실패 패널티 ──────────────────────────────────────────────────
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # ── 커스텀: 폴 직립 보너스 ───────────────────────────────────────
    pole_upright = RewTerm(func=pole_upright_bonus, weight=2.0)

    # ── 커스텀: 카트 중앙 보상 (가우시안) ────────────────────────────
    cart_center = RewTerm(
        func=cart_center_reward,
        weight=0.5,
        params={"sigma": 1.5},
    )

    # ── 커스텀: 에너지 패널티 ────────────────────────────────────────
    energy = RewTerm(func=energy_penalty, weight=-0.001)

    # ── 기본 MDP: 폴 각속도 억제 ─────────────────────────────────────
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


# ══════════════════════════════════════════════════════════════════════════
# Termination 설정 (20강과 동일)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class TerminationsCfg:
    """에피소드 종료 조건."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# 전체 환경 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleCustomEnvCfg(ManagerBasedRLEnvCfg):
    """커스텀 관측/보상이 적용된 Cartpole 환경 설정."""

    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
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
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """커스텀 관측/보상 환경을 생성하고 실행합니다."""

    env_cfg = CartpoleCustomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print("[INFO] 커스텀 관측/보상 환경 생성 완료!")
    print(f"  환경 수:     {env.num_envs}")
    print(f"  관측 공간:   {env.observation_space}")
    print(f"  액션 공간:   {env.action_space}")
    print("=" * 70)

    # ── 보상 텀 이름 출력 ────────────────────────────────────────────────
    print("\n[보상 텀 목록]")
    for name, cfg in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
        print(f"  {name}: weight={cfg.weight}")

    # ── 환경 리셋 ────────────────────────────────────────────────────────
    obs, info = env.reset()
    print(f"\n[리셋] 관측 shape: {obs['policy'].shape}")
    print(f"[리셋] 관측 (env 0): {obs['policy'][0]}")

    # ── 실행 루프 ────────────────────────────────────────────────────────
    total_steps = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_count = 0

    while simulation_app.is_running():
        # 랜덤 액션
        action = torch.randn(env.num_envs, 1, device=env.device)

        obs, rew, terminated, truncated, info = env.step(action)
        episode_rewards += rew
        total_steps += 1

        # 에피소드 종료 시 개별 보상 텀 출력
        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                episode_count += 1
                # 개별 보상 텀의 마지막 값 확인
                print(
                    f"\n[에피소드 {episode_count}] Env {idx.item()}"
                    f" | 총 보상: {episode_rewards[idx].item():.2f}"
                    f" | terminated={terminated[idx].item()}"
                )
                # 각 보상 텀의 마지막 기여도 출력
                for term_idx, name in enumerate(env.reward_manager._term_names):
                    term_reward = env.reward_manager._step_reward[idx, term_idx].item()
                    print(f"    {name:20s}: {term_reward:+.4f}")

            episode_rewards[done] = 0.0

        # 주기적 상태 출력
        if total_steps % 100 == 0:
            print(
                f"\n[Step {total_steps:5d}] "
                f"보상: {rew.mean().item():+.4f} | "
                f"관측 dim: {obs['policy'].shape[-1]}"
            )

        if total_steps >= 300:
            print(f"\n[INFO] {total_steps} 스텝 완료")
            break

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
