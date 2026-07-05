"""
23_custom_env_complete.py - 커스텀 RL 환경 완전 가이드

20~22강의 모든 개념을 합쳐 처음부터 완전한 RL 환경을 만듭니다:
  - 태스크 정의: "폴을 세우고 카트를 중앙에 유지" (Cartpole Balancing)
  - Scene, Action, Observation, Reward, Termination, Event를 체계적으로 설계
  - 커스텀 관측 (sin/cos), 커스텀 보상 (가우시안, 에너지)
  - 이 환경은 24강에서 실제 RL 학습에 사용됩니다

실행:
    source ~/isaac/env_isaaclab/bin/activate
    cd ~/isaac/isaacsim_tutorials/23_custom_env_complete
    python 23_custom_env_complete.py

    # GUI 없이 실행
    python 23_custom_env_complete.py --headless

    # 환경 수 변경 (학습 시에는 1024~4096 사용)
    python 23_custom_env_complete.py --num_envs 16
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="23 - 커스텀 RL 환경 완전 가이드")
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

import isaaclab.envs.mdp as mdp

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수들
# ══════════════════════════════════════════════════════════════════════════


def obs_pole_sin_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """폴 각도를 sin/cos 변환하여 반환 → (num_envs, 2)."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def obs_cart_normalized(env: ManagerBasedRLEnv, max_pos: float = 3.0) -> torch.Tensor:
    """카트 위치를 정규화하여 반환 → (num_envs, 1)."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    cart_pos = robot.data.joint_pos[:, joint_ids[0]]
    return (cart_pos / max_pos).unsqueeze(-1)


def rew_pole_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    """폴 직립 보상: cos(angle), 직립 시 1.0."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.cos(angle)


def rew_cart_center(env: ManagerBasedRLEnv, sigma: float = 1.0) -> torch.Tensor:
    """카트 중앙 보상: exp(-pos^2 / 2sigma^2)."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    pos = robot.data.joint_pos[:, joint_ids[0]]
    return torch.exp(-pos**2 / (2.0 * sigma**2))


def rew_energy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """에너지 패널티: action^2 합."""
    return torch.sum(env.action_manager.action**2, dim=-1)


def term_pole_too_far(env: ManagerBasedRLEnv, max_angle: float = 0.5 * math.pi) -> torch.Tensor:
    """폴이 max_angle 이상 기울어지면 종료."""
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.abs(angle) > max_angle


# ══════════════════════════════════════════════════════════════════════════
#  Scene 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyCartpoleSceneCfg(InteractiveSceneCfg):
    """커스텀 Cartpole 씬: 바닥 + 로봇 + 조명."""
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
#  Action: 카트에 힘 적용
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyActionsCfg:
    """에이전트 출력 → 카트 힘."""
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Observation: 기본 + 커스텀 관측
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyObservationsCfg:
    """관측: 기본 관절 상태 + sin/cos + 정규화 위치.

    총 관측 차원: 2(pos) + 2(vel) + 2(sin/cos) + 1(norm_pos) = 7
    """

    @configclass
    class PolicyCfg(ObsGroup):
        # 기본: 관절 위치/속도 (상대값)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        # 커스텀: 폴 각도 sin/cos
        pole_sin_cos = ObsTerm(func=obs_pole_sin_cos)
        # 커스텀: 카트 위치 정규화 + 클리핑
        cart_norm = ObsTerm(
            func=obs_cart_normalized,
            params={"max_pos": 3.0},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ══════════════════════════════════════════════════════════════════════════
#  Event: 에피소드 리셋 시 랜덤 초기화
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyEventCfg:
    """리셋 이벤트: 카트/폴 위치와 속도를 랜덤하게 초기화."""
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset, mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
#  Reward: 보상 설계 (생존 + 목표 추적 + 패널티)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyRewardsCfg:
    """보상 텀들.

    설계 철학:
      - alive(+1.0): 기본 생존 보상 → 에피소드를 오래 유지하도록
      - pole_upright(+2.0): 폴 직립 보너스 → 핵심 목표
      - cart_center(+0.5): 카트 중앙 유지 → 보조 목표
      - terminating(-2.0): 실패 패널티 → 나쁜 상태 회피
      - energy(-0.001): 에너지 절약 → 효율적 제어
      - pole_vel(-0.005): 폴 안정화 → 진동 억제
    """
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=rew_pole_upright, weight=2.0)
    cart_center = RewTerm(func=rew_cart_center, weight=0.5, params={"sigma": 1.5})
    energy = RewTerm(func=rew_energy, weight=-0.001)
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


# ══════════════════════════════════════════════════════════════════════════
#  Termination: 에피소드 종료 조건
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyTerminationsCfg:
    """종료 조건.

    3가지 종료 조건:
      1) 시간 초과 (truncation) - 학습 중 에피소드 길이 제한
      2) 카트가 경계 이탈 (termination) - 복구 불가능한 실패
      3) 폴이 90도 이상 기울어짐 (termination) - 밸런싱 실패
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )
    pole_too_far = DoneTerm(
        func=term_pole_too_far,
        params={"max_angle": 0.5 * math.pi},
    )


# ══════════════════════════════════════════════════════════════════════════
#  전체 환경 설정 조립
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MyCartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """완전한 커스텀 Cartpole 환경 설정.

    이 설정은 24강에서 RL 학습에 그대로 사용됩니다.
    """

    # 씬: 로봇 + 바닥 + 조명
    scene: MyCartpoleSceneCfg = MyCartpoleSceneCfg(num_envs=4, env_spacing=4.0)

    # MDP 구성 요소
    actions: MyActionsCfg = MyActionsCfg()
    observations: MyObservationsCfg = MyObservationsCfg()
    events: MyEventCfg = MyEventCfg()
    rewards: MyRewardsCfg = MyRewardsCfg()
    terminations: MyTerminationsCfg = MyTerminationsCfg()

    def __post_init__(self) -> None:
        # 물리 설정
        self.decimation = 2          # 정책 1회 → 물리 2회
        self.episode_length_s = 5.0  # 에피소드 최대 5초
        self.sim.dt = 1.0 / 120.0   # 물리 120Hz → 정책 60Hz
        self.sim.render_interval = self.decimation
        # 카메라
        self.viewer.eye = (8.0, 0.0, 5.0)


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수: 환경 테스트
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """커스텀 환경을 생성하고 랜덤 액션으로 테스트합니다."""

    env_cfg = MyCartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print("[INFO] 커스텀 Cartpole 환경 생성 완료!")
    print(f"  환경 수:       {env.num_envs}")
    print(f"  관측 공간:     {env.observation_space}")
    print(f"  액션 공간:     {env.action_space}")
    print(f"  에피소드 길이:  {env_cfg.episode_length_s}초")
    print(f"  정책 주기:     {env_cfg.decimation * env_cfg.sim.dt:.4f}초")
    print("=" * 70)

    # ── 보상/종료 텀 정보 출력 ────────────────────────────────────────────
    print("\n[보상 텀]")
    for name, cfg in zip(env.reward_manager._term_names, env.reward_manager._term_cfgs):
        print(f"  {name:20s} weight={cfg.weight:+.4f}")

    print("\n[종료 텀]")
    for name, cfg in zip(env.termination_manager._term_names, env.termination_manager._term_cfgs):
        is_timeout = getattr(cfg, 'time_out', False)
        print(f"  {name:20s} time_out={is_timeout}")

    # ── 환경 리셋 ────────────────────────────────────────────────────────
    obs, info = env.reset()
    print(f"\n[리셋] 관측 shape: {obs['policy'].shape}")

    # ── 환경 테스트 루프 ─────────────────────────────────────────────────
    total_steps = 0
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    completed_rewards = []

    while simulation_app.is_running():
        action = torch.randn(env.num_envs, 1, device=env.device).clamp(-1, 1)
        obs, rew, terminated, truncated, info = env.step(action)

        episode_rewards += rew
        episode_lengths += 1
        total_steps += 1

        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                ep_rew = episode_rewards[idx].item()
                ep_len = episode_lengths[idx].item()
                completed_rewards.append(ep_rew)

                reason = "timeout" if truncated[idx].item() else "terminated"
                print(
                    f"[에피소드 {len(completed_rewards):3d}] "
                    f"Env {idx.item()} | 길이: {ep_len:4d} | "
                    f"보상: {ep_rew:+8.2f} | 종료: {reason}"
                )

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0

        if total_steps >= 500:
            break

    # ── 통계 출력 ────────────────────────────────────────────────────────
    if completed_rewards:
        avg_reward = sum(completed_rewards) / len(completed_rewards)
        print(f"\n{'=' * 70}")
        print(f"[통계] 완료 에피소드: {len(completed_rewards)}")
        print(f"[통계] 평균 보상: {avg_reward:.2f}")
        print(f"[통계] 최대 보상: {max(completed_rewards):.2f}")
        print(f"[통계] 최소 보상: {min(completed_rewards):.2f}")
        print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
