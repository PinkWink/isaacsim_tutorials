"""
20_base_env.py - RL 환경의 기초: ManagerBasedRLEnv

IsaacLab의 ManagerBasedRLEnv를 사용하여 강화학습 환경의 기본 구조를 이해합니다:
  - ManagerBasedRLEnvCfg: Scene, Action, Observation, Reward, Termination, Event 설정
  - Cartpole 로봇을 사용한 최소한의 RL 환경 구성
  - 랜덤 액션으로 환경 step/reset 사이클 실행
  - decimation(물리 스텝 반복)과 에피소드 관리 개념

실행:
    source env_isaaclab/bin/activate
    cd lectures/20_base_env
    python 20_base_env.py

    # GUI 없이 실행
    python 20_base_env.py --headless

    # 환경 수 변경
    python 20_base_env.py --num_envs 16
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="20 - RL 환경 기초 (ManagerBasedRLEnv)")
parser.add_argument("--num_envs", type=int, default=4, help="생성할 환경(env) 개수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
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

# ── 3. 공통 MDP 함수 import ──────────────────────────────────────────────
# isaaclab.envs.mdp에는 미리 정의된 관측, 보상, 종료, 이벤트, 액션 함수가 있습니다.
import isaaclab.envs.mdp as mdp

# ── 4. 사전 정의된 Cartpole 로봇 설정 ────────────────────────────────────
# CARTPOLE_CFG: 카트(slider_to_cart) + 폴(cart_to_pole) 2-관절 로봇
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# 1단계: Scene 설정 — 어떤 로봇/환경을 사용할 것인가?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole RL 환경의 씬 설정.

    구성 요소:
      - 바닥면 (GroundPlane)
      - Cartpole 로봇 (ArticulationCfg)
      - 조명 (DomeLight)
    """

    # ── 바닥면 ───────────────────────────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )

    # ── Cartpole 로봇 ───────────────────────────────────────────────────
    # {ENV_REGEX_NS}: 각 환경마다 별도의 인스턴스를 생성하기 위한 네임스페이스 패턴
    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── 조명 ─────────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ══════════════════════════════════════════════════════════════════════════
# 2단계: Action 설정 — 에이전트가 할 수 있는 행동은?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """에이전트의 액션 스펙을 정의합니다.

    Cartpole에서 에이전트는 카트에 좌/우 힘(effort)을 가합니다.
    - joint_names: 힘을 가할 관절 이름
    - scale: 액션값에 곱할 스케일 ([-1, 1] → [-100, 100] N)
    """
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3단계: Observation 설정 — 에이전트가 볼 수 있는 정보는?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:
    """관측(Observation) 스펙을 정의합니다.

    ObservationGroup으로 관측을 그룹화합니다:
      - "policy" 그룹: 정책 네트워크에 입력되는 관측
      - concatenate_terms=True: 모든 관측 텀을 하나의 텐서로 합침
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """정책 네트워크에 들어갈 관측값."""

        # joint_pos_rel: 각 관절의 현재 위치 - 기본 위치 (2차원)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel_rel: 각 관절의 현재 속도 - 기본 속도 (2차원)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False  # 노이즈 비활성화
            self.concatenate_terms = True   # 모든 텀을 concat → (num_envs, 4)

    # "policy"라는 이름의 관측 그룹 등록
    policy: PolicyCfg = PolicyCfg()


# ══════════════════════════════════════════════════════════════════════════
# 4단계: Event 설정 — 리셋 시 무엇을 초기화할 것인가?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class EventCfg:
    """이벤트(리셋, 도메인 랜덤화) 설정.

    mode="reset": 에피소드 시작(리셋)마다 실행됩니다.
    - 카트 위치를 [-1, 1]m 범위에서 랜덤하게 초기화
    - 폴 각도를 [-45°, 45°] 범위에서 랜덤하게 초기화
    """

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
# 5단계: Reward 설정 — 무엇이 좋은 행동인가?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """보상(Reward) 텀들을 정의합니다.

    각 RewTerm의 weight를 곱해서 총 보상을 계산합니다:
      total_reward = Σ (weight_i × term_i)
    """

    # (1) 생존 보상: 에피소드가 끝나지 않았으면 +1
    alive = RewTerm(func=mdp.is_alive, weight=1.0)

    # (2) 종료 패널티: 에피소드가 끝나면 -2
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)

    # (3) 폴을 세우는 보상: 폴 각도가 0(직립)에서 벗어날수록 패널티
    #     joint_deviation_l1: 현재 관절 위치와 기본값(0.0)의 차이를 L1 페널티
    pole_pos = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
        },
    )

    # (4) 카트 속도 억제: 카트가 너무 빠르게 움직이면 패널티
    cart_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"])},
    )

    # (5) 폴 각속도 억제: 폴이 너무 빠르게 흔들리면 패널티
    pole_vel = RewTerm(
        func=mdp.joint_vel_l1,
        weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])},
    )


# ══════════════════════════════════════════════════════════════════════════
# 6단계: Termination 설정 — 에피소드가 끝나는 조건은?
# ══════════════════════════════════════════════════════════════════════════


@configclass
class TerminationsCfg:
    """에피소드 종료 조건을 정의합니다."""

    # (1) 시간 초과: episode_length_s 초가 지나면 종료 (time_out=True → truncation)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) 카트가 경계를 벗어남: ±3m 이상 이동하면 종료
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# 7단계: 전체 환경 설정 조합 — ManagerBasedRLEnvCfg
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleEnvCfg(ManagerBasedRLEnvCfg):
    """Cartpole RL 환경의 전체 설정.

    ManagerBasedRLEnvCfg는 위에서 정의한 6개 구성 요소를 하나로 묶습니다:
      - scene:        씬 (로봇, 조명, 바닥)
      - actions:       액션 (에이전트가 할 수 있는 행동)
      - observations:  관측 (에이전트가 볼 수 있는 정보)
      - events:        이벤트 (리셋 시 초기화)
      - rewards:       보상 (좋은 행동의 기준)
      - terminations:  종료 조건 (에피소드가 끝나는 시점)
    """

    # 씬 설정
    scene: CartpoleSceneCfg = CartpoleSceneCfg(
        num_envs=4, env_spacing=4.0
    )
    # MDP 구성 요소
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """환경 초기화 후 추가 설정."""
        # decimation: 정책이 1번 결정할 때 물리 시뮬레이션을 몇 번 반복할지
        # decimation=2, dt=1/120 → 정책 주기 = 2 × (1/120) = 1/60초 ≈ 60Hz
        self.decimation = 2
        # 에피소드 최대 길이 (초)
        self.episode_length_s = 5.0
        # 카메라 시점
        self.viewer.eye = (8.0, 0.0, 5.0)
        # 물리 시뮬레이션 타임스텝
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수: ManagerBasedRLEnv 생성 및 실행
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Cartpole RL 환경을 생성하고 랜덤 액션으로 실행합니다."""

    # ── 환경 설정 생성 ──────────────────────────────────────────────────
    env_cfg = CartpoleEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── ManagerBasedRLEnv 생성 ──────────────────────────────────────────
    # 설정(Cfg)만 넘기면 환경이 자동으로 구성됩니다:
    #   1) SimulationContext 생성 및 초기화
    #   2) InteractiveScene 생성 (로봇, 조명, 바닥)
    #   3) 각 Manager 초기화 (Action, Observation, Reward, Termination, Event)
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print("[INFO] ManagerBasedRLEnv 생성 완료!")
    print(f"  환경 수:         {env.num_envs}")
    print(f"  관측 차원:       {env.observation_space}")
    print(f"  액션 차원:       {env.action_space}")
    print(f"  에피소드 길이:    {env_cfg.episode_length_s}초")
    print(f"  decimation:     {env_cfg.decimation}")
    print(f"  물리 dt:         {env_cfg.sim.dt}초")
    print(f"  정책 주기:       {env_cfg.decimation * env_cfg.sim.dt:.4f}초")
    print("=" * 70)

    # ── 환경 리셋 (첫 관측 획득) ────────────────────────────────────────
    # env.reset()은 모든 환경을 초기화하고 첫 관측(obs)을 반환합니다.
    obs, info = env.reset()
    print(f"\n[리셋] 초기 관측 shape: {obs['policy'].shape}")
    print(f"[리셋] 초기 관측값 (env 0): {obs['policy'][0]}")

    # ── 환경 실행 루프 ──────────────────────────────────────────────────
    total_steps = 0
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_count = torch.zeros(env.num_envs, device=env.device)

    while simulation_app.is_running():
        # ── 랜덤 액션 생성 ──────────────────────────────────────────────
        # 액션 공간에서 균일 랜덤 샘플링 [-1, 1]
        action = torch.randn(env.num_envs, 1, device=env.device)

        # ── env.step(action) — RL 환경의 핵심 ──────────────────────────
        # 반환값:
        #   obs:  관측 딕셔너리 {"policy": tensor(num_envs, obs_dim)}
        #   rew:  보상 tensor(num_envs,)
        #   terminated: 에피소드 종료 여부 tensor(num_envs,) — 실패로 끝남
        #   truncated:  시간 초과 여부 tensor(num_envs,) — 시간 제한으로 끝남
        #   info: 추가 정보 딕셔너리
        obs, rew, terminated, truncated, info = env.step(action)

        total_reward += rew
        total_steps += 1

        # ── 에피소드 종료 처리 ──────────────────────────────────────────
        # terminated 또는 truncated인 환경은 자동으로 리셋됩니다.
        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                episode_count[idx] += 1
                print(
                    f"[Step {total_steps:5d}] Env {idx.item()}: "
                    f"에피소드 {int(episode_count[idx].item())} 종료 | "
                    f"누적 보상: {total_reward[idx].item():.2f} | "
                    f"terminated={terminated[idx].item()}, "
                    f"truncated={truncated[idx].item()}"
                )
            total_reward[done] = 0.0

        # ── 주기적 상태 출력 ────────────────────────────────────────────
        if total_steps % 100 == 0:
            print(
                f"[Step {total_steps:5d}] "
                f"평균 보상: {rew.mean().item():+.4f} | "
                f"관측 (env 0): [{obs['policy'][0, 0]:.3f}, {obs['policy'][0, 1]:.3f}, "
                f"{obs['policy'][0, 2]:.3f}, {obs['policy'][0, 3]:.3f}]"
            )

        # 300 스텝 후 종료 (데모용)
        if total_steps >= 300:
            print(f"\n[INFO] {total_steps} 스텝 완료, 시뮬레이션 종료")
            break

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
