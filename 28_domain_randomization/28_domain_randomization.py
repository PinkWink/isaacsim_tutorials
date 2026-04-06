"""
28_domain_randomization.py - 도메인 랜덤화 (Domain Randomization)

IsaacLab의 EventTermCfg를 사용하여 도메인 랜덤화(Domain Randomization)를 구현합니다.
시뮬레이션과 실제 환경의 차이(sim-to-real gap)를 줄이기 위한 핵심 기술입니다:
  - "startup" 모드: 학습 시작 시 1회 실행 (예: 질량 랜덤화)
  - "reset" 모드: 매 에피소드 리셋 시 실행 (예: 초기 상태 랜덤화)
  - "interval" 모드: 에피소드 진행 중 일정 간격으로 실행 (예: 외력, 속도 교란)

이것은 시리즈의 마지막 강의(28강)입니다.
20~25강에서 구축한 Cartpole 환경에 다양한 도메인 랜덤화를 적용합니다.

실행:
    source env_isaaclab/bin/activate
    cd lectures/28_domain_randomization
    python 28_domain_randomization.py

    # GUI 없이 실행
    python 28_domain_randomization.py --headless

    # 환경 수 변경
    python 28_domain_randomization.py --num_envs 16
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
# IsaacLab의 모든 스크립트는 반드시 AppLauncher를 먼저 초기화해야 합니다.
# Omniverse Kit 앱이 초기화된 후에야 시뮬레이션 관련 모듈을 import 할 수 있습니다.
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="28 - 도메인 랜덤화 (Domain Randomization)")
parser.add_argument("--num_envs", type=int, default=4, help="병렬 환경 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
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
# 도메인 랜덤화에 사용할 이벤트 함수들도 여기에 포함되어 있습니다:
#   - randomize_rigid_body_mass: 강체 질량 랜덤화 (class-based term)
#   - reset_joints_by_offset: 관절 상태를 오프셋으로 초기화
#   - push_by_setting_velocity: 루트 속도를 설정하여 밀기(push) 효과
#   - apply_external_force_torque: 외력/외부 토크 적용
import isaaclab.envs.mdp as mdp

# ── 4. 사전 정의된 Cartpole 로봇 설정 ────────────────────────────────────
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  Scene 설정 — Cartpole + 바닥 + 조명
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Cartpole 환경의 씬 설정.

    20강에서 사용한 것과 동일한 구성입니다:
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
    # {ENV_REGEX_NS}: 각 환경마다 별도의 인스턴스를 생성하는 네임스페이스 패턴
    robot: ArticulationCfg = CARTPOLE_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── 조명 ─────────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ══════════════════════════════════════════════════════════════════════════
#  Action 설정 — 카트에 힘 가하기
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """에이전트의 액션 스펙.

    Cartpole에서 에이전트는 카트(slider_to_cart)에 좌/우 힘을 가합니다.
    scale=100.0으로 [-1, 1] 출력을 [-100, 100]N 범위로 변환합니다.
    """
    joint_effort = mdp.JointEffortActionCfg(
        asset_name="robot",
        joint_names=["slider_to_cart"],
        scale=100.0,
    )


# ══════════════════════════════════════════════════════════════════════════
#  Observation 설정 — 관측값 정의
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ObservationsCfg:
    """관측(Observation) 스펙.

    정책 네트워크에 입력되는 관측값:
      - joint_pos_rel: 관절 위치 (2차원: cart, pole)
      - joint_vel_rel: 관절 속도 (2차원: cart, pole)
    총 4차원의 관측 벡터를 생성합니다.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self) -> None:
            self.enable_corruption = False  # 관측 노이즈 비활성화
            self.concatenate_terms = True   # 모든 텀을 concat → (num_envs, 4)

    policy: PolicyCfg = PolicyCfg()


# ══════════════════════════════════════════════════════════════════════════
#  ★ Event 설정 — 도메인 랜덤화의 핵심! ★
# ══════════════════════════════════════════════════════════════════════════
#
# EventTermCfg에는 3가지 mode가 있습니다:
#
#   1. "startup" — 학습 시작 시 딱 1회 실행
#      → 물리적 속성(질량, 마찰 등) 랜덤화에 사용
#      → 환경마다 다른 물리 특성을 갖게 하여 다양한 조건에서 학습
#
#   2. "reset" — 매 에피소드 리셋 시 실행
#      → 초기 상태(관절 위치/속도) 랜덤화에 사용
#      → 에피소드마다 다른 시작 조건에서 학습
#
#   3. "interval" — 에피소드 진행 중 일정 간격으로 실행
#      → 외부 교란(외력, 속도 변화) 적용에 사용
#      → 예측 불가능한 교란에 대한 강건성 학습
#
# ══════════════════════════════════════════════════════════════════════════


@configclass
class DomainRandEventCfg:
    """도메인 랜덤화 이벤트 설정.

    이 설정은 세 가지 모드의 이벤트를 모두 포함하여
    도메인 랜덤화의 전체적인 흐름을 보여줍니다.
    """

    # ──────────────────────────────────────────────────────────────────
    # [startup] 학습 시작 시 1회 실행 — 물리 속성 랜덤화
    # ──────────────────────────────────────────────────────────────────
    #
    # randomize_rigid_body_mass는 class-based EventTerm입니다.
    # 로봇의 각 바디(cart, pole)의 질량에 랜덤 값을 더합니다.
    # 이를 통해 각 병렬 환경마다 서로 다른 질량을 갖게 됩니다.
    #
    # 매개변수 설명:
    #   - asset_cfg: 랜덤화할 로봇과 바디 지정 (".*"은 모든 바디)
    #   - mass_distribution_params: 랜덤 값의 범위 (min, max)
    #   - operation: "add" (기존 질량에 더하기), "scale" (배율), "abs" (절대값 설정)
    #
    # 예: 기본 질량이 1.0kg인 바디에 (-0.5, 0.5) add 적용
    #     → 각 환경에서 0.5~1.5kg 사이의 질량을 갖게 됨
    #
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (-0.5, 0.5),
            "operation": "add",
        },
    )

    # ──────────────────────────────────────────────────────────────────
    # [reset] 매 에피소드 리셋 시 실행 — 초기 상태 랜덤화
    # ──────────────────────────────────────────────────────────────────
    #
    # reset_joints_by_offset는 기본 관절 위치/속도에 랜덤 오프셋을 더합니다.
    # 매 에피소드 시작 시 서로 다른 초기 상태에서 시작하게 하여
    # 정책이 다양한 상황에 대응할 수 있도록 합니다.
    #
    # 카트 초기 위치: 기본 위치 ± 1.0m 범위에서 랜덤
    # 카트 초기 속도: ± 0.5 m/s 범위에서 랜덤
    #
    reset_cart = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "position_range": (-1.0, 1.0),
            "velocity_range": (-0.5, 0.5),
        },
    )

    # 폴 초기 각도: 기본 각도 ± 45° (0.25π) 범위에서 랜덤
    # 폴 초기 각속도: ± 45°/s 범위에서 랜덤
    reset_pole = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
            "position_range": (-0.25 * math.pi, 0.25 * math.pi),
            "velocity_range": (-0.25 * math.pi, 0.25 * math.pi),
        },
    )

    # ──────────────────────────────────────────────────────────────────
    # [interval] 에피소드 진행 중 일정 간격으로 실행 — 외부 교란
    # ──────────────────────────────────────────────────────────────────
    #
    # push_by_setting_velocity는 로봇의 루트 속도에 랜덤 값을 더합니다.
    # 마치 외부에서 로봇을 밀어(push)내는 것과 같은 효과입니다.
    #
    # 매개변수 설명:
    #   - velocity_range: 각 축별 속도 범위를 딕셔너리로 지정
    #     {"x": (-0.5, 0.5)} → x 방향으로 ±0.5 m/s의 랜덤 속도 교란
    #   - interval_range_s: 이벤트 실행 간격 범위 (초)
    #     (10.0, 15.0) → 10~15초 사이의 랜덤 간격으로 push 적용
    #
    # 실제 로봇에서는 사람이 로봇을 밀거나, 바람, 충격 등이 발생할 수 있습니다.
    # interval 이벤트로 이런 예측 불가능한 교란을 시뮬레이션합니다.
    #
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {"x": (-0.5, 0.5)},
        },
    )

    # apply_external_force_torque는 로봇 바디에 지속적인 외력/외부 토크를 적용합니다.
    # push_by_setting_velocity와 달리 힘이 매 시뮬레이션 스텝마다 적용됩니다.
    #
    # 매개변수 설명:
    #   - force_range: 외력의 범위 (N)
    #   - torque_range: 외부 토크의 범위 (Nm)
    #   - interval_range_s: 새로운 랜덤 외력을 샘플링하는 간격 (초)
    #
    # 예: 바람이 주기적으로 방향을 바꾸는 것을 시뮬레이션
    #
    apply_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*pole"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-0.1, 0.1),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
#  Reward 설정 — 보상 함수
# ══════════════════════════════════════════════════════════════════════════


@configclass
class RewardsCfg:
    """보상 설정 (20강과 동일)."""
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


# ══════════════════════════════════════════════════════════════════════════
#  Termination 설정 — 에피소드 종료 조건
# ══════════════════════════════════════════════════════════════════════════


@configclass
class TerminationsCfg:
    """에피소드 종료 조건 (20강과 동일)."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
            "bounds": (-3.0, 3.0),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
#  전체 환경 설정 — ManagerBasedRLEnvCfg
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleDomainRandEnvCfg(ManagerBasedRLEnvCfg):
    """도메인 랜덤화가 적용된 Cartpole RL 환경.

    핵심은 events 필드에 DomainRandEventCfg를 사용한다는 것입니다.
    startup/reset/interval 세 가지 모드의 이벤트가 모두 포함되어 있어,
    환경이 생성되고 학습이 진행되는 동안 다양한 시점에서 랜덤화가 적용됩니다.
    """
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: DomainRandEventCfg = DomainRandEventCfg()  # ★ 도메인 랜덤화 이벤트!
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """환경 초기화 후 추가 설정."""
        self.decimation = 2            # 정책 1회당 물리 2회
        self.episode_length_s = 5.0    # 에피소드 최대 5초
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0     # 물리 120Hz
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """도메인 랜덤화가 적용된 Cartpole 환경을 생성하고 실행합니다."""

    # ── 환경 설정 생성 ──────────────────────────────────────────────────
    env_cfg = CartpoleDomainRandEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── 환경 생성 ────────────────────────────────────────────────────────
    # ManagerBasedRLEnv가 생성될 때 내부적으로 EventManager가 초기화됩니다.
    # 이때 mode="startup"인 이벤트(randomize_mass)가 자동으로 1회 실행됩니다.
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── 환경 정보 출력 ──────────────────────────────────────────────────
    print("=" * 70)
    print("[INFO] 도메인 랜덤화 Cartpole 환경 생성 완료!")
    print(f"  환경 수:         {env.num_envs}")
    print(f"  관측 차원:       {env.observation_space}")
    print(f"  액션 차원:       {env.action_space}")
    print(f"  에피소드 길이:    {env_cfg.episode_length_s}초")
    print(f"  decimation:     {env_cfg.decimation}")
    print(f"  물리 dt:         {env_cfg.sim.dt}초")
    print("=" * 70)

    # ── 이벤트 매니저 정보 출력 ──────────────────────────────────────────
    # EventManager에 등록된 이벤트 텀들을 모드별로 확인합니다.
    event_manager = env.event_manager
    print("\n[EVENT MANAGER] 등록된 이벤트 텀:")

    # 각 모드별 이벤트 텀 이름 출력
    active = event_manager.active_terms
    for mode_name in ["startup", "reset", "interval"]:
        terms = active.get(mode_name, {})
        if terms:
            term_names = list(terms.keys()) if isinstance(terms, dict) else [t for t in terms]
            print(f"  [{mode_name:>8}] {len(term_names)}개: {term_names}")
        else:
            print(f"  [{mode_name:>8}] 없음")

    # ── startup 이벤트 결과 확인: 질량 랜덤화 ──────────────────────────
    # startup 모드의 randomize_mass가 이미 실행되었으므로,
    # 각 환경의 로봇 질량이 서로 다를 것입니다.
    robot = env.scene["robot"]
    masses = robot.root_physx_view.get_masses()
    print("\n[STARTUP] 질량 랜덤화 결과 (환경별 바디 질량):")
    for i in range(min(env.num_envs, 4)):
        mass_str = ", ".join(f"{m:.3f}" for m in masses[i].tolist())
        print(f"  Env {i}: [{mass_str}] kg")
    print("  → 각 환경마다 질량이 다르게 설정됨 (startup 이벤트 결과)")

    # ── 환경 리셋 (첫 관측 획득) ────────────────────────────────────────
    # env.reset()을 호출하면 mode="reset"인 이벤트가 실행됩니다.
    # (reset_cart, reset_pole)
    print("\n" + "-" * 70)
    print("[RESET] 환경 리셋 실행 — reset 이벤트가 트리거됩니다...")
    obs, info = env.reset()
    print(f"  초기 관측 shape: {obs['policy'].shape}")
    for i in range(min(env.num_envs, 4)):
        obs_vals = obs["policy"][i].tolist()
        obs_str = ", ".join(f"{v:+.4f}" for v in obs_vals)
        print(f"  Env {i}: [{obs_str}]")
    print("  → 에피소드마다 카트 위치, 폴 각도가 다르게 초기화됨 (reset 이벤트 결과)")

    # ── 환경 실행 루프 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[RUN] 랜덤 액션으로 환경 실행 시작...")
    print("  interval 이벤트 (push_robot, apply_force)가 주기적으로 트리거됩니다.")
    print("=" * 70)

    total_steps = 0
    total_reward = torch.zeros(env.num_envs, device=env.device)
    episode_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
    step_in_episode = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)

    while simulation_app.is_running():
        # ── 랜덤 액션 생성 ──────────────────────────────────────────────
        # 실제 학습에서는 정책 네트워크가 액션을 생성하지만,
        # 여기서는 도메인 랜덤화의 효과를 확인하기 위해 랜덤 액션을 사용합니다.
        action = torch.randn(env.num_envs, 1, device=env.device)

        # ── env.step() — 이 안에서 interval 이벤트가 트리거될 수 있음 ──
        # env.step() 내부에서 EventManager가 interval_range_s 간격을 체크하고,
        # 조건이 맞으면 push_robot, apply_force 이벤트를 자동으로 실행합니다.
        obs, rew, terminated, truncated, info = env.step(action)

        total_reward += rew
        total_steps += 1
        step_in_episode += 1

        # ── 에피소드 종료 처리 ──────────────────────────────────────────
        # 에피소드가 종료되면 자동으로 리셋되면서 reset 이벤트가 다시 실행됩니다.
        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                episode_count[idx] += 1
                print(
                    f"[Step {total_steps:5d}] Env {idx.item()}: "
                    f"에피소드 {int(episode_count[idx].item())} 종료 | "
                    f"길이: {int(step_in_episode[idx].item())} 스텝 | "
                    f"누적 보상: {total_reward[idx].item():.2f} | "
                    f"terminated={terminated[idx].item()}, "
                    f"truncated={truncated[idx].item()}"
                )
                # 종료된 환경의 reset 이벤트는 env.step() 내부에서 자동 실행됨
                step_in_episode[idx] = 0
            total_reward[done] = 0.0

        # ── 주기적 상태 출력 ────────────────────────────────────────────
        if total_steps % 100 == 0:
            elapsed_sim_time = total_steps * env_cfg.decimation * env_cfg.sim.dt
            print(
                f"[Step {total_steps:5d}] "
                f"시뮬레이션 시간: {elapsed_sim_time:.1f}초 | "
                f"평균 보상: {rew.mean().item():+.4f} | "
                f"완료된 에피소드: {int(episode_count.sum().item())}"
            )

        # 500 스텝 후 종료 (데모용)
        if total_steps >= 500:
            elapsed_sim_time = total_steps * env_cfg.decimation * env_cfg.sim.dt
            print(f"\n[INFO] {total_steps} 스텝 완료 (시뮬레이션 시간: {elapsed_sim_time:.1f}초)")
            break

    # ── 최종 통계 출력 ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[SUMMARY] 도메인 랜덤화 실행 결과:")
    print(f"  총 스텝 수:        {total_steps}")
    print(f"  완료된 에피소드:    {int(episode_count.sum().item())}")
    print(f"  환경별 에피소드:    {episode_count.tolist()}")
    print()
    print("  적용된 도메인 랜덤화:")
    print("    [startup]  질량 랜덤화     — 환경마다 다른 질량")
    print("    [reset]    초기 상태 랜덤화 — 에피소드마다 다른 시작 조건")
    print("    [interval] 속도 교란(push)  — 10~15초 간격으로 밀기")
    print("    [interval] 외력 적용        — 5~10초 간격으로 외력")
    print()
    print("  이러한 랜덤화는 sim-to-real transfer에 필수적입니다.")
    print("  시뮬레이션에서 다양한 조건에 노출된 정책은")
    print("  실제 환경의 불확실성에 더 강건하게 대응합니다.")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
