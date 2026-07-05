"""
24_train_cartpole.py - Cartpole PPO 학습

23강에서 만든 커스텀 Cartpole 환경에 RSL-RL의 PPO를 적용하여 학습합니다:
  - RslRlVecEnvWrapper로 환경을 RSL-RL 인터페이스에 연결
  - RslRlOnPolicyRunnerCfg로 PPO 하이퍼파라미터 설정
  - OnPolicyRunner로 학습 실행 및 체크포인트 저장
  - TensorBoard로 학습 과정 모니터링

실행:
    source ~/isaac/env_isaaclab/bin/activate
    cd ~/isaac/isaacsim_tutorials/24_train_cartpole
    python 24_train_cartpole.py

    # GUI 없이 학습 (권장)
    python 24_train_cartpole.py --headless --num_envs 1024

    # 학습 반복 횟수 변경
    python 24_train_cartpole.py --headless --num_envs 1024 --max_iterations 300

    # 학습 결과 확인 (TensorBoard)
    tensorboard --logdir logs/rsl_rl/cartpole_tutorial
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="24 - Cartpole PPO 학습")
parser.add_argument("--num_envs", type=int, default=1024, help="병렬 환경 수")
parser.add_argument("--max_iterations", type=int, default=150, help="학습 반복 횟수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
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

# RSL-RL 관련 import
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_assets.robots.cartpole import CARTPOLE_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수 (23강에서 가져옴)
# ══════════════════════════════════════════════════════════════════════════


def obs_pole_sin_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    angle = robot.data.joint_pos[:, joint_ids[0]]
    return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)


def obs_cart_normalized(env: ManagerBasedRLEnv, max_pos: float = 3.0) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    pos = robot.data.joint_pos[:, joint_ids[0]]
    return (pos / max_pos).unsqueeze(-1)


def rew_pole_upright(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    return torch.cos(robot.data.joint_pos[:, joint_ids[0]])


def rew_cart_center(env: ManagerBasedRLEnv, sigma: float = 1.0) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("slider_to_cart")
    pos = robot.data.joint_pos[:, joint_ids[0]]
    return torch.exp(-pos**2 / (2.0 * sigma**2))


def rew_energy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(env.action_manager.action**2, dim=-1)


def term_pole_too_far(env: ManagerBasedRLEnv, max_angle: float = 0.5 * math.pi) -> torch.Tensor:
    robot = env.scene["robot"]
    joint_ids, _ = robot.find_joints("cart_to_pole")
    return torch.abs(robot.data.joint_pos[:, joint_ids[0]]) > max_angle


# ══════════════════════════════════════════════════════════════════════════
#  환경 설정 (23강 MyCartpoleEnvCfg와 동일)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)))
    robot: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(prim_path="/World/DomeLight", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0))


@configclass
class ActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=["slider_to_cart"], scale=100.0)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        pole_sin_cos = ObsTerm(func=obs_pole_sin_cos)
        cart_norm = ObsTerm(func=obs_cart_normalized, params={"max_pos": 3.0}, clip=(-1.0, 1.0))
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_cart = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]),
                "position_range": (-1.0, 1.0), "velocity_range": (-0.5, 0.5)})
    reset_pole = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"]),
                "position_range": (-0.25 * math.pi, 0.25 * math.pi),
                "velocity_range": (-0.25 * math.pi, 0.25 * math.pi)})


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)
    terminating = RewTerm(func=mdp.is_terminated, weight=-2.0)
    pole_upright = RewTerm(func=rew_pole_upright, weight=2.0)
    cart_center = RewTerm(func=rew_cart_center, weight=0.5, params={"sigma": 1.5})
    energy = RewTerm(func=rew_energy, weight=-0.001)
    pole_vel = RewTerm(func=mdp.joint_vel_l1, weight=-0.005,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["cart_to_pole"])})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    cart_out_of_bounds = DoneTerm(func=mdp.joint_pos_out_of_manual_limit,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["slider_to_cart"]), "bounds": (-3.0, 3.0)})
    pole_too_far = DoneTerm(func=term_pole_too_far, params={"max_angle": 0.5 * math.pi})


@configclass
class CartpoleTrainEnvCfg(ManagerBasedRLEnvCfg):
    """학습용 Cartpole 환경 설정."""
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=1024, env_spacing=4.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.viewer.eye = (8.0, 0.0, 5.0)
        self.sim.dt = 1.0 / 120.0
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
#  RSL-RL PPO 하이퍼파라미터 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class CartpolePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Cartpole PPO 학습 설정.

    하이퍼파라미터 설명:
      - num_steps_per_env: 업데이트 전 각 환경에서 수집할 스텝 수
      - max_iterations: 총 학습 반복 횟수
      - actor_hidden_dims: 정책 네트워크 레이어 크기
      - learning_rate: 학습률
      - gamma: 할인율 (미래 보상의 현재 가치)
      - lam: GAE lambda (advantage 추정의 편향-분산 트레이드오프)
    """
    num_steps_per_env = 16       # 업데이트 전 수집 스텝 (16 × 1024 = 16384 샘플/업데이트)
    max_iterations = 150         # 학습 반복 (Cartpole은 150이면 충분)
    save_interval = 50           # 체크포인트 저장 주기
    experiment_name = "cartpole_tutorial"
    run_name = ""
    logger = "tensorboard"

    # 정책 네트워크 설정 (Actor-Critic)
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,               # 초기 탐색 노이즈 (높을수록 많이 탐색)
        actor_hidden_dims=[32, 32],       # 정책(액터) 네트워크: 2층 × 32 뉴런
        critic_hidden_dims=[32, 32],      # 가치(크리틱) 네트워크: 2층 × 32 뉴런
        activation="elu",                 # 활성화 함수
    )

    # PPO 알고리즘 설정
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,              # 가치 함수 손실 가중치
        use_clipped_value_loss=True,      # 가치 손실 클리핑
        clip_param=0.2,                   # PPO 클리핑 범위 ε
        entropy_coef=0.005,               # 엔트로피 보너스 (탐색 장려)
        num_learning_epochs=5,            # 수집 데이터로 학습 반복 횟수
        num_mini_batches=4,               # 미니배치 수
        learning_rate=1.0e-3,             # 학습률
        schedule="adaptive",             # 적응적 학습률 (KL 기반)
        gamma=0.99,                       # 할인율
        lam=0.95,                         # GAE lambda
        desired_kl=0.01,                  # 목표 KL divergence
        max_grad_norm=1.0,                # gradient 클리핑
    )


# ══════════════════════════════════════════════════════════════════════════
#  메인 학습 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Cartpole PPO 학습을 실행합니다."""

    # ── 환경 설정 ────────────────────────────────────────────────────────
    env_cfg = CartpoleTrainEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── 에이전트 설정 ────────────────────────────────────────────────────
    agent_cfg = CartpolePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    # ── 로그 디렉토리 설정 ───────────────────────────────────────────────
    # OnPolicyRunner는 log_dir에 바로 체크포인트/TensorBoard 로그를 기록하므로
    # experiment_name까지 포함한 경로를 만들어 전달합니다.
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 로그 경로: {log_root_path}")

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print("[INFO] 학습 환경 생성 완료!")
    print(f"  환경 수:       {env.num_envs}")
    print(f"  관측 차원:     {env.observation_space}")
    print(f"  액션 차원:     {env.action_space}")
    print(f"  학습 반복:     {agent_cfg.max_iterations}")
    print(f"  스텝/업데이트:  {agent_cfg.num_steps_per_env} × {env.num_envs} = "
          f"{agent_cfg.num_steps_per_env * env.num_envs}")
    print("=" * 70)

    # ── RSL-RL 환경 래퍼 ─────────────────────────────────────────────────
    # IsaacLab 환경 → RSL-RL VecEnv 인터페이스로 변환
    env = RslRlVecEnvWrapper(env)

    # ── PPO Runner 생성 ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device=agent_cfg.device)

    # ── 학습 실행 ────────────────────────────────────────────────────────
    print("\n[INFO] PPO 학습 시작!")
    print(f"  네트워크: Actor {agent_cfg.policy.actor_hidden_dims}, "
          f"Critic {agent_cfg.policy.critic_hidden_dims}")
    print(f"  학습률: {agent_cfg.algorithm.learning_rate}")
    print(f"  할인율(γ): {agent_cfg.algorithm.gamma}")
    print(f"  PPO 클리핑(ε): {agent_cfg.algorithm.clip_param}")
    print("=" * 70)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── 학습 완료 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[INFO] 학습 완료!")
    print(f"  체크포인트 저장 위치: {log_root_path}")
    print("  TensorBoard 확인: tensorboard --logdir logs/rsl_rl/cartpole_tutorial")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 학습 종료")
