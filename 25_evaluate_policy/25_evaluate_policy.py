"""
25_evaluate_policy.py - 학습된 정책 평가 및 시각화

24강에서 학습한 PPO 체크포인트를 로드하여 정책을 평가합니다:
  - 체크포인트(model_*.pt) 로드
  - 학습된 정책으로 환경 실행 (시각화)
  - 에피소드 보상, 길이, 성공률 통계
  - 랜덤 정책과 학습 정책 성능 비교

실행:
    source env_isaaclab/bin/activate
    cd lectures/25_evaluate_policy

    # 학습된 정책 평가 (체크포인트 경로 지정)
    python 25_evaluate_policy.py --checkpoint /path/to/model_150.pt

    # 랜덤 정책으로 평가 (비교용)
    python 25_evaluate_policy.py --random

    # GUI로 시각화
    python 25_evaluate_policy.py --checkpoint /path/to/model_150.pt --num_envs 4
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os
import glob

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="25 - 학습된 정책 평가")
parser.add_argument("--num_envs", type=int, default=4, help="평가할 환경 수")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로")
parser.add_argument("--random", action="store_true", help="랜덤 정책으로 평가")
parser.add_argument("--num_episodes", type=int, default=20, help="평가할 에피소드 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
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


# ════════════════════════════════════════════════════════════════════��═════
#  커스텀 MDP 함수 (23, 24강과 동일)
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
#  환경 설정 (24강과 동일 구조)
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
class CartpoleEvalEnvCfg(ManagerBasedRLEnvCfg):
    scene: CartpoleSceneCfg = CartpoleSceneCfg(num_envs=4, env_spacing=4.0)
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
#  체크포인트 자동 탐색
# ══════════════════════════════════════════════════════════════════════════


def find_latest_checkpoint() -> str | None:
    """가장 최근 학습 체크포인트를 자동으로 찾습니다."""
    # 탐색할 경로 후보 (우선순위 순)
    search_paths = [
        # 24강과 같은 워크스페이스 루트에서 실행한 경우
        os.path.join(os.getcwd(), "logs", "rsl_rl"),
        # isaac 워크스페이스 루트
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs", "rsl_rl"),
        # 24강 디렉토리 기준
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "24_train_cartpole", "logs", "rsl_rl"),
    ]

    for log_dir in search_paths:
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            continue

        # 직접 model_*.pt가 있는 경우 (서브디렉토리 없이)
        direct_ckpts = sorted(glob.glob(os.path.join(log_dir, "model_*.pt")))
        if direct_ckpts:
            return direct_ckpts[-1]

        # 서브디렉토리 안에 model_*.pt가 있는 경우
        for sub in sorted(glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)):
            pass  # 마지막 값을 사용
        sub_ckpts = sorted(glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True))
        if sub_ckpts:
            return sub_ckpts[-1]

    return None


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """학습된 정책 또는 랜덤 정책으로 환경을 평가합니다."""

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env_cfg = CartpoleEvalEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # ── 정책 로드 (OnPolicyRunner로 24강과 동일하게) ─────────────────────
    policy = None
    use_random = args_cli.random

    if not use_random:
        checkpoint_path = args_cli.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 체크포인트: {checkpoint_path}")
            from isaaclab_rl.rsl_rl import (
                RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg,
                RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper,
            )
            from rsl_rl.runners import OnPolicyRunner

            # 24강과 동일한 에이전트 설정
            @configclass
            class EvalRunnerCfg(RslRlOnPolicyRunnerCfg):
                num_steps_per_env = 16
                max_iterations = 0
                save_interval = 50
                experiment_name = "cartpole_eval"
                run_name = ""
                logger = "tensorboard"
                policy = RslRlPpoActorCriticCfg(
                    init_noise_std=1.0,
                    actor_hidden_dims=[32, 32],
                    critic_hidden_dims=[32, 32],
                    activation="elu",
                )
                algorithm = RslRlPpoAlgorithmCfg(
                    value_loss_coef=1.0, use_clipped_value_loss=True,
                    clip_param=0.2, entropy_coef=0.005, num_learning_epochs=8,
                    num_mini_batches=4, learning_rate=3e-3, gamma=0.99, lam=0.95,
                    desired_kl=0.01, max_grad_norm=1.0,
                )

            agent_cfg = EvalRunnerCfg()
            env_wrapped = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(
                env_wrapped, agent_cfg.to_dict(),
                log_dir=None, device=str(env.device),
            )
            runner.load(checkpoint_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=str(env.device))
            # wrapped env를 평가 루프에서도 사용 (obs 형식 일치)
            eval_env = env_wrapped
            print("[INFO] 학습된 정책 로드 완료!")
        else:
            print("[WARNING] 체크포인트를 찾을 수 없습니다. 랜덤 정책으로 평가합니다.")
            print("  먼저 24강 학습을 실행하세요:")
            print("    python ../24_train_cartpole/24_train_cartpole.py --headless")
            use_random = True

    # wrapped env가 없으면 (랜덤 모드) 원래 env 사용
    if use_random:
        eval_env = None

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] {mode_name} 정책 평가 시작")
    print(f"  환경 수: {env.num_envs}")
    print(f"  평가 에피소드: {args_cli.num_episodes}")
    print(f"{'=' * 70}")

    # ── 평가 루프 ────────────────────────────────────────────────────────
    if use_random:
        obs, info = env.reset()
    else:
        obs = eval_env.get_observations()

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    all_rewards = []
    all_lengths = []
    total_steps = 0

    while simulation_app.is_running() and len(all_rewards) < args_cli.num_episodes:
        # 액션 선택
        if use_random:
            action = torch.randn(env.num_envs, 1, device=env.device).clamp(-1, 1)
            obs, rew, terminated, truncated, info = env.step(action)
        else:
            with torch.no_grad():
                action = policy(obs)
            obs, rew, dones, info = eval_env.step(action)
            truncated = info.get("time_outs", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            terminated = dones.bool() & ~truncated

        episode_rewards += rew.squeeze() if rew.dim() > 1 else rew
        episode_lengths += 1
        total_steps += 1

        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                ep_rew = episode_rewards[idx].item()
                ep_len = episode_lengths[idx].item()
                all_rewards.append(ep_rew)
                all_lengths.append(ep_len)

                reason = "timeout" if truncated[idx].item() else "fail"
                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {ep_rew:+8.2f} | 길이: {ep_len:4d} | {reason}"
                )

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0

    # ── 통계 출력 ────────────────────────────────────────────────────��───
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        avg_length = sum(all_lengths) / len(all_lengths)
        success_rate = sum(1 for l in all_lengths if l >= 290) / len(all_lengths) * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] {mode_name} 정책 평가 완료")
        print(f"  에피소드 수:    {len(all_rewards)}")
        print(f"  평균 보상:      {avg_reward:.2f}")
        print(f"  평균 길이:      {avg_length:.1f}")
        print(f"  최대 보상:      {max(all_rewards):.2f}")
        print(f"  최소 보상:      {min(all_rewards):.2f}")
        print(f"  성공률 (5초 생존): {success_rate:.1f}%")
        print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 평가 종료")
