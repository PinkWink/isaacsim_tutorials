"""
27_evaluate_locomotion.py - 학습된 ANYmal 보행 정책 추론

26강에서 학습한 PPO 체크포인트를 로드하여 ANYmal 로봇의 보행을 추론합니다:
  - 체크포인트(model_*.pt) 로드 (자동 탐색 또는 직접 지정)
  - 학습된 정책으로 환경 실행 (시각화)
  - 에피소드 보상, 길이 통계
  - 랜덤 정책과의 성능 비교 옵션

25강(Cartpole 평가)과의 차이:
  - gym.make()로 사전 등록된 환경을 사용 (26강과 동일)
  - 네트워크가 더 큼 ([128,128,128]) → Cfg가 학습 시와 일치해야 함
  - 액션 차원 12 (4족 × 3관절)
  - 관측에는 속도 명령(velocity command)이 포함됨

실행:
    source env_isaaclab/bin/activate
    cd lectures/27_evaluate_locomotion

    # 가장 최근 체크포인트 자동 탐색
    python 27_evaluate_locomotion.py

    # 체크포인트 직접 지정
    python 27_evaluate_locomotion.py --checkpoint /path/to/model_299.pt

    # 랜덤 정책으로 평가 (비교용)
    python 27_evaluate_locomotion.py --random

    # 환경 수 변경 (GUI 시각화 시 적게)
    python 27_evaluate_locomotion.py --num_envs 16

    # Headless로 빠르게 통계만 수집
    python 27_evaluate_locomotion.py --headless --num_envs 64 --num_episodes 50
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="27 - 학습된 ANYmal 보행 정책 추론")
parser.add_argument("--num_envs", type=int, default=16,
                    help="평가할 환경 수 (GUI는 16 권장)")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="체크포인트(.pt) 경로. 미지정 시 자동 탐색")
parser.add_argument("--random", action="store_true",
                    help="랜덤 정책으로 평가 (학습 정책과 비교용)")
parser.add_argument("--num_episodes", type=int, default=20,
                    help="평가할 에피소드 수")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-B-v0",
                    help="등록된 환경 ID (학습 시와 동일해야 함)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401  (사전 등록 환경 활성화)

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


# ══════════════════════════════════════════════════════════════════════════
#  추론용 Runner 설정 (26강 학습 설정과 네트워크 형상이 일치해야 함)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class AnymalEvalRunnerCfg(RslRlOnPolicyRunnerCfg):
    """ANYmal 추론용 Runner 설정.

    중요: actor/critic_hidden_dims은 26강 학습 설정과 동일해야 합니다.
    형상이 다르면 state_dict 로드에서 에러가 발생합니다.
    """
    num_steps_per_env = 24
    max_iterations = 0           # 추론이므로 학습 반복은 0
    save_interval = 50
    experiment_name = "anymal_locomotion_eval"
    run_name = ""
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],   # 26강과 동일
        critic_hidden_dims=[128, 128, 128],  # 26강과 동일
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


# ══════════════════════════════════════════════════════════════════════════
#  체크포인트 자동 탐색
# ══════════════════════════════════════════════════════════════════════════


def find_latest_checkpoint() -> str | None:
    """26강에서 저장된 가장 최근 체크포인트를 자동으로 찾습니다.

    탐색 순서:
      1. 현재 작업 디렉토리의 logs/rsl_rl/anymal_locomotion_tutorial
      2. 워크스페이스 루트(스크립트 기준 ../..)의 동일 경로
      3. 26강 디렉토리 옆의 logs (혹시 26강 디렉토리에서 실행한 경우)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    exp_name = "anymal_locomotion_tutorial"

    search_paths = [
        os.path.join(os.getcwd(), "logs", "rsl_rl", exp_name),
        os.path.join(script_dir, "..", "logs", "rsl_rl", exp_name),
        os.path.join(script_dir, "..", "26_train_locomotion", "logs", "rsl_rl", exp_name),
    ]

    for log_dir in search_paths:
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            continue

        # 직접 model_*.pt가 있는 경우
        direct_ckpts = sorted(
            glob.glob(os.path.join(log_dir, "model_*.pt")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
        if direct_ckpts:
            return direct_ckpts[-1]

        # 서브디렉토리 안에 model_*.pt가 있는 경우
        sub_ckpts = sorted(
            glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True),
            key=lambda p: (os.path.dirname(p), int(os.path.basename(p).split("_")[1].split(".")[0])),
        )
        if sub_ckpts:
            return sub_ckpts[-1]

    return None


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """학습된 ANYmal 보행 정책으로 추론하거나 랜덤 정책으로 평가합니다."""

    # ── 환경 설정 로드 ───────────────────────────────────────────────────
    task_name = args_cli.task
    print(f"[INFO] 태스크: {task_name}")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    # 에피소드를 너무 일찍 끊지 않도록 최소 길이 유지
    # (학습용 cfg를 그대로 사용 — 보행은 보통 20초 에피소드)

    # ── 환경 생성 (26강과 동일하게 gym.make 사용) ────────────────────────
    env = gym.make(task_name, cfg=env_cfg)
    unwrapped = env.unwrapped

    print("=" * 70)
    print("[INFO] ANYmal 평가 환경 생성 완료!")
    print(f"  태스크:    {task_name}")
    print(f"  환경 수:   {unwrapped.num_envs}")
    print(f"  관측 차원: {unwrapped.observation_space}")
    print(f"  액션 차원: {unwrapped.action_space}")
    print("=" * 70)

    # ── RSL-RL 래퍼 ──────────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)
    device = str(unwrapped.device)

    # ── 정책 로드 ────────────────────────────────────────────────────────
    policy = None
    use_random = args_cli.random

    if not use_random:
        checkpoint_path = args_cli.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 체크포인트: {checkpoint_path}")

            agent_cfg = AnymalEvalRunnerCfg()
            runner = OnPolicyRunner(
                env, agent_cfg.to_dict(),
                log_dir=None, device=device,
            )
            runner.load(checkpoint_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            print("[INFO] 학습된 정책 로드 완료!")
        else:
            print("[WARNING] 체크포인트를 찾을 수 없습니다. 랜덤 정책으로 평가합니다.")
            print("  먼저 26강 학습을 실행하세요:")
            print("    python ../26_train_locomotion/26_train_locomotion.py --headless")
            use_random = True

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] {mode_name} 정책 평가 시작")
    print(f"  환경 수:       {unwrapped.num_envs}")
    print(f"  평가 에피소드: {args_cli.num_episodes}")
    print(f"{'=' * 70}\n")

    # ── 평가 루프 ────────────────────────────────────────────────────────
    # RslRlVecEnvWrapper는 gym.make 직후 자동으로 reset된 상태이므로
    # get_observations()로 초기 관측을 바로 받아올 수 있습니다.
    obs = env.get_observations()

    num_envs = unwrapped.num_envs
    action_dim = unwrapped.action_space.shape[-1]

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

    all_rewards: list[float] = []
    all_lengths: list[int] = []

    while simulation_app.is_running() and len(all_rewards) < args_cli.num_episodes:
        if use_random:
            # 보행 액션은 일반적으로 [-1, 1] 범위, 정규분포로 흉내
            action = torch.randn(num_envs, action_dim, device=device).clamp(-1.0, 1.0)
        else:
            with torch.no_grad():
                action = policy(obs)

        obs, rew, dones, info = env.step(action)
        truncated = info.get("time_outs", torch.zeros(num_envs, dtype=torch.bool, device=device))
        terminated = dones.bool() & ~truncated

        episode_rewards += rew.squeeze() if rew.dim() > 1 else rew
        episode_lengths += 1

        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                ep_rew = float(episode_rewards[idx].item())
                ep_len = int(episode_lengths[idx].item())
                all_rewards.append(ep_rew)
                all_lengths.append(ep_len)

                reason = "timeout" if bool(truncated[idx].item()) else "fail"
                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {ep_rew:+9.2f} | 길이: {ep_len:4d} | {reason}"
                )

                if len(all_rewards) >= args_cli.num_episodes:
                    break

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0

    # ── 통계 출력 ────────────────────────────────────────────────────────
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        avg_length = sum(all_lengths) / len(all_lengths)
        # 보행에서는 "timeout까지 살아남음"이 성공 지표에 가까움
        # 기본 에피소드 길이가 20초 × (1/dt/decimation) 정도이므로
        # 평균 길이의 90% 이상을 성공으로 간주
        max_len = max(all_lengths)
        success_threshold = int(max_len * 0.9)
        success_rate = sum(1 for l in all_lengths if l >= success_threshold) / len(all_lengths) * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] {mode_name} 정책 평가 완료")
        print(f"  에피소드 수:        {len(all_rewards)}")
        print(f"  평균 보상:          {avg_reward:.2f}")
        print(f"  평균 길이:          {avg_length:.1f} 스텝")
        print(f"  최대/최소 보상:     {max(all_rewards):.2f} / {min(all_rewards):.2f}")
        print(f"  생존률 (≥{success_threshold}스텝): {success_rate:.1f}%")
        print(f"{'=' * 70}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 보행 정책 추론 종료")
