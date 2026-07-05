"""
35_3_play_standing.py - 학습된 G1 Standing 정책 평가 및 시각화

35_2에서 학습한 PPO 체크포인트를 로드하여 G1이 스스로 서 있는 모습을
확인하고 성능을 평가합니다:
  - 체크포인트(model_*.pt) 자동 탐색 및 로드
  - 학습된 정책으로 환경 실행 (시각화)
  - 에피소드 보상, 길이, 생존율 통계
  - 랜덤 정책과 비교 (--random)

실행:
    source env_isaaclab/bin/activate
    cd lectures/35_g1_standing

    # 최신 체크포인트 자동 탐색 후 평가
    python 35_3_play_standing.py

    # 체크포인트 직접 지정
    python 35_3_play_standing.py --checkpoint logs/rsl_rl/g1_standing_tutorial/model_1000.pt

    # 랜덤 정책으로 평가 (학습 전후 비교용)
    python 35_3_play_standing.py --random
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import glob
import os
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="35_3 - G1 Standing 정책 평가")
parser.add_argument("--num_envs", type=int, default=4, help="평가할 환경 수")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로")
parser.add_argument("--random", action="store_true", help="랜덤 정책으로 평가")
parser.add_argument("--num_episodes", type=int, default=10, help="평가할 에피소드 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv

# 공유 환경 모듈 (같은 폴더의 g1_standing_env.py)
# 학습(35_2)과 동일한 관측/보상/네트워크 정의를 써야 체크포인트가 정상 동작합니다.
from g1_standing_env import G1StandingEnvCfg, make_ppo_runner_cfg


# ══════════════════════════════════════════════════════════════════════════
#  체크포인트 자동 탐색
# ══════════════════════════════════════════════════════════════════════════


def find_latest_checkpoint() -> str | None:
    """가장 최근 학습 체크포인트를 자동으로 찾습니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 탐색할 경로 후보 (우선순위 순)
    search_paths = [
        # 35_2와 같은 위치에서 실행한 경우
        os.path.join(os.getcwd(), "logs", "rsl_rl", "g1_standing_tutorial"),
        # 이 스크립트 폴더 기준
        os.path.join(script_dir, "logs", "rsl_rl", "g1_standing_tutorial"),
        # 워크스페이스 루트 기준
        os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "g1_standing_tutorial"),
    ]

    def iteration_number(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    for log_dir in search_paths:
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            continue
        ckpts = glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)
        if ckpts:
            # model_1000 > model_950이 되도록 iteration 번호 기준 정렬 (사전순 정렬은 오답)
            return max(ckpts, key=iteration_number)

    return None


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """학습된 정책 또는 랜덤 정책으로 G1 Standing을 평가합니다."""

    # ── 환경 생성 (35_2와 동일 환경, 개수만 축소) ────────────────────────
    env_cfg = G1StandingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 에피소드 최대 길이 (스텝 수): 5초 / (0.005 × 4) = 250
    max_episode_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    # ── 정책 로드 (OnPolicyRunner로 35_2와 동일하게) ─────────────────────
    policy = None
    eval_env = None
    use_random = args_cli.random

    if not use_random:
        checkpoint_path = args_cli.checkpoint
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 체크포인트: {checkpoint_path}")
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            # 35_2와 동일한 에이전트 설정 (네트워크 형상이 다르면 로드 실패)
            agent_cfg = make_ppo_runner_cfg(max_iterations=0)

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
            print("  먼저 35_2 학습을 실행하세요:")
            print("    python 35_2_train_standing.py --headless")
            use_random = True

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] {mode_name} 정책 평가 시작")
    print(f"  환경 수: {env.num_envs}")
    print(f"  평가 에피소드: {args_cli.num_episodes}")
    print(f"  에피소드 최대 길이: {max_episode_len} 스텝 ({env_cfg.episode_length_s}초)")
    print(f"{'=' * 70}")

    # ── 평가 루프 ────────────────────────────────────────────────────────
    if use_random:
        obs, info = env.reset()
        action_dim = env.action_manager.total_action_dim
    else:
        obs = eval_env.get_observations()

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

    all_rewards = []
    all_lengths = []

    while simulation_app.is_running() and len(all_rewards) < args_cli.num_episodes:
        # 액션 선택
        if use_random:
            action = torch.randn(env.num_envs, action_dim, device=env.device).clamp(-1, 1)
            obs, rew, terminated, truncated, info = env.step(action)
        else:
            with torch.inference_mode():
                action = policy(obs)
            obs, rew, dones, info = eval_env.step(action)
            truncated = info.get("time_outs", torch.zeros(env.num_envs, dtype=torch.bool, device=env.device))
            terminated = dones.bool() & ~truncated

        episode_rewards += rew.squeeze() if rew.dim() > 1 else rew
        episode_lengths += 1

        done = terminated | truncated
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                ep_rew = episode_rewards[idx].item()
                ep_len = episode_lengths[idx].item()
                all_rewards.append(ep_rew)
                all_lengths.append(ep_len)

                # timeout = 5초 동안 서 있었다 = 성공 / fail = 도중에 넘어짐
                reason = "생존 (5초)" if truncated[idx].item() else "넘어짐"
                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {ep_rew:+9.2f} | 길이: {ep_len:4d} | {reason}"
                )

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0

    # ── 통계 출력 ────────────────────────────────────────────────────────
    if all_rewards:
        avg_reward = sum(all_rewards) / len(all_rewards)
        avg_length = sum(all_lengths) / len(all_lengths)
        success_threshold = int(max_episode_len * 0.95)
        success_rate = sum(1 for l in all_lengths if l >= success_threshold) / len(all_lengths) * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] {mode_name} 정책 평가 완료")
        print(f"  에피소드 수:       {len(all_rewards)}")
        print(f"  평균 보상:         {avg_reward:.2f}")
        print(f"  평균 길이:         {avg_length:.1f} / {max_episode_len} 스텝")
        print(f"  생존율 (5초 서있기): {success_rate:.1f}%")
        print(f"{'=' * 70}")
        if use_random:
            print("  → 학습된 정책과 비교해 보세요: python 35_3_play_standing.py")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 평가 종료")
