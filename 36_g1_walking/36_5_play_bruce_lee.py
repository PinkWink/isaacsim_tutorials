"""
36_5_play_bruce_lee.py - 학습된 G1 Bruce Lee 정책 평가 및 시각화

36_4에서 학습한 PPO 체크포인트를 로드하여 G1이 무술 모션을 따라하는
모습을 확인하고 성능을 평가합니다:
  - 체크포인트(logs/rsl_rl/g1_brucelee_tutorial/model_*.pt) 자동 탐색/로드
  - 모방 지표: 관절 추종 RMS 오차 / yaw 추종 RMS 오차 / 생존율
  - 랜덤 정책과 비교 (--random)

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # 최신 체크포인트 자동 탐색 후 평가 (GUI)
    python 36_5_play_bruce_lee.py

    # 체크포인트 직접 지정
    python 36_5_play_bruce_lee.py --checkpoint logs/rsl_rl/g1_brucelee_tutorial/model_2999.pt

    # 랜덤 정책으로 평가 (학습 전후 비교용)
    python 36_5_play_bruce_lee.py --random
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_5 - G1 Bruce Lee 정책 평가")
parser.add_argument("--num_envs", type=int, default=4, help="평가할 환경 수")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로")
parser.add_argument("--random", action="store_true", help="랜덤 정책으로 평가")
parser.add_argument("--num_episodes", type=int, default=10, help="평가할 에피소드 수")
parser.add_argument("--motion_pkl", type=str, default="Bruce_Lee_pose",
                    help="레퍼런스 모션 이름 또는 pkl 경로 (기본: Bruce_Lee_pose)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv

# 공유 환경 모듈 (같은 폴더의 g1_brucelee_env.py)
# 학습(36_4)과 동일한 관측/보상/네트워크 정의를 써야 체크포인트가 정상 동작합니다.
from g1_brucelee_env import (
    MOTION,
    G1BruceLeeEnvCfg,
    _quat_yaw_wxyz,
    find_latest_bruce_checkpoint,
    make_bruce_ppo_cfg,
)


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """학습된 정책 또는 랜덤 정책으로 G1 Bruce Lee 모방을 평가합니다."""

    # ── 레퍼런스 모션 로드 (36_4와 동일해야 관측/위상이 일치) ────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 환경 생성 (36_4와 동일 환경, 개수만 축소) ────────────────────────
    env_cfg = G1BruceLeeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    max_episode_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))

    # ── 정책 로드 (OnPolicyRunner로 36_4와 동일하게) ─────────────────────
    policy = None
    eval_env = None
    use_random = args_cli.random

    if not use_random:
        checkpoint_path = args_cli.checkpoint or find_latest_bruce_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 체크포인트: {checkpoint_path}")
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
            from rsl_rl.runners import OnPolicyRunner

            agent_cfg = make_bruce_ppo_cfg(max_iterations=0)
            env_wrapped = RslRlVecEnvWrapper(env)
            runner = OnPolicyRunner(
                env_wrapped, agent_cfg.to_dict(),
                log_dir=None, device=str(env.device),
            )
            runner.load(checkpoint_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=str(env.device))
            eval_env = env_wrapped
            print("[INFO] 학습된 정책 로드 완료!")
        else:
            print("[WARNING] 체크포인트를 찾을 수 없습니다. 랜덤 정책으로 평가합니다.")
            print("  먼저 36_4 학습을 실행하세요:")
            print("    python 36_4_train_bruce_lee.py --headless")
            use_random = True

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] {mode_name} 정책 평가 시작")
    print(f"  환경 수: {env.num_envs} | 에피소드: {args_cli.num_episodes} | "
          f"최대 길이: {max_episode_len} 스텝 ({env_cfg.episode_length_s}초)")
    print(f"  레퍼런스: {MOTION.pkl_name} ({MOTION.T / MOTION.fps:.1f}초 순환)")
    print(f"{'=' * 70}")

    # ── 평가 루프 ────────────────────────────────────────────────────────
    if use_random:
        obs, info = env.reset()
        action_dim = env.action_manager.total_action_dim
    else:
        obs = eval_env.get_observations()

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 모방 지표 누적 버퍼
    track_err_sq = torch.zeros(env.num_envs, device=env.device)  # 관절 추종 오차² 합
    yaw_err_sq = torch.zeros(env.num_envs, device=env.device)    # yaw 추종 오차² 합

    all_rewards, all_lengths, all_track_rms, all_yaw_rms = [], [], [], []

    while simulation_app.is_running() and len(all_rewards) < args_cli.num_episodes:
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

        # ── 모방 지표 누적 (done env의 1스텝 오염은 무시 — 36_3과 동일) ──
        idx = MOTION.frame_idx(env)
        cur = robot.data.joint_pos[:, MOTION.joint_map]
        track_err_sq += torch.sum(torch.square(cur - MOTION.dof[idx]), dim=1)
        yaw_err = _quat_yaw_wxyz(robot.data.root_quat_w) - MOTION.ref_yaw[idx]
        yaw_err = torch.atan2(torch.sin(yaw_err), torch.cos(yaw_err))
        yaw_err_sq += torch.square(yaw_err)

        done = terminated | truncated
        if done.any():
            for i in done.nonzero(as_tuple=False).squeeze(-1):
                ep_len = episode_lengths[i].item()
                rms = (track_err_sq[i] / (ep_len * 23)).sqrt().item()
                yaw_rms = (yaw_err_sq[i] / ep_len).sqrt().item()
                all_rewards.append(episode_rewards[i].item())
                all_lengths.append(ep_len)
                all_track_rms.append(rms)
                all_yaw_rms.append(yaw_rms)

                reason = "생존 (12초)" if truncated[i].item() else "이탈/넘어짐"
                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {episode_rewards[i].item():+9.2f} | 길이: {ep_len:4d} | "
                    f"관절 RMS: {rms:.3f} rad | yaw RMS: {torch.rad2deg(torch.tensor(yaw_rms)):.1f}° | {reason}"
                )

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0
            track_err_sq[done] = 0.0
            yaw_err_sq[done] = 0.0

    # ── 통계 출력 ────────────────────────────────────────────────────────
    if all_rewards:
        n = len(all_rewards)
        success_threshold = int(max_episode_len * 0.95)
        success_rate = sum(1 for l in all_lengths if l >= success_threshold) / n * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] {mode_name} 정책 평가 완료")
        print(f"  에피소드 수:        {n}")
        print(f"  평균 보상:          {sum(all_rewards) / n:.2f}")
        print(f"  평균 길이:          {sum(all_lengths) / n:.1f} / {max_episode_len} 스텝")
        print(f"  관절 추종 RMS 오차:  {sum(all_track_rms) / n:.3f} rad")
        print(f"  yaw 추종 RMS 오차:   {torch.rad2deg(torch.tensor(sum(all_yaw_rms) / n)):.1f}°")
        print(f"  생존율 (12초):       {success_rate:.1f}%")
        print(f"{'=' * 70}")
        if use_random:
            print("  → 학습된 정책과 비교해 보세요: python 36_5_play_bruce_lee.py")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 평가 종료")
