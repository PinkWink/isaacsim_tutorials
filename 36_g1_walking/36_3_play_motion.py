"""
36_3_play_motion.py - 학습된 G1 Walking 정책 평가 및 시각화

36_2에서 학습한 PPO 체크포인트를 로드하여 G1이 레퍼런스 모션처럼 걷는
모습을 확인하고 성능을 평가합니다:
  - 체크포인트(model_*.pt) 자동 탐색 및 로드
  - 학습된 정책으로 환경 실행 (시각화)
  - 보행 지표: 관절 추종 RMS 오차 / 평균 전진 속도 / 생존율
  - 관절 토크 모니터링: 왼쪽 다리 6관절 + 허리(torso) — env 0 기준
    실시간 그래프(GUI) + 종료 시 36_3_torque_result.png 저장 + 통계 출력
  - 랜덤 정책과 비교 (--random)

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # 최신 체크포인트 자동 탐색 후 평가
    python 36_3_play_motion.py

    # 체크포인트 직접 지정
    python 36_3_play_motion.py --checkpoint logs/rsl_rl/g1_walking_tutorial/model_2000.pt

    # 랜덤 정책으로 평가 (학습 전후 비교용)
    python 36_3_play_motion.py --random
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_3 - G1 Walking 정책 평가")
parser.add_argument("--num_envs", type=int, default=4, help="평가할 환경 수")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로")
parser.add_argument("--random", action="store_true", help="랜덤 정책으로 평가")
parser.add_argument("--num_episodes", type=int, default=10, help="평가할 에피소드 수")
parser.add_argument("--motion_pkl", type=str, default=None,
                    help="레퍼런스 모션 이름 또는 pkl 경로 (예: run1_subject2)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv

# 공유 환경 모듈 (같은 폴더의 g1_walking_env.py)
# 학습(36_2)과 동일한 관측/보상/네트워크 정의를 써야 체크포인트가 정상 동작합니다.
from g1_walking_env import MOTION, G1WalkingEnvCfg, find_latest_checkpoint, make_ppo_runner_cfg

# ── 3. Matplotlib (토크 모니터링용) ──────────────────────────────────────
import matplotlib

# --headless일 때는 화면 표시 없이 PNG 저장만 (plt.show() 블로킹 방지)
if os.environ.get("DISPLAY") and not args_cli.headless:
    try:
        matplotlib.use("TkAgg")
        INTERACTIVE = True
    except Exception:
        matplotlib.use("Agg")
        INTERACTIVE = False
else:
    matplotlib.use("Agg")
    INTERACTIVE = False
import matplotlib.pyplot as plt

# 토크를 모니터링할 관절: 왼쪽 다리 6개 + 허리(torso) 1개
TORQUE_JOINT_EXPR = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "torso_joint",
]


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """학습된 정책 또는 랜덤 정책으로 G1 Walking을 평가합니다."""

    # ── 레퍼런스 모션 로드 (36_2와 동일해야 관측/위상이 일치) ────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 환경 생성 (36_2와 동일 환경, 개수만 축소) ────────────────────────
    env_cfg = G1WalkingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]

    # 에피소드 최대 길이 (스텝 수): 10초 / (0.005 × 4) = 500
    max_episode_len = int(env_cfg.episode_length_s / (env_cfg.sim.dt * env_cfg.decimation))
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    # ── 정책 로드 (OnPolicyRunner로 36_2와 동일하게) ─────────────────────
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

            # 36_2와 동일한 에이전트 설정 (네트워크 형상이 다르면 로드 실패)
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
            print("  먼저 36_2 학습을 실행하세요:")
            print("    python 36_2_train_motion.py --headless")
            use_random = True

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] {mode_name} 정책 평가 시작")
    print(f"  환경 수: {env.num_envs}")
    print(f"  평가 에피소드: {args_cli.num_episodes}")
    print(f"  에피소드 최대 길이: {max_episode_len} 스텝 ({env_cfg.episode_length_s}초)")
    print(f"  레퍼런스 평균 속도: {MOTION.ref_speed.mean():.2f} m/s")
    print(f"{'=' * 70}")

    # ── 토크 모니터링 설정 (env 0: 왼쪽 다리 6관절 + 허리) ───────────────
    # G1은 implicit(PD) 액추에이터라 PhysX가 토크를 직접 노출하지 않지만,
    # IsaacLab이 매 스텝 PD 법칙(stiffness×위치오차 + damping×속도오차)으로
    # 근사 토크를 계산해 applied_torque에 저장한다 (effort limit 클리핑 포함).
    torque_ids, torque_names = robot.find_joints(TORQUE_JOINT_EXPR, preserve_order=True)
    print(f"[INFO] 토크 모니터링 관절 ({len(torque_ids)}개): {torque_names}")

    torque_hist = []       # env 0의 스텝별 토크 (7,) [Nm]

    if INTERACTIVE:
        plt.ion()
    # 3단 그래프: (1) 다리 4관절 (hip×3 + knee) / (2) 발목 2관절 / (3) 허리
    # 발목은 토크 스케일이 다리보다 작아서 따로 그려야 파형이 잘 보인다.
    fig, (ax_leg, ax_ankle, ax_waist) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    # 그래프 제목은 영문으로 (matplotlib 기본 폰트는 한글 미지원 → 네모 깨짐)
    policy_label = "random" if use_random else "trained"
    fig.suptitle(f"G1 Walking Joint Torques — env 0 ({policy_label} policy)", fontsize=12)
    leg_lines = [ax_leg.plot([], [], lw=1.0, label=n.replace("left_", ""))[0]
                 for n in torque_names[:4]]                    # hip pitch/roll/yaw + knee
    ankle_lines = [ax_ankle.plot([], [], lw=1.0, label=n.replace("left_", ""))[0]
                   for n in torque_names[4:6]]                 # ankle pitch/roll
    waist_line = ax_waist.plot([], [], lw=1.0, color="tab:red", label=torque_names[-1])[0]
    ax_leg.set_ylabel("Hip/Knee torque [Nm]")
    ax_leg.legend(loc="upper right", fontsize=7, ncol=2)
    ax_ankle.set_ylabel("Ankle torque [Nm]")
    ax_ankle.legend(loc="upper right", fontsize=7)
    ax_waist.set_ylabel("Waist torque [Nm]")
    ax_waist.set_xlabel("Time [s]")
    ax_waist.legend(loc="upper right", fontsize=7)
    torque_axes = (ax_leg, ax_ankle, ax_waist)
    for ax in torque_axes:
        ax.grid(alpha=0.3)
    plt.tight_layout()
    PLOT_INTERVAL = 25    # N스텝마다 그래프 갱신

    def update_torque_plot() -> None:
        """기록된 토크 시계열을 그래프 라인에 반영합니다."""
        data = torch.stack(torque_hist)
        tt = torch.arange(data.shape[0]) * step_dt
        for j, ln in enumerate(leg_lines):
            ln.set_data(tt, data[:, j])
        for j, ln in enumerate(ankle_lines):
            ln.set_data(tt, data[:, 4 + j])
        waist_line.set_data(tt, data[:, -1])
        for ax in torque_axes:
            ax.relim()
            ax.autoscale_view()

    # ── 평가 루프 ────────────────────────────────────────────────────────
    if use_random:
        obs, info = env.reset()
        action_dim = env.action_manager.total_action_dim
    else:
        obs = eval_env.get_observations()

    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    # 보행 지표 누적 버퍼
    track_err_sq = torch.zeros(env.num_envs, device=env.device)   # 관절 추종 오차² 합
    forward_dist = torch.zeros(env.num_envs, device=env.device)   # 몸 기준 전진 거리 적분

    all_rewards = []
    all_lengths = []
    all_track_rms = []
    all_speeds = []

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

        # ── 보행 지표 누적 ───────────────────────────────────────────────
        # (주의: done env는 step 안에서 이미 리셋됐지만, 오차 1스텝 정도의
        #  오염은 통계에 무시할 수준이라 강의 코드에서는 단순하게 둔다)
        cur = robot.data.joint_pos[:, MOTION.joint_map]
        ref = MOTION.dof[MOTION.frame_idx(env)]
        track_err_sq += torch.sum(torch.square(cur - ref), dim=1)
        forward_dist += robot.data.root_lin_vel_b[:, 0] * step_dt

        # ── 토크 기록 + 실시간 그래프 (env 0) ────────────────────────────
        torque_hist.append(robot.data.applied_torque[0, torque_ids].detach().cpu().clone())
        if INTERACTIVE and len(torque_hist) % PLOT_INTERVAL == 0:
            update_torque_plot()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        done = terminated | truncated
        # env 0의 에피소드 경계를 회색 점선으로 표시
        if done[0]:
            t_mark = len(torque_hist) * step_dt
            for ax in torque_axes:
                ax.axvline(t_mark, color="gray", ls="--", lw=0.8, alpha=0.7)
        if done.any():
            done_envs = done.nonzero(as_tuple=False).squeeze(-1)
            for idx in done_envs:
                ep_rew = episode_rewards[idx].item()
                ep_len = episode_lengths[idx].item()
                # 관절 추종 RMS 오차 (rad): sqrt(오차²합 / (스텝수 × 23관절))
                rms = (track_err_sq[idx] / (ep_len * 23)).sqrt().item()
                # 평균 전진 속도 (m/s)
                speed = forward_dist[idx].item() / (ep_len * step_dt)
                all_rewards.append(ep_rew)
                all_lengths.append(ep_len)
                all_track_rms.append(rms)
                all_speeds.append(speed)

                # timeout = 10초 동안 걸었다 = 성공 / fail = 도중에 넘어짐
                reason = "생존 (10초)" if truncated[idx].item() else "넘어짐"
                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {ep_rew:+9.2f} | 길이: {ep_len:4d} | "
                    f"추종 RMS: {rms:.3f} rad | 속도: {speed:+.2f} m/s | {reason}"
                )

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0
            track_err_sq[done] = 0.0
            forward_dist[done] = 0.0

    # ── 통계 출력 ────────────────────────────────────────────────────────
    if all_rewards:
        n = len(all_rewards)
        avg_reward = sum(all_rewards) / n
        avg_length = sum(all_lengths) / n
        avg_rms = sum(all_track_rms) / n
        avg_speed = sum(all_speeds) / n
        success_threshold = int(max_episode_len * 0.95)
        success_rate = sum(1 for l in all_lengths if l >= success_threshold) / n * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] {mode_name} 정책 평가 완료")
        print(f"  에피소드 수:        {n}")
        print(f"  평균 보상:          {avg_reward:.2f}")
        print(f"  평균 길이:          {avg_length:.1f} / {max_episode_len} 스텝")
        print(f"  관절 추종 RMS 오차:  {avg_rms:.3f} rad")
        print(f"  평균 전진 속도:      {avg_speed:+.2f} m/s (레퍼런스 {MOTION.ref_speed.mean():.2f})")
        print(f"  생존율 (10초 보행):  {success_rate:.1f}%")
        print(f"{'=' * 70}")
        if use_random:
            print("  → 학습된 정책과 비교해 보세요: python 36_3_play_motion.py")

    # ── 토크 통계 및 그래프 저장 ─────────────────────────────────────────
    if torque_hist:
        update_torque_plot()
        data = torch.stack(torque_hist)

        print(f"\n[토크 통계] env 0, {data.shape[0]}스텝 (PD 근사 토크)")
        print(f"  {'관절':30s} {'평균|τ| [Nm]':>14s} {'최대|τ| [Nm]':>14s}")
        for j, name in enumerate(torque_names):
            print(f"  {name:30s} {data[:, j].abs().mean():12.2f} {data[:, j].abs().max():12.2f}")

        save_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "36_3_torque_result.png"
        )
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[INFO] 토크 그래프 저장: {save_path}")

        if INTERACTIVE:
            plt.ioff()
            print("[INFO] 그래프 창을 닫으면 종료합니다.")
            plt.show()
        else:
            plt.close(fig)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 평가 종료")
