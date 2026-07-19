"""
36_4_train_bruce_lee.py - G1 휴머노이드 Bruce Lee(무술 모션 모방) PPO 학습

36_2(보행)와 같은 골격이지만, 회전·낮은 자세·후진이 섞인 무술 클립
(kungfu_retargeted/Bruce_Lee_pose.pkl)을 학습할 수 있게 수정한
g1_brucelee_env.py 환경을 사용합니다. 보행 환경과의 차이 4가지
(레퍼런스 대비 높이 종료 / 속도 벡터 추종 / yaw 추종 / 감점 완화)는
g1_brucelee_env.py 모듈 docstring을 참고하세요.

  - 체크포인트는 50 iteration마다 logs/rsl_rl/g1_brucelee_tutorial에 저장
    (보행 체크포인트와 폴더가 달라 섞이지 않음)
  - 관측이 107차원(보행 103 + yaw 오차 2 + 레퍼런스 속도 2)이므로
    보행 체크포인트와 호환되지 않습니다

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # GUI 없이 학습 (권장, 기본 2048 env × 3000 iter)
    python 36_4_train_bruce_lee.py --headless

    # 환경 수 / 학습 반복 횟수 변경 (스모크 테스트)
    python 36_4_train_bruce_lee.py --headless --num_envs 64 --max_iterations 5

    # 학습 결과 확인 (TensorBoard, 새 터미널에서)
    tensorboard --logdir logs/rsl_rl/g1_brucelee_tutorial

학습 로그 읽는 법:
  - Episode_Termination/base_height가 초반 이후 0 근처면 종료 조건 수정이 유효
  - Episode_Reward/yaw_tracking이 1.5+ 로 오르면 몸통 회전을 따라가는 중
  - joint_tracking이 보행(4.3+)만큼 오르지 않아도 정상 — 자세 다양성이 크다
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_4 - G1 Bruce Lee(무술 모션 모방) PPO 학습")
parser.add_argument("--num_envs", type=int, default=2048, help="병렬 환경 수")
parser.add_argument("--max_iterations", type=int, default=3000, help="학습 반복 횟수")
parser.add_argument("--motion_pkl", type=str, default="Bruce_Lee_pose",
                    help="레퍼런스 모션 이름 또는 pkl 경로 (기본: Bruce_Lee_pose)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# 공유 환경 모듈 (같은 폴더의 g1_brucelee_env.py — 보행 환경의 확장판)
from g1_brucelee_env import MOTION, G1BruceLeeEnvCfg, make_bruce_ppo_cfg


# ══════════════════════════════════════════════════════════════════════════
#  메인 학습 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """G1 Bruce Lee(무술 모션 모방) PPO 학습을 실행합니다."""

    # ── 레퍼런스 모션 로드 (env 생성 전에 파일만 미리 확인) ──────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 환경 설정 ────────────────────────────────────────────────────────
    env_cfg = G1BruceLeeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # ── 에이전트 설정 ────────────────────────────────────────────────────
    agent_cfg = make_bruce_ppo_cfg(args_cli.max_iterations)

    # ── 로그 디렉토리 설정 ───────────────────────────────────────────────
    log_dir = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] 로그 경로: {log_dir}")

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("=" * 70)
    print("[INFO] 학습 환경 생성 완료!")
    print(f"  환경 수:       {env.num_envs}")
    print(f"  관측 차원:     {env.observation_space}")
    print(f"  액션 차원:     {env.action_space}")
    print(f"  학습 반복:     {agent_cfg.max_iterations}")
    print(f"  레퍼런스:      {MOTION.pkl_name} ({MOTION.T}프레임, "
          f"{MOTION.T / MOTION.fps:.1f}초 순환)")
    print("=" * 70)

    # ── RSL-RL 환경 래퍼 ─────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)

    # ── PPO Runner 생성 ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)

    # ── 학습 실행 ────────────────────────────────────────────────────────
    print("\n[INFO] PPO 학습 시작!")
    print("=" * 70)

    # RSI가 시작 위상을 흩뜨리므로 False (36_2와 같은 이유)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # ── 학습 완료 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[INFO] 학습 완료!")
    print(f"  체크포인트 저장 위치: {log_dir}")
    print("  TensorBoard 확인: tensorboard --logdir logs/rsl_rl/g1_brucelee_tutorial")
    print("  평가 실행: python 36_5_play_bruce_lee.py")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 학습 종료")
