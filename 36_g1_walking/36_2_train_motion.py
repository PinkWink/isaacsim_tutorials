"""
36_2_train_motion.py - G1 휴머노이드 Walking(모션 모방) PPO 학습

g1_walking_env.py의 환경(관측 8항목 / 보상 10항목 / RSI 이벤트)에
RSL-RL의 PPO를 적용하여 G1이 레퍼런스 모션처럼 걷도록 학습합니다:
  - 모방 학습: 레퍼런스 관절각 추종 보상(joint_tracking)이 핵심
  - RSI: 리셋 시 레퍼런스의 임의 프레임 자세에서 시작 → 수렴 가속
  - 체크포인트는 50 iteration마다 logs/rsl_rl/g1_walking_tutorial에 저장
  - TensorBoard로 학습 과정 모니터링

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # GUI 없이 학습 (권장, 기본 2048 env × 2000 iter)
    python 36_2_train_motion.py --headless

    # 환경 수 / 학습 반복 횟수 변경 (스모크 테스트)
    python 36_2_train_motion.py --headless --num_envs 64 --max_iterations 5

    # 학습 결과 확인 (TensorBoard, 새 터미널에서)
    tensorboard --logdir logs/rsl_rl/g1_walking_tutorial

학습 로그 읽는 법:
  - Episode_Reward/joint_tracking이 초반부터 0보다 크면 RSI가 정상 동작
  - Mean episode length가 500(10초)을 향해 늘어나면 넘어지지 않고 걷는 중
  - Mean reward 증가 + episode length 정체라면 서서 버티기만 배우는 중
    → speed_tracking 가중치를 높여본다
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_2 - G1 Walking(모션 모방) PPO 학습")
parser.add_argument("--num_envs", type=int, default=2048, help="병렬 환경 수")
parser.add_argument("--max_iterations", type=int, default=2000, help="학습 반복 횟수")
parser.add_argument("--motion_pkl", type=str, default=None,
                    help="레퍼런스 모션 이름 또는 pkl 경로 (예: run1_subject2)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv

# RSL-RL 관련 import
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# 공유 환경 모듈 (같은 폴더의 g1_walking_env.py)
# PPO 하이퍼파라미터도 여기 정의되어 있음 (36_3 평가와 공유)
from g1_walking_env import MOTION, G1WalkingEnvCfg, make_ppo_runner_cfg


# ══════════════════════════════════════════════════════════════════════════
#  메인 학습 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """G1 Walking(모션 모방) PPO 학습을 실행합니다."""

    # ── 레퍼런스 모션 로드 (env 생성 전에 파일만 미리 확인) ──────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 환경 설정 ────────────────────────────────────────────────────────
    env_cfg = G1WalkingEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device

    # ── 에이전트 설정 ────────────────────────────────────────────────────
    agent_cfg = make_ppo_runner_cfg(args_cli.max_iterations)

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
    print(f"  스텝/업데이트:  {agent_cfg.num_steps_per_env} × {env.num_envs} = "
          f"{agent_cfg.num_steps_per_env * env.num_envs}")
    print("=" * 70)

    # ── RSL-RL 환경 래퍼 ─────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)

    # ── PPO Runner 생성 ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=args_cli.device)

    # ── 학습 실행 ────────────────────────────────────────────────────────
    print("\n[INFO] PPO 학습 시작!")
    print(f"  네트워크: Actor {agent_cfg.policy.actor_hidden_dims}, "
          f"Critic {agent_cfg.policy.critic_hidden_dims}")
    print(f"  학습률: {agent_cfg.algorithm.learning_rate}")
    print(f"  할인율(γ): {agent_cfg.algorithm.gamma}")
    print(f"  PPO 클리핑(ε): {agent_cfg.algorithm.clip_param}")
    print("=" * 70)

    # init_at_random_ep_len=False에 주의!
    # 35강에서는 True로 두어 에피소드 길이를 흩뜨렸지만, 여기서는 RSI가
    # 시작 위상을 흩뜨려 준다. True로 두면 episode_length_buf 기반 위상
    # 계산이 RSI로 맞춰둔 시작 자세와 어긋나 모방 학습이 망가진다.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)

    # ── 학습 완료 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[INFO] 학습 완료!")
    print(f"  체크포인트 저장 위치: {log_dir}")
    print("  TensorBoard 확인: tensorboard --logdir logs/rsl_rl/g1_walking_tutorial")
    print("  평가 실행: python 36_3_play_motion.py")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 학습 종료")
