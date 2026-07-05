"""
35_2_train_standing.py - G1 휴머노이드 Standing PPO 학습

g1_standing_env.py의 환경(관측 6항목 / 보상 6항목 / 종료 3조건)에
RSL-RL의 PPO를 적용하여 G1이 스스로 서 있도록 학습합니다:
  - RslRlVecEnvWrapper로 환경을 RSL-RL 인터페이스에 연결
  - PPO: 정책을 한 번에 크게 바꾸지 않고 조금씩 업데이트하는 알고리즘
  - 체크포인트는 50 iteration마다 logs/rsl_rl/g1_standing_tutorial에 저장
  - TensorBoard로 학습 과정 모니터링

실행:
    source env_isaaclab/bin/activate
    cd lectures/35_g1_standing

    # GUI 없이 학습 (권장)
    python 35_2_train_standing.py --headless

    # 환경 수 / 학습 반복 횟수 변경
    python 35_2_train_standing.py --headless --num_envs 512 --max_iterations 1000

    # 학습 결과 확인 (TensorBoard, 새 터미널에서)
    tensorboard --logdir logs/rsl_rl/g1_standing_tutorial

학습 로그 읽는 법:
  - Mean reward가 서서히 증가하고 Mean episode length가 함께 늘어나면 정상 학습
  - Mean reward가 감소하면 보상 설계를 재검토
  - entropy가 너무 빠르게 0이 되면 entropy_coef를 높인다

참고:
    실행 중 numpy 관련 에러가 발생하면 버전을 맞춰주세요:
    pip install numpy==1.26.0
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="35_2 - G1 Standing PPO 학습")
parser.add_argument("--num_envs", type=int, default=512, help="병렬 환경 수")
parser.add_argument("--max_iterations", type=int, default=1000, help="학습 반복 횟수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
from isaaclab.envs import ManagerBasedRLEnv

# RSL-RL 관련 import
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

# 공유 환경 모듈 (같은 폴더의 g1_standing_env.py)
# PPO 하이퍼파라미터도 여기 정의되어 있음 (35_3 평가와 공유)
from g1_standing_env import G1StandingEnvCfg, make_ppo_runner_cfg


# ══════════════════════════════════════════════════════════════════════════
#  메인 학습 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """G1 Standing PPO 학습을 실행합니다."""

    # ── 환경 설정 ────────────────────────────────────────────────────────
    env_cfg = G1StandingEnvCfg()
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
    # IsaacLab 환경 → RSL-RL VecEnv 인터페이스로 변환
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

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── 학습 완료 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[INFO] 학습 완료!")
    print(f"  체크포인트 저장 위치: {log_dir}")
    print("  TensorBoard 확인: tensorboard --logdir logs/rsl_rl/g1_standing_tutorial")
    print("  평가 실행: python 35_3_play_standing.py")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 학습 종료")
