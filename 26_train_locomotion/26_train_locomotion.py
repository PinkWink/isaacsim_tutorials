"""
26_train_locomotion.py - ANYmal 4족 보행 로봇 PPO 학습

IsaacLab의 사전 등록된 ANYmal-B 보행 환경을 사용하여 RL로 걷기를 학습합니다:
  - gym.make()로 "Isaac-Velocity-Flat-Anymal-B-v0" 환경을 생성
  - RslRlVecEnvWrapper로 RSL-RL 인터페이스에 연결
  - PPO로 속도 추적(velocity tracking) 정책 학습
  - 24강(Cartpole)과의 차이: 더 큰 네트워크, 더 많은 환경, 더 복잡한 보상 구조

실행:
    source env_isaaclab/bin/activate
    cd lectures/26_train_locomotion
    python 26_train_locomotion.py --headless --num_envs 2048

    # 환경 수를 줄여서 실행 (GPU 메모리 부족 시)
    python 26_train_locomotion.py --headless --num_envs 512

    # 학습 반복 횟수 변경
    python 26_train_locomotion.py --headless --num_envs 2048 --max_iterations 500

    # 시각화하며 학습 (느리지만 로봇 움직임 확인 가능)
    python 26_train_locomotion.py --num_envs 64

    # 학습 결과 확인 (TensorBoard)
    tensorboard --logdir logs/rsl_rl/anymal_locomotion_tutorial
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
# 모든 IsaacLab 스크립트는 다른 import 전에 AppLauncher를 먼저 설정해야 합니다.
# Omniverse 시뮬레이터는 초기화 순서에 민감하기 때문입니다.
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="26 - ANYmal 보행 PPO 학습")
parser.add_argument("--num_envs", type=int, default=2048,
                    help="병렬 환경 수 (기본 2048, GPU 메모리에 따라 조절)")
parser.add_argument("--max_iterations", type=int, default=300,
                    help="학습 반복 횟수 (flat 지형은 300이면 충분)")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-B-v0",
                    help="등록된 환경 ID (Flat 또는 Rough)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
# AppLauncher가 초기화된 후에만 Isaac 관련 모듈을 import할 수 있습니다.
import gymnasium as gym
import torch

# isaaclab_tasks를 import하면 모든 사전 등록된 환경이 gym에 등록됩니다.
# 이 한 줄이 "Isaac-Velocity-Flat-Anymal-B-v0" 등의 환경을 사용 가능하게 합니다.
import isaaclab_tasks  # noqa: F401

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

# RSL-RL 관련 import
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
    RslRlVecEnvWrapper,
)
from rsl_rl.runners import OnPolicyRunner

# 환경 설정 로드 유틸리티
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


# ══════════════════════════════════════════════════════════════════════════
#  PPO 하이퍼파라미터 설정 (보행 학습용)
# ══════════════════════════════════════════════════════════════════════════
#
# 24강(Cartpole)과의 핵심 차이점:
#   1. 네트워크 크기: [32,32] → [128,128,128] (12개 관절 + 센서 데이터 처리)
#   2. 병렬 환경 수: 1024 → 2048 (더 많은 샘플이 필요)
#   3. 수집 스텝: 16 → 24 (보행 주기를 충분히 포함)
#   4. 엔트로피 계수: 0.005 → 0.01 (더 많은 탐색 필요)
#   5. 학습 반복: 150 → 300 (더 복잡한 태스크)


@configclass
class AnymalLocoPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """ANYmal 보행 학습용 PPO 설정.

    보행 학습이 Cartpole보다 어려운 이유:
      - 관측 공간이 훨씬 큼 (기체 속도, 중력 투영, 관절 위치/속도, 속도 명령 등)
      - 행동 공간이 12차원 (4개 다리 × 3개 관절)
      - 보상이 다중 목표 (속도 추적 + 자세 유지 + 에너지 절약 + 발 타이밍)
      - 물리적 안정성 유지가 필요 (넘어지지 않아야 함)
    """

    # ── 학습 루프 설정 ──────────────────────────────────────────────────
    num_steps_per_env = 24       # 업데이트 전 각 환경에서 수집할 스텝 수
                                 # 24 × 2048 = 49,152 샘플/업데이트
                                 # 보행 주기(~0.5초)를 여러 번 포함하도록 설정
    max_iterations = 300         # Flat 지형에서 300이면 충분 (Rough는 1500 권장)
    save_interval = 50           # 체크포인트 저장 주기
    experiment_name = "anymal_locomotion_tutorial"
    run_name = ""
    logger = "tensorboard"

    # ── 정책 네트워크 설정 (Actor-Critic) ────────────────────────────────
    # Cartpole [32,32]보다 훨씬 큰 네트워크가 필요합니다.
    # 3층 × 128 뉴런으로 복잡한 보행 패턴을 표현합니다.
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,               # 초기 탐색 노이즈
        actor_hidden_dims=[128, 128, 128], # 정책(액터): obs → 128 → 128 → 128 → 12 actions
        critic_hidden_dims=[128, 128, 128],# 가치(크리틱): obs → 128 → 128 → 128 → 1 value
        activation="elu",                  # ELU 활성화 (음수 영역에서도 기울기 유지)
    )

    # ── PPO 알고리즘 설정 ────────────────────────────────────────────────
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,              # 가치 함수 손실 가중치
        use_clipped_value_loss=True,      # 가치 손실 클리핑 (안정적 학습)
        clip_param=0.2,                   # PPO 클리핑 범위 ε
        entropy_coef=0.01,                # 엔트로피 보너스 (Cartpole 0.005보다 높음)
                                          # 보행은 다양한 걸음걸이 탐색이 중요
        num_learning_epochs=5,            # 수집 데이터로 학습 반복 횟수
        num_mini_batches=4,               # 미니배치 수
        learning_rate=1.0e-3,             # 학습률 (적응적 스케줄링 사용)
        schedule="adaptive",              # KL divergence 기반 학습률 조절
        gamma=0.99,                       # 할인율
        lam=0.95,                         # GAE lambda
        desired_kl=0.01,                  # 목표 KL divergence
        max_grad_norm=1.0,                # gradient 클리핑 (발산 방지)
    )


# ══════════════════════════════════════════════════════════════════════════
#  메인 학습 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """ANYmal 보행 PPO 학습을 실행합니다.

    24강(Cartpole)과의 주요 차이점:
      1. 환경을 직접 정의하지 않고 gym.make()로 사전 등록된 환경 사용
      2. 속도 명령(velocity command)이 관측에 포함됨
      3. 지형 커리큘럼(terrain curriculum)이 자동 적용됨
      4. 보상이 속도 추적, 자세 유지, 에너지 절약 등 다중 목표로 구성됨
    """

    # ── 환경 설정 로드 ───────────────────────────────────────────────────
    # gym 레지스트리에서 환경 설정을 로드합니다.
    # "env_cfg_entry_point"는 __init__.py에서 gym.register() 시 등록된 경로입니다.
    task_name = args_cli.task
    print(f"[INFO] 태스크: {task_name}")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")

    # 명령줄에서 지정한 환경 수로 오버라이드
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── 에이전트 설정 ────────────────────────────────────────────────────
    agent_cfg = AnymalLocoPPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    # ── 로그 디렉토리 설정 ───────────────────────────────────────────────
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] 로그 경로: {log_root_path}")

    # ── gym.make()로 환경 생성 ───────────────────────────────────────────
    # 24강에서는 ManagerBasedRLEnv(cfg=...)로 직접 생성했지만,
    # 여기서는 gym.make()를 사용합니다. IsaacLab의 사전 등록된 환경은
    # 씬(로봇, 센서, 지형), 관측, 보상, 종료 조건이 모두 설정되어 있습니다.
    env = gym.make(task_name, cfg=env_cfg)

    # gym.make()가 반환하는 것은 gymnasium.Env 래퍼입니다.
    # 내부의 실제 IsaacLab 환경에 접근하려면 env.unwrapped를 사용합니다.
    unwrapped = env.unwrapped

    print("=" * 70)
    print("[INFO] ANYmal 보행 학습 환경 생성 완료!")
    print(f"  태스크:        {task_name}")
    print(f"  환경 수:       {unwrapped.num_envs}")
    print(f"  관측 차원:     {unwrapped.observation_space}")
    print(f"  액션 차원:     {unwrapped.action_space}")
    print(f"  학습 반복:     {agent_cfg.max_iterations}")
    print(f"  스텝/업데이트:  {agent_cfg.num_steps_per_env} × {unwrapped.num_envs} = "
          f"{agent_cfg.num_steps_per_env * unwrapped.num_envs}")
    print("=" * 70)

    # ── 보행 환경의 관측 구성 설명 ───────────────────────────────────────
    # 24강 Cartpole 관측 (7차원): [joint_pos, joint_vel, sin/cos, cart_pos]
    # ANYmal 보행 관측 (~48차원, Flat 기준):
    #   - base_lin_vel (3):     기체 선속도 (x, y, z)
    #   - base_ang_vel (3):     기체 각속도 (roll, pitch, yaw)
    #   - projected_gravity (3): 기체 프레임에서의 중력 방향 (자세 파악)
    #   - velocity_commands (3): 목표 속도 명령 (vx, vy, ωz)
    #   - joint_pos (12):       12개 관절의 상대 위치
    #   - joint_vel (12):       12개 관절의 상대 속도
    #   - actions (12):         이전 스텝의 액션 (부드러운 동작 유도)
    print("\n[INFO] 보행 환경 관측 구성:")
    print("  기체 선속도 (3) + 각속도 (3) + 중력 투영 (3)")
    print("  + 속도 명령 (3) + 관절 위치 (12) + 관절 속도 (12)")
    print("  + 이전 액션 (12) = ~48차원")

    # ── 보행 환경의 보상 구성 설명 ───────────────────────────────────────
    # Cartpole 보상: alive + upright + center + energy (4개)
    # ANYmal 보행 보상 (다중 목표 구조):
    #   [추적 보상] 속도 명령을 얼마나 잘 따르는가
    #     - track_lin_vel_xy_exp: 선속도 추적 (가장 중요, weight=1.0)
    #     - track_ang_vel_z_exp:  각속도 추적 (weight=0.5)
    #   [스타일 보상] 자연스러운 보행 패턴
    #     - feet_air_time: 발이 공중에 있는 시간 (걸음걸이 리듬)
    #   [페널티] 원치 않는 행동 억제
    #     - lin_vel_z_l2:       수직 방향 속도 (뛰지 않게)
    #     - ang_vel_xy_l2:      roll/pitch 각속도 (흔들리지 않게)
    #     - dof_torques_l2:     관절 토크 (에너지 절약)
    #     - dof_acc_l2:         관절 가속도 (부드러운 동작)
    #     - action_rate_l2:     액션 변화율 (급격한 동작 방지)
    #     - undesired_contacts: 허벅지 접촉 (넘어지지 않게)
    print("\n[INFO] 보행 보상 구조:")
    print("  [추적] 선속도/각속도 명령 추적")
    print("  [스타일] 발 공중 시간 (걸음걸이 리듬)")
    print("  [페널티] 수직속도, 흔들림, 토크, 가속도, 접촉")

    # ── RSL-RL 환경 래퍼 ─────────────────────────────────────────────────
    # gym.make()의 결과를 RSL-RL이 기대하는 VecEnv 인터페이스로 변환합니다.
    # clip_actions: 액션을 [-clip, clip] 범위로 제한 (기본값 사용)
    env = RslRlVecEnvWrapper(env)

    # ── PPO Runner 생성 ──────────────────────────────────────────────────
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_root_path, device=agent_cfg.device)

    # ── 학습 실행 ────────────────────────────────────────────────────────
    print("\n[INFO] ANYmal PPO 학습 시작!")
    print(f"  네트워크: Actor {agent_cfg.policy.actor_hidden_dims}, "
          f"Critic {agent_cfg.policy.critic_hidden_dims}")
    print(f"  학습률: {agent_cfg.algorithm.learning_rate} (적응적 스케줄)")
    print(f"  할인율(γ): {agent_cfg.algorithm.gamma}")
    print(f"  엔트로피 계수: {agent_cfg.algorithm.entropy_coef}")
    print(f"  PPO 클리핑(ε): {agent_cfg.algorithm.clip_param}")
    print("=" * 70)
    print()

    # init_at_random_ep_len=True:
    #   각 환경의 에피소드를 랜덤 길이에서 시작하여
    #   모든 환경이 동시에 리셋되는 것을 방지합니다.
    #   이렇게 하면 학습 데이터의 다양성이 높아집니다.
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # ── 학습 완료 ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("[INFO] ANYmal 보행 학습 완료!")
    print(f"  체크포인트 저장 위치: {log_root_path}")
    print(f"  TensorBoard 확인: tensorboard --logdir {log_root_path}")
    print()
    print("[TIP] 학습된 정책 평가:")
    print("  Flat 환경에서 학습 후 Rough 환경에서 fine-tuning하면")
    print("  울퉁불퉁한 지형에서도 안정적으로 걸을 수 있습니다.")
    print("=" * 70)

    # ── 환경 종료 ────────────────────────────────────────────────────────
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 보행 학습 종료")
