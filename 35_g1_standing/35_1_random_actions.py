"""
35_1_random_actions.py - 병렬 환경 스폰 + 랜덤 액션 (학습 전 상태 관찰)

관측/보상/종료를 정의하지 않은 환경에 G1 휴머노이드를 대량으로 스폰하고
매 스텝 랜덤 관절 명령을 주면 어떻게 되는지 관찰합니다:
  - ManagerBasedRLEnv는 관측/보상/종료 매니저가 비어 있어도 생성 가능
  - 행동(관절 위치 명령)만 정의하고 랜덤 값을 넣으면
    로봇마다 제각각 허우적대다 사방으로 넘어진다
  - 병렬 환경이 많을수록 같은 시간에 더 많은 경험을 모을 수 있다 → RL의 핵심

이 "넘어지는 상태"가 출발점이고, 35_2에서 보상 설계 + PPO로
로봇이 스스로 서 있도록 학습시킵니다.

실행:
    source env_isaaclab/bin/activate
    cd lectures/35_g1_standing
    python 35_1_random_actions.py

    # 환경 수를 늘려서 (학습 때는 512개 사용)
    python 35_1_random_actions.py --num_envs 256 --headless

참고:
    실행 중 numpy 관련 에러가 발생하면 버전을 맞춰주세요:
    pip install numpy==1.26.0
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="35_1 - G1 병렬 환경 스폰 + 랜덤 액션")
parser.add_argument("--num_envs", type=int, default=64, help="병렬 환경 수 (GUI에서는 64 권장)")
parser.add_argument("--max_steps", type=int, default=500, help="실행할 스텝 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.utils import configclass

# 공유 환경 모듈 (같은 폴더의 g1_standing_env.py)
# 행동 정의(ActionsCfg)는 35_2 학습 환경과 동일한 것을 사용
from g1_standing_env import ActionsCfg, G1SceneCfg


# ══════════════════════════════════════════════════════════════════════════
#  최소 RL 환경: 행동만 있고 관측/보상/종료는 비어 있는 환경
# ══════════════════════════════════════════════════════════════════════════


@configclass
class EmptyManagerCfg:
    """아무 항목도 없는 빈 매니저. 관측/보상/종료가 전부 비어 있다."""

    pass


@configclass
class G1RandomEnvCfg(ManagerBasedRLEnvCfg):
    """행동만 정의된 G1 환경. 씬/행동 구성은 35_2 학습 환경과 동일."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=64, env_spacing=3.0)
    observations: EmptyManagerCfg = EmptyManagerCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: EmptyManagerCfg = EmptyManagerCfg()
    terminations: EmptyManagerCfg = EmptyManagerCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
#  메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """빈 환경에 G1을 스폰하고 랜덤 액션으로 실행합니다."""

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env_cfg = G1RandomEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)
    action_dim = env.action_manager.total_action_dim

    print("=" * 70)
    print("[INFO] 환경 생성 완료!")
    print(f"  환경 수:   {env.num_envs}")
    print(f"  관측 공간: {env.observation_space}  (비어 있음)")
    print(f"  액션 차원: {action_dim}  (전체 관절 위치 명령)")
    print("=" * 70)
    print("[INFO] 랜덤 관절 명령이라 로봇은 허우적대다 전부 넘어집니다.")
    print("       35_2에서 보상 설계 + PPO 학습으로 서 있게 만듭니다.\n")

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    obs, _ = env.reset()
    # 넘어지는 방향 분석용: 시작 시점의 몸체 위치 기록
    start_pos = env.scene["robot"].data.root_pos_w[:, :2].clone()
    step_count = 0

    # 로봇마다 다른 "랜덤 성향" 부여: 매 스텝 새로 뽑는 노이즈는 평균이 0이라
    # 서로 상쇄되어 고주파 떨림만 남고, 결국 전부 같은 방향(기본 자세의
    # 불안정 방향)으로 넘어진다. 에피소드 내내 유지되는 관절 bias를 로봇마다
    # 랜덤하게 주면 제각각 다른 방향으로 기울다 넘어진다.
    # bias 범위 ±2 → 관절 offset 최대 ±0.5 rad (scale=0.25와 곱해짐)
    action_bias = 2.0 * (2.0 * torch.rand(env.num_envs, action_dim, device=env.device) - 1.0)

    while simulation_app.is_running() and step_count < args_cli.max_steps:
        # 유지되는 성향(bias) + 매 스텝 랜덤 노이즈 = 로봇마다 다른 허우적거림
        noise = 0.5 * torch.randn(env.num_envs, action_dim, device=env.device)
        action = (action_bias + noise).clamp(-2.0, 2.0)

        obs, _, _, _, _ = env.step(action)
        step_count += 1

        # 100스텝마다 넘어진 로봇 수 집계 (몸체 높이 0.35m 미만 = 넘어짐)
        if step_count % 100 == 0:
            z = env.scene["robot"].data.root_pos_w[:, 2]
            fallen = (z < 0.35).sum().item()
            print(f"  [{step_count:4d}/{args_cli.max_steps}] "
                  f"넘어진 로봇: {fallen:3d}/{env.num_envs} | "
                  f"평균 높이: {z.mean().item():.3f} m")

    # ── 최종 통계 ────────────────────────────────────────────────────────
    z = env.scene["robot"].data.root_pos_w[:, 2]
    fallen = (z < 0.35).sum().item()

    # 넘어진 방향 분포: 시작 위치 대비 몸체가 이동한 방향을 4분면으로 분류
    delta = env.scene["robot"].data.root_pos_w[:, :2] - start_pos
    angle = torch.atan2(delta[:, 1], delta[:, 0])  # -pi ~ pi (x+ = 앞)
    front = ((angle > -0.785) & (angle <= 0.785)).sum().item()
    left = ((angle > 0.785) & (angle <= 2.356)).sum().item()
    back = ((angle > 2.356) | (angle <= -2.356)).sum().item()
    right = ((angle > -2.356) & (angle <= -0.785)).sum().item()

    print("\n" + "=" * 70)
    print("[결과] 랜덤 액션 실행 완료")
    print(f"  총 스텝:      {step_count}")
    print(f"  넘어진 로봇:  {fallen}/{env.num_envs} ({fallen / env.num_envs * 100:.0f}%)")
    print(f"  평균 높이:    {z.mean().item():.3f} m  (서 있을 때 0.74 m)")
    print(f"  넘어진 방향:  앞 {front} | 뒤 {back} | 왼쪽 {left} | 오른쪽 {right}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 35_1 종료")
