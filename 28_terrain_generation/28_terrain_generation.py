"""
28_terrain_generation.py - 학습된 보행 정책을 다양한 지형에서 실행
                            (도메인 시프트 / domain shift 데모)

26강에서 평탄 지형(Flat) 위 ANYmal-B로 학습한 PPO 정책을, 본 강의에서 절차적으로
생성한 거친 지형(계단, 랜덤 그리드, 파도) 위에서 그대로 실행합니다.
학습 분포(Flat)와 다른 지형에서 정책 성능이 어떻게 떨어지는지를 관찰함으로써,
"환경이 바뀌면 다시 학습해야 한다"는 강화학습의 근본적 한계를 체험합니다.

핵심 아이디어:
  - 26강 환경(Isaac-Velocity-Flat-Anymal-B-v0)의 cfg를 그대로 로드 → 관측/행동
    공간이 정책과 일치하므로 체크포인트가 깨끗하게 로드됩니다.
  - cfg의 scene.terrain만 평면 → 절차적 거친 지형으로 교체합니다.
  - 정책 입장에서는 입력 차원은 같지만 발 밑 물리가 달라집니다 = 순수 domain shift.

관찰 포인트:
  - Flat에서는 timeout(20초 생존)까지 가던 정책이
  - Rough에서는 계단/턱에서 발이 걸려 일찍 종료(fail)되는 빈도 증가
  - 평균 에피소드 길이 감소, 평균 보상 감소
  → 더 다양한 지형에서 재학습(curriculum / domain randomization)이 필요함을 확인

대안 (실제 해결책):
  - Isaac-Velocity-Rough-Anymal-B-v0로 처음부터 학습 (height scanner 관측 포함)
  - Flat 정책에서 시작해 Rough로 fine-tuning
  - 29강: 도메인 랜덤화로 sim-to-real gap 완화

실행:
    source env_isaaclab/bin/activate
    cd lectures/28_terrain_generation

    # 26강 체크포인트 자동 탐색
    python 28_terrain_generation.py --num_envs 16

    # 체크포인트 직접 지정
    python 28_terrain_generation.py --checkpoint /path/to/model_299.pt

    # headless로 통계만 빠르게 수집
    python 28_terrain_generation.py --headless --num_envs 64 --num_episodes 50

    # 정책 없이 (랜덤 행동) 비교
    python 28_terrain_generation.py --random --num_envs 16
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import glob
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="28 - 학습 정책을 거친 지형에서 실행 (domain shift 데모)")
parser.add_argument("--num_envs", type=int, default=16, help="평가할 환경 수")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="26강 체크포인트(.pt). 미지정 시 자동 탐색")
parser.add_argument("--random", action="store_true",
                    help="학습 정책 대신 랜덤 행동 (비교용)")
parser.add_argument("--num_episodes", type=int, default=20,
                    help="평가할 에피소드 수")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-Anymal-B-v0",
                    help="베이스 환경 (26강 학습 시와 동일해야 정책이 로드됨)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
import isaaclab_tasks  # noqa: F401

from isaaclab.terrains import TerrainImporterCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.terrains.trimesh.mesh_terrains_cfg import (
    MeshPlaneTerrainCfg,
    MeshPyramidStairsTerrainCfg,
    MeshRandomGridTerrainCfg,
)
from isaaclab.terrains.height_field.hf_terrains_cfg import HfWaveTerrainCfg
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
#  거친 지형 정의 (TerrainGeneratorCfg)
# ══════════════════════════════════════════════════════════════════════════
#
# 26강 학습 시에는 평평한 지면(MeshPlaneTerrainCfg 단일)이었습니다.
# 여기서는 4종의 거친 지형을 섞어 학습 분포 밖의 상황을 만듭니다.
# curriculum=True로 row마다 난이도가 올라가, 같은 정책이 쉬운 지형/어려운 지형에서
# 각각 어떻게 행동하는지 비교 관찰할 수 있습니다.

TUTORIAL_TERRAIN_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    num_rows=4,                 # 난이도 4단계
    num_cols=4,                 # 지형 유형 4종
    border_width=5.0,

    curriculum=True,            # row가 클수록 어려움
    difficulty_range=(0.0, 1.0),

    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,

    color_scheme="height",

    sub_terrains={
        # 1) 평탄 — 학습 분포와 거의 동일, 정책이 잘 동작해야 정상
        "flat": MeshPlaneTerrainCfg(proportion=0.25),

        # 2) 피라미드 계단 — 학습한 적 없는 수직 변위, 발이 걸려 넘어지기 쉬움
        "pyramid_stairs": MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.05, 0.20),
            step_width=0.3,
            platform_width=2.0,
            border_width=0.5,
        ),

        # 3) 랜덤 그리드 — 작은 턱이 무작위로 깔린 면, 발 접촉이 불안정
        "random_grid": MeshRandomGridTerrainCfg(
            proportion=0.25,
            grid_width=0.45,
            grid_height_range=(0.02, 0.15),
            platform_width=2.0,
        ),

        # 4) 파도 지형 — 연속적인 경사, 학습 시 보지 못한 경사면 적응 필요
        "wave": HfWaveTerrainCfg(
            proportion=0.25,
            amplitude_range=(0.02, 0.15),
            num_waves=3,
            border_width=0.25,
        ),
    },
)


# ══════════════════════════════════════════════════════════════════════════
#  26강 학습 설정과 일치하는 Runner Cfg (정책 로드 호환성)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class AnymalEvalRunnerCfg(RslRlOnPolicyRunnerCfg):
    """추론용 Runner — actor/critic 네트워크 형상이 26강과 동일해야 합니다."""
    num_steps_per_env = 24
    max_iterations = 0
    save_interval = 50
    experiment_name = "anymal_domain_shift_eval"
    run_name = ""
    logger = "tensorboard"

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[128, 128, 128],
        critic_hidden_dims=[128, 128, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0, use_clipped_value_loss=True,
        clip_param=0.2, entropy_coef=0.01,
        num_learning_epochs=5, num_mini_batches=4,
        learning_rate=1.0e-3, schedule="adaptive",
        gamma=0.99, lam=0.95,
        desired_kl=0.01, max_grad_norm=1.0,
    )


# ══════════════════════════════════════════════════════════════════════════
#  체크포인트 자동 탐색 (27강과 동일)
# ══════════════════════════════════════════════════════════════════════════


def find_latest_checkpoint() -> str | None:
    """26강에서 저장된 가장 최근 체크포인트를 자동으로 찾습니다."""
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

        direct_ckpts = sorted(
            glob.glob(os.path.join(log_dir, "model_*.pt")),
            key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]),
        )
        if direct_ckpts:
            return direct_ckpts[-1]

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
    """26강 정책을 거친 지형에서 실행하여 도메인 시프트의 영향을 관찰합니다."""

    # ── 환경 cfg 로드 (26강 학습 시와 동일한 Flat-Anymal-B cfg) ────────────
    task_name = args_cli.task
    print(f"[INFO] 베이스 환경: {task_name}")

    env_cfg = load_cfg_from_registry(task_name, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs

    # ── 핵심: scene.terrain만 평면 → 거친 지형으로 교체 ────────────────────
    # 관측/행동 공간은 그대로이므로 26강 정책이 그대로 로드됩니다.
    # 다만 발 밑 물리만 바뀌어, 정책이 본 적 없는 상황을 마주합니다.
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TUTORIAL_TERRAIN_CFG,
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        debug_vis=True,
    )

    # ── 환경 생성 ────────────────────────────────────────────────────────
    env = gym.make(task_name, cfg=env_cfg)
    unwrapped = env.unwrapped

    # ── 지형 정보 출력 ───────────────────────────────────────────────────
    tcfg = TUTORIAL_TERRAIN_CFG
    print("=" * 70)
    print("[INFO] 거친 지형 위에 26강 학습 정책을 적용합니다")
    print(f"  환경 수:        {unwrapped.num_envs}")
    print(f"  관측 차원:      {unwrapped.observation_space}")
    print(f"  액션 차원:      {unwrapped.action_space}")
    print(f"  지형 그리드:    {tcfg.num_rows}행 × {tcfg.num_cols}열 (난이도 0.0→1.0)")
    print("  지형 종류:")
    for name, sub in tcfg.sub_terrains.items():
        print(f"    - {name:18s} (비율 {sub.proportion:.0%})")
    print("=" * 70)

    # ── RSL-RL 래퍼 ──────────────────────────────────────────────────────
    env = RslRlVecEnvWrapper(env)
    device = str(unwrapped.device)

    # ── 정책 로드 ────────────────────────────────────────────────────────
    policy = None
    use_random = args_cli.random

    if not use_random:
        checkpoint_path = args_cli.checkpoint or find_latest_checkpoint()
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[INFO] 체크포인트: {checkpoint_path}")
            agent_cfg = AnymalEvalRunnerCfg()
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=device)
            runner.load(checkpoint_path, load_optimizer=False)
            policy = runner.get_inference_policy(device=device)
            print("[INFO] 26강 학습 정책 로드 완료")
        else:
            print("[WARNING] 체크포인트를 찾을 수 없습니다. 랜덤 행동으로 평가합니다.")
            print("  먼저 26강 학습을 실행하세요:")
            print("    python ../26_train_locomotion/26_train_locomotion.py --headless")
            use_random = True

    mode_name = "랜덤" if use_random else "26강 학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] '{mode_name}' 정책을 거친 지형에서 실행")
    print(f"  평가 에피소드: {args_cli.num_episodes}")
    print(f"{'=' * 70}\n")

    # ── 평가 루프 ────────────────────────────────────────────────────────
    obs = env.get_observations()
    num_envs = unwrapped.num_envs
    action_dim = unwrapped.action_space.shape[-1]

    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device, dtype=torch.long)

    all_rewards: list[float] = []
    all_lengths: list[int] = []
    fail_count = 0
    timeout_count = 0

    while simulation_app.is_running() and len(all_rewards) < args_cli.num_episodes:
        if use_random:
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

                is_timeout = bool(truncated[idx].item())
                if is_timeout:
                    timeout_count += 1
                    reason = "timeout(생존)"
                else:
                    fail_count += 1
                    reason = "fail(넘어짐)"

                print(
                    f"  [{len(all_rewards):3d}/{args_cli.num_episodes}] "
                    f"보상: {ep_rew:+9.2f} | 길이: {ep_len:4d} | {reason}"
                )

                if len(all_rewards) >= args_cli.num_episodes:
                    break

            episode_rewards[done] = 0.0
            episode_lengths[done] = 0

    # ── 통계 및 도메인 시프트 해석 ──────────────────────────────────────────
    if all_rewards:
        n = len(all_rewards)
        avg_reward = sum(all_rewards) / n
        avg_length = sum(all_lengths) / n
        max_len = max(all_lengths)
        survival_rate = timeout_count / n * 100
        fall_rate = fail_count / n * 100

        print(f"\n{'=' * 70}")
        print(f"[결과] '{mode_name}' 정책 — 거친 지형 평가")
        print(f"  에피소드 수:      {n}")
        print(f"  평균 보상:        {avg_reward:.2f}")
        print(f"  평균 길이:        {avg_length:.1f} 스텝 (최대 관측 {max_len})")
        print(f"  최대/최소 보상:   {max(all_rewards):.2f} / {min(all_rewards):.2f}")
        print(f"  생존(timeout):    {timeout_count} / {n}  ({survival_rate:.1f}%)")
        print(f"  넘어짐(fail):     {fail_count} / {n}  ({fall_rate:.1f}%)")
        print(f"{'=' * 70}")

        if not use_random:
            print("\n[해석] 도메인 시프트 (domain shift)")
            print("  26강에서는 Flat 지형에서 timeout까지 안정 보행했지만,")
            print("  거친 지형에서는 fail 비율이 크게 증가합니다.")
            print("  → 학습 분포(Flat)와 다른 환경에서 정책은 일반화되지 않습니다.")
            print("\n[해결 방향]")
            print("  1) 학습 단계에서 더 다양한 지형 사용 (terrain curriculum)")
            print("     예: Isaac-Velocity-Rough-Anymal-B-v0로 처음부터 학습")
            print("  2) Flat 정책에서 시작해 거친 지형으로 fine-tuning")
            print("  3) 29강 - 도메인 랜덤화: 물리 파라미터 다양화로 robust한 정책 학습")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 도메인 시프트 데모 종료")
