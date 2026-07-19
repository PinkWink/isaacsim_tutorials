"""
36_1_motion_playback.py - 레퍼런스 모션 재생 vs 학습 정책 비교

두 개의 환경을 나란히 띄워 같은 보행 모션을 비교합니다:
  - env 0 (정책 로봇): 36_2에서 학습한 체크포인트로 추론. 중력/물리가
    정상 적용된 채 정책이 스스로 균형을 잡으며 걷습니다.
  - env 1 (재생 로봇): 리타게팅 pkl 데이터를 물리 영향 없이(중력 OFF,
    PD 게인 0) 매 스텝 그대로 기록하는 kinematic 재생 — 학습의 '정답지'.

두 로봇은 같은 위상에서 출발하므로 정책이 정답 동작에 얼마나 가까운지
눈으로 직접 비교할 수 있습니다. 재생 로봇은 자체 시계로 돌아가므로
정책 로봇이 넘어져 리셋되더라도 끊김 없이 계속 재생됩니다.
체크포인트가 없으면 랜덤 정책으로 대체됩니다 (학습 전/후 차이 확인용).

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # 최신 체크포인트 자동 탐색 후 비교 재생 (GUI, 창을 닫으면 종료)
    python 36_1_motion_playback.py

    # 체크포인트 직접 지정 / 실행 시간 제한(초)
    python 36_1_motion_playback.py --checkpoint logs/rsl_rl/g1_walking_tutorial/.../model_1999.pt
    python 36_1_motion_playback.py --duration 30

    # 재생 가능한 모션 목록 확인 (Isaac Sim 부팅 없이 즉시 출력)
    python 36_1_motion_playback.py --motion_list

    # 다른 모션 재생 — 이름만 지정하면 데이터셋 폴더에서 자동 검색
    # (주의: 정책은 기본 보행 클립으로 학습됨 — 다른 모션은 재생 로봇만 유효)
    python 36_1_motion_playback.py --motion_pkl run1_subject2

참고:
    --headless로 실행하면 종료 단계에서 프로세스가 멈추는 경우가 있습니다.
    "[INFO] 비교 재생 종료"가 찍혔다면 작업은 끝난 것이므로 Ctrl+C로
    종료해도 됩니다. (GUI 모드에서는 창을 닫으면 종료)
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_1 - G1 레퍼런스 모션 재생 vs 학습 정책 비교")
parser.add_argument("--checkpoint", type=str, default=None, help="체크포인트 파일 경로 (생략 시 최신 자동 탐색)")
parser.add_argument("--duration", type=float, default=0.0,
                    help="실행 시간(초). 0이면 GUI는 창 닫을 때까지, headless는 15초")
parser.add_argument("--motion_pkl", type=str, default=None,
                    help="모션 이름 또는 pkl 경로 (예: run1_subject2)")
parser.add_argument("--motion_list", action="store_true",
                    help="재생 가능한 모션 목록만 출력하고 종료")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()


def print_motion_list() -> None:
    """데이터셋 폴더의 pkl 목록을 하위 폴더별로 출력합니다.

    g1_walking_env는 isaaclab을 import해서 Isaac Sim 부팅 전에는 쓸 수 없으므로,
    검색 경로(_MOTION_CANDIDATES와 동일)를 여기서 직접 훑습니다.
    """
    this_dir = Path(__file__).resolve().parent
    candidates = [
        this_dir / "g1_retargeted_motions",
        this_dir.parent / "g1" / "g1_retargeted_motions",
    ]
    for base in candidates:
        pkls = sorted(base.glob("**/*.pkl")) if base.exists() else []
        if not pkls:
            continue
        print(f"\n[INFO] 재생 가능한 모션 {len(pkls)}개 — {base}")
        current_sub = None
        for p in pkls:
            sub = p.parent.relative_to(base)
            if sub != current_sub:
                current_sub = sub
                print(f"\n  [{sub}]")
            print(f"    {p.stem}")
        print("\n사용법: python 36_1_motion_playback.py --motion_pkl <이름>")
        print("        (확장자 생략 가능, 부분 일치도 허용)")
        return
    print("[ERROR] 모션 데이터셋 폴더를 찾을 수 없습니다.")
    print(f"  검색 위치: {[str(p) for p in candidates]}")
    print("  아래 명령으로 내려받으세요:")
    print(f"  git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions {candidates[0]}")
    sys.exit(1)


# --motion_list는 Isaac Sim 부팅 없이 목록만 출력하고 바로 종료
if args_cli.motion_list:
    print_motion_list()
    sys.exit(0)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

from isaaclab.envs import ManagerBasedRLEnv

# 공유 환경 모듈 (같은 폴더의 g1_walking_env.py)
# 학습(36_2)과 동일한 관측/네트워크 정의를 써야 체크포인트가 정상 동작합니다.
from g1_walking_env import (
    CYCLE_END,
    CYCLE_START,
    DEFAULT_MOTION_NAME,
    MOTION,
    G1WalkingEnvCfg,
    find_latest_checkpoint,
    make_ppo_runner_cfg,
)

# 환경 역할: env 0 = 학습 정책 추론(물리 정상), env 1 = 레퍼런스 재생(물리 OFF)
POLICY_ENV, GHOST_ENV = 0, 1


def setup_ghost_env(robot, device: str) -> None:
    """env 1(재생 로봇)에서 물리 영향을 제거합니다.

    (1) 링크별 중력 비활성화 — kinematic_enabled=True는 GPU 파이프라인에서
        write_root_pose_to_sim 호출 시 크래시하므로, dynamic 강체를 유지한 채
        PhysX 뷰의 per-link 중력 플래그만 런타임에 끕니다.
    (2) PD 게인 0 — 정책 액션이 만드는 관절 목표를 무시하게 하여, 물리
        서브스텝 동안 재생 자세가 흐트러지지 않게 합니다.
    """
    try:
        view = robot.root_physx_view
        grav = view.get_disable_gravities().clone().reshape(view.count, view.max_links)
        grav[GHOST_ENV, :] = 1
        idx = torch.arange(view.count, dtype=torch.int32, device=grav.device)
        view.set_disable_gravities(grav, idx)
        print(f"[INFO] env {GHOST_ENV}(재생 로봇) 중력 비활성화 완료")
    except Exception as e:
        # 실패해도 매 스텝 상태 덮어쓰기 덕분에 재생 자체는 정상 동작
        print(f"[WARNING] 중력 비활성화 실패({e}) — 상태 덮어쓰기로 재생은 계속됩니다")

    ghost_ids = torch.tensor([GHOST_ENV], dtype=torch.long, device=device)
    zeros = torch.zeros(1, robot.num_joints, device=device)
    robot.write_joint_stiffness_to_sim(zeros, env_ids=ghost_ids)
    robot.write_joint_damping_to_sim(zeros, env_ids=ghost_ids)
    print(f"[INFO] env {GHOST_ENV}(재생 로봇) PD 게인 0 설정 완료")


def main() -> None:
    """학습 정책(env 0)과 레퍼런스 재생(env 1)을 나란히 실행합니다."""

    # ── 레퍼런스 모션 로드 (36_2와 동일해야 관측/위상이 일치) ────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 환경 생성: 무조건 2개 (정책 + 재생) ──────────────────────────────
    env_cfg = G1WalkingEnvCfg()
    env_cfg.scene.num_envs = 2
    env_cfg.sim.device = args_cli.device
    env_cfg.viewer.eye = (5.5, 4.5, 2.5)
    env_cfg.viewer.lookat = (0.0, 1.0, 0.8)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    robot = env.scene["robot"]
    device = env.device
    step_dt = env_cfg.sim.dt * env_cfg.decimation

    # ── 정책 로드 (36_3과 동일 패턴, 없으면 랜덤 정책) ───────────────────
    policy = None
    eval_env = None
    use_random = False

    checkpoint_path = args_cli.checkpoint or find_latest_checkpoint()
    if checkpoint_path and Path(checkpoint_path).exists():
        print(f"[INFO] 체크포인트: {checkpoint_path}")
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from rsl_rl.runners import OnPolicyRunner

        agent_cfg = make_ppo_runner_cfg(max_iterations=0)
        env_wrapped = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(
            env_wrapped, agent_cfg.to_dict(),
            log_dir=None, device=str(device),
        )
        runner.load(checkpoint_path, load_optimizer=False)
        policy = runner.get_inference_policy(device=str(device))
        eval_env = env_wrapped
        print("[INFO] 학습된 정책 로드 완료!")
    else:
        print("[WARNING] 체크포인트를 찾을 수 없습니다. env 0은 랜덤 정책으로 움직입니다.")
        print("  학습된 정책과 비교하려면 먼저 36_2 학습을 실행하세요:")
        print("    python 36_2_train_motion.py --headless")
        use_random = True

    # ── env 1(재생 로봇) 물리 제거 + 재생 데이터 준비 ────────────────────
    setup_ghost_env(robot, device)

    # 학습에 쓰는 것과 같은 구간(기본 클립은 순환 구간, 그 외는 전체 클립)
    if MOTION.pkl_name == DEFAULT_MOTION_NAME:
        s, e = CYCLE_START, CYCLE_END
    else:
        s, e = 0, MOTION.dof_full.shape[0]
    T = e - s
    trans = MOTION.root_trans_full[s:e].to(device)
    quat = MOTION.root_rot_full[s:e][:, [3, 0, 1, 2]].to(device)  # xyzw → wxyz
    origin_offset = trans[0].clone()   # 시작 위치를 env 원점으로 정규화 (xy만)
    origin_offset[2] = 0.0

    ghost_ids = torch.tensor([GHOST_ENV], dtype=torch.long, device=device)
    zero_root_vel = torch.zeros(1, 6, device=device)
    zero_joint_vel = torch.zeros(1, robot.num_joints, device=device)

    # ── 실행 시간 결정 ───────────────────────────────────────────────────
    if args_cli.duration > 0:
        total_steps = int(args_cli.duration / step_dt)
    elif args_cli.headless:
        total_steps = int(15.0 / step_dt)
        print("[INFO] --headless: 기본 15초 실행 (--duration으로 조정)")
    else:
        total_steps = None   # GUI: 창을 닫을 때까지

    mode_name = "랜덤" if use_random else "학습"
    print(f"\n{'=' * 70}")
    print(f"[INFO] 비교 재생 시작 — 같은 위상에서 출발, env {GHOST_ENV}는 리셋과 무관하게 계속 재생")
    print(f"  env {POLICY_ENV}: {mode_name} 정책 추론 (중력 ON, 물리 정상)")
    print(f"  env {GHOST_ENV}: 레퍼런스 kinematic 재생 (중력 OFF, '정답지')")
    print(f"  재생 구간: [{s}:{e}] {T}프레임 @ {MOTION.fps}fps")
    print(f"{'=' * 70}\n")

    # ── 비교 재생 루프 ───────────────────────────────────────────────────
    if use_random:
        obs, _ = env.reset()
        action_dim = env.action_manager.total_action_dim
    else:
        obs = eval_env.get_observations()

    # env 1의 재생 시계는 env 0의 에피소드와 독립: 시작 위상만 env 0과 맞추고,
    # 이후에는 env 0이 넘어져 리셋되더라도 끊김 없이 계속 재생한다.
    ghost_frame = float(MOTION.phase0[POLICY_ENV].item())

    step = 0
    ep_steps = 0
    ep_count = 0
    while simulation_app.is_running() and (total_steps is None or step < total_steps):
        # (1) env 0: 정책 추론 → 두 env 모두 step (env 1의 액션은 무의미)
        if use_random:
            action = torch.randn(env.num_envs, action_dim, device=device).clamp(-1, 1)
            obs, rew, terminated, truncated, info = env.step(action)
            done0 = bool((terminated | truncated)[POLICY_ENV])
            fell0 = bool(terminated[POLICY_ENV])
        else:
            with torch.inference_mode():
                action = policy(obs)
            obs, rew, dones, info = eval_env.step(action)
            timeout = info.get("time_outs", torch.zeros_like(dones)).bool()
            done0 = bool(dones[POLICY_ENV])
            fell0 = done0 and not bool(timeout[POLICY_ENV])

        # (2) env 1: 레퍼런스 프레임을 그대로 기록 (자체 시계로 계속 재생)
        #     env 0이 넘어져 리셋돼도 여기는 멈추거나 되감지 않는다.
        ghost_frame = (ghost_frame + step_dt * MOTION.fps) % T
        i0 = int(ghost_frame)
        i1 = (i0 + 1) % T
        alpha = ghost_frame - i0

        dof_i = (1 - alpha) * MOTION.dof[i0] + alpha * MOTION.dof[i1]
        trans_i = (1 - alpha) * trans[i0] + alpha * trans[i1]

        ghost_pos = env.scene.env_origins[GHOST_ENV] + (trans_i - origin_offset)
        root_pose = torch.cat([ghost_pos, quat[i0]]).unsqueeze(0)
        robot.write_root_pose_to_sim(root_pose, env_ids=ghost_ids)
        robot.write_root_velocity_to_sim(zero_root_vel, env_ids=ghost_ids)

        joint_pos = robot.data.default_joint_pos[GHOST_ENV:GHOST_ENV + 1].clone()
        joint_pos[:, MOTION.joint_map] = dof_i
        robot.write_joint_state_to_sim(joint_pos, zero_joint_vel, env_ids=ghost_ids)

        # (3) env 0 에피소드 경계 출력
        step += 1
        ep_steps += 1
        if done0:
            ep_count += 1
            reason = "넘어짐" if fell0 else "생존 (10초)"
            print(f"  [에피소드 {ep_count}] env {POLICY_ENV}: {ep_steps * step_dt:.1f}초 — {reason}")
            ep_steps = 0

    print(f"\n[INFO] 비교 재생 종료 — 정답지(env {GHOST_ENV})와 정책(env {POLICY_ENV})의 차이가")
    print("  곧 36_2 학습이 좁혀야 했던 간격입니다.")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 재생 종료")
