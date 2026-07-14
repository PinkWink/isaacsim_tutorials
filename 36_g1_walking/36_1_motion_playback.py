"""
36_1_motion_playback.py - 레퍼런스 보행 모션 재생 (학습 전 목표 동작 확인)

사람 모션 캡처를 G1으로 리타게팅한 보행 데이터(.pkl)를 물리 없이
kinematic하게 재생합니다. 36_2에서 학습할 "정답 동작"을 먼저 눈으로
확인하는 단계입니다:
  - 매 스텝 레퍼런스의 관절각/루트 pose를 로봇에 직접 기록 (RL 없음)
  - 30fps 데이터를 렌더 주기에 맞춰 선형 보간
  - --use_cycle: 학습에 실제 사용하는 순환 구간(프레임 42~155)만 반복 재생

실행:
    source env_isaaclab/bin/activate
    cd lectures/36_g1_walking

    # 전체 클립 3회 재생 (GUI)
    python 36_1_motion_playback.py

    # 학습용 순환 구간만 반복 재생
    python 36_1_motion_playback.py --use_cycle

    # 다른 모션 재생 — 이름만 지정하면 데이터셋 폴더에서 자동 검색
    # (g1_retargeted_motions/의 ACCAD/lafan1/dance_db/kungfu 하위 폴더)
    python 36_1_motion_playback.py --motion_pkl run1_subject2
    python 36_1_motion_playback.py --motion_pkl walk_turn_left   # 부분 일치도 가능

참고:
    --headless로 실행하면 재생 완료 후 Isaac Sim 종료 단계에서 프로세스가
    멈추는 경우가 있습니다. "[INFO] 재생 완료"가 찍혔다면 작업은 끝난
    것이므로 Ctrl+C로 종료해도 됩니다. (GUI 모드에서는 창을 닫으면 종료)
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="36_1 - G1 레퍼런스 모션 재생")
parser.add_argument("--num_envs", type=int, default=1, help="로봇 수")
parser.add_argument("--loops", type=int, default=3, help="반복 재생 횟수")
parser.add_argument("--use_cycle", action="store_true", help="학습용 순환 구간만 재생")
parser.add_argument("--motion_pkl", type=str, default=None,
                    help="모션 이름 또는 pkl 경로 (예: run1_subject2)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import ───────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_apply

# 공유 환경 모듈 (같은 폴더의 g1_walking_env.py)
from g1_walking_env import CYCLE_END, CYCLE_START, MOTION, G1SceneCfg


def main() -> None:
    """레퍼런스 모션을 kinematic하게 재생합니다."""

    # ── 모션 로드 ────────────────────────────────────────────────────────
    MOTION.load_file(args_cli.motion_pkl)

    # ── 시뮬레이션 / 씬 생성 ─────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.0, 3.0, 2.5], target=[0.0, 0.0, 0.8])

    scene = InteractiveScene(G1SceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0))
    sim.reset()
    robot = scene["robot"]
    device = robot.device

    # ── 재생할 프레임 구간 선택 ──────────────────────────────────────────
    if args_cli.use_cycle:
        dof = MOTION.dof_full[CYCLE_START:CYCLE_END].to(device)
        root_trans = MOTION.root_trans_full[CYCLE_START:CYCLE_END].to(device)
        root_rot = MOTION.root_rot_full[CYCLE_START:CYCLE_END].to(device)
        seg_name = f"순환 구간 [{CYCLE_START}:{CYCLE_END}]"
    else:
        dof = MOTION.dof_full.to(device)
        root_trans = MOTION.root_trans_full.to(device)
        root_rot = MOTION.root_rot_full.to(device)
        seg_name = "전체 클립"

    num_frames = dof.shape[0]
    duration = num_frames / MOTION.fps

    # ── 관절 매핑 (pkl 23관절 → 로봇 관절 인덱스) ────────────────────────
    joint_map = MOTION.build_joint_map(robot.data.joint_names, device)
    print("\n[INFO] 관절 매핑 (pkl → 로봇):")
    from g1_walking_env import ALIAS, PKL_JOINT_NAMES
    for pkl_name in PKL_JOINT_NAMES:
        robot_name = ALIAS.get(pkl_name, pkl_name)
        mark = "  (alias)" if pkl_name in ALIAS else ""
        print(f"  {pkl_name:32s} → {robot_name}{mark}")

    print(f"\n[INFO] {seg_name}: {num_frames}프레임 @ {MOTION.fps}fps = {duration:.2f}초")
    print(f"[INFO] {args_cli.loops}회 반복 재생 시작\n")

    # ── 재생 준비 ────────────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    # 시작 위치를 env 원점으로 정규화 (xy만; z는 레퍼런스 그대로)
    origin_offset = root_trans[0].clone()
    origin_offset[2] = 0.0
    env_origins = scene.env_origins  # (num_envs, 3)
    zero_root_vel = torch.zeros(scene.num_envs, 6, device=device)
    zero_joint_vel = torch.zeros(scene.num_envs, robot.num_joints, device=device)
    up_warned = False

    total_steps = int(args_cli.loops * duration / sim_dt)
    for step in range(total_steps):
        if not simulation_app.is_running():
            break

        # ── 현재 시각의 프레임 (선형 보간) ───────────────────────────────
        t = step * sim_dt
        frame_f = (t * MOTION.fps) % num_frames
        i0 = int(frame_f)
        i1 = (i0 + 1) % num_frames if args_cli.use_cycle else min(i0 + 1, num_frames - 1)
        alpha = frame_f - i0

        dof_i = (1 - alpha) * dof[i0] + alpha * dof[i1]
        trans_i = (1 - alpha) * root_trans[i0] + alpha * root_trans[i1]
        # pkl의 쿼터니언은 xyzw 순서 → IsaacLab의 wxyz로 변환 (보간 생략: 30fps면 충분)
        quat_i = root_rot[i0][[3, 0, 1, 2]]

        # ── 루트 pose 기록 ──────────────────────────────────────────────
        pos = env_origins + (trans_i - origin_offset)
        root_pose = torch.cat(
            [pos, quat_i.expand(scene.num_envs, 4)], dim=1
        )
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(zero_root_vel)

        # ── 관절 상태 기록 (기본자세 + 매핑된 23관절 덮어쓰기) ───────────
        joint_pos = robot.data.default_joint_pos.clone()
        joint_pos[:, joint_map] = dof_i
        robot.write_joint_state_to_sim(joint_pos, zero_joint_vel)

        sim.step()
        scene.update(sim_dt)

        # ── 쿼터니언 검증: 몸통 위 방향이 world +z 근처인지 ──────────────
        if not up_warned:
            up = quat_apply(quat_i.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=device))
            if up[0, 2] < 0.9:
                print(f"[WARNING] 몸통 위 방향 z={up[0, 2]:.2f} — 쿼터니언 변환(xyzw→wxyz)을 확인하세요!")
                up_warned = True

        # ── 진행 상황 (loop 단위) ────────────────────────────────────────
        if step % int(duration / sim_dt) == 0:
            loop_no = step // int(duration / sim_dt) + 1
            print(f"  [Loop {loop_no}/{args_cli.loops}] 재생 중... "
                  f"(root z={trans_i[2]:.3f}m)")

    print(f"\n[INFO] 재생 완료 — 이 동작이 36_2 학습의 '정답지'입니다.")
    print("  다음 단계: python 36_2_train_walking.py --headless")


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 재생 종료")
