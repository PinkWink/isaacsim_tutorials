"""
31_urdf_to_usd.py - URDF → USD 변환 및 Isaac Sim 스폰

30강에서 준비한 Pinky Pro URDF를 IsaacLab의 UrdfConverter로 USD로 변환하고,
변환된 USD를 Isaac Sim에 스폰하여 로봇 구조와 재질을 확인합니다:
  - UrdfConverterCfg: 변환 옵션 (fix_base, merge_fixed_joints, collision 등)
  - 변환된 USD를 ArticulationCfg로 로드
  - 관절 구조/바디 구조 출력 및 검증
  - 바퀴 관절에 velocity target을 적용하여 동작 테스트

실행:
    source env_isaaclab/bin/activate
    python lectures/31_urdf_to_usd/31_urdf_to_usd.py

    # GUI 없이 실행
    python lectures/31_urdf_to_usd/31_urdf_to_usd.py --headless
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="31 - URDF → USD 변환 + 스폰")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils import configclass

# ── 3. 경로 설정 ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "..", "30_urdf_preparation", "pinky_pro.urdf")
URDF_PATH = os.path.abspath(URDF_PATH)
USD_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "usd_output")

assert os.path.exists(URDF_PATH), (
    f"URDF 파일을 찾을 수 없습니다: {URDF_PATH}\n"
    f"먼저 30강을 실행하세요: python lectures/30_urdf_preparation/30_urdf_preparation.py"
)

# ══════════════════════════════════════════════════════════════════════════
#  URDF → USD 변환
# ══════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("[1단계] URDF → USD 변환")
print("=" * 70)
print(f"  URDF 입력: {URDF_PATH}")
print(f"  USD 출력:  {USD_OUTPUT_DIR}/")

# UrdfConverterCfg: 변환 옵션 설정
urdf_cfg = UrdfConverterCfg(
    asset_path=URDF_PATH,
    usd_dir=USD_OUTPUT_DIR,
    usd_file_name="pinky_pro.usd",

    # ── 물리 설정 ────────────────────────────────────────────────────
    fix_base=False,                    # 모바일 로봇이므로 고정하지 않음 (floating-base)
    merge_fixed_joints=True,           # fixed joint로 연결된 링크를 병합 (성능 향상)
    self_collision=False,              # 자체 충돌 비활성화

    # ── 충돌 설정 ────────────────────────────────────────────────────
    collision_from_visuals=False,      # URDF에 collision 메시가 별도로 있으므로 False
    collider_type="convex_hull",       # 충돌 메시 처리 방식

    # ── 관절 드라이브 설정 ────────────────────────────────────────────
    # 바퀴 관절에 velocity 제어가 가능하도록 설정
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        drive_type="force",
        target_type="velocity",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
            stiffness=0.0,       # 위치 제어 비활성
            damping=10.0,        # 속도 제어용 댐핑
        ),
    ),

    # ── 변환 옵션 ────────────────────────────────────────────────────
    force_usd_conversion=True,         # 항상 재변환 (개발 중)
    make_instanceable=False,           # 단일 로봇이므로 instanceable 불필요
)

# 변환 실행
converter = UrdfConverter(urdf_cfg)
usd_path = converter.usd_path

print(f"\n  변환 완료!")
print(f"  USD 파일: {usd_path}")
print(f"  파일 크기: {os.path.getsize(usd_path):,} bytes")

# ══════════════════════════════════════════════════════════════════════════
#  변환된 USD로 로봇 스폰
# ══════════════════════════════════════════════════════════════════════════

print(f"\n{'=' * 70}")
print("[2단계] USD 로봇 스폰 및 구조 확인")
print("=" * 70)

# Pinky Pro ArticulationCfg
PINKY_USD_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.05),     # 바닥 위 살짝 띄워서 스폰
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*wheel.*"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)


@configclass
class PinkyUsdSceneCfg(InteractiveSceneCfg):
    """변환된 Pinky Pro USD 스폰 씬."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=500.0, color=(0.8, 0.8, 0.85)),
    )
    distant_light = AssetBaseCfg(
        prim_path="/World/DistantLight",
        spawn=sim_utils.DistantLightCfg(intensity=800.0, color=(1.0, 1.0, 0.95)),
    )
    robot: ArticulationCfg = PINKY_USD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ══════════════════════════════════════════════════════════════════════════
#  차동 구동 유틸리티
# ══════════════════════════════════════════════════════════════════════════

# Pinky Pro 물리 파라미터
WHEEL_RADIUS = 0.028     # 바퀴 반지름 (m) — URDF의 sphere radius
WHEEL_BASE = 0.0811      # 바퀴 간 거리 (m) — 좌우 wheel origin y 간격 (0.04055 * 2)


def unicycle_to_wheel_vel(v: float, omega: float) -> tuple:
    """유니사이클 (v, ω) → 좌/우 바퀴 각속도 (rad/s)."""
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
#  시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """Pinky Pro의 구조를 확인하고 바퀴 제어를 테스트합니다."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # ── 관절 구조 출력 ────────────────────────────────────────────────────
    print(f"\n  총 관절 수: {robot.num_joints}")
    for i, name in enumerate(robot.joint_names):
        print(f"    {i:3d} | {name}")

    # ── 바디 구조 출력 ────────────────────────────────────────────────────
    print(f"\n  총 바디 수: {robot.num_bodies}")
    for i, name in enumerate(robot.body_names):
        pos = robot.data.body_pos_w[0, i]
        print(f"    {i:3d} | {name:<30s} ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})")

    # ── 바퀴 관절 ID 탐색 ────────────────────────────────────────────────
    left_ids, left_names = robot.find_joints("l_wheel.*")
    right_ids, right_names = robot.find_joints("r_wheel.*")
    wheel_ids = left_ids + right_ids
    print(f"\n  좌측 바퀴: {left_names} (ID: {left_ids})")
    print(f"  우측 바퀴: {right_names} (ID: {right_ids})")

    # ── 모션 패턴 ────────────────────────────────────────────────────────
    PATTERNS = [
        ("전진", 0.1, 0.0, 300),
        ("좌회전", 0.0, 2.0, 200),
        ("전진", 0.1, 0.0, 300),
        ("우회전", 0.0, -2.0, 200),
        ("정지", 0.0, 0.0, 200),
    ]
    total_steps = sum(p[3] for p in PATTERNS)
    ep_step = 0

    print(f"\n{'=' * 70}")
    print("[3단계] 바퀴 제어 테스트")
    print(f"  R={WHEEL_RADIUS}m, L={WHEEL_BASE}m")
    print("=" * 70)

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # 리셋
        if ep_step % (total_steps + 200) == 0:
            ep_step = 0
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            root_state[:, 2] += 0.05
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            print("\n[리셋]")

        # 패턴 결정
        v, omega = 0.0, 0.0
        cumulative = 0
        for name, pv, pw, dur in PATTERNS:
            if cumulative <= ep_step < cumulative + dur:
                v, omega = pv, pw
                if ep_step == cumulative:
                    print(f"  [Phase] {name} (v={v:.2f}, ω={omega:.2f})")
                break
            cumulative += dur

        # 바퀴 속도 적용
        wl, wr = unicycle_to_wheel_vel(v, omega)
        wheel_vel = torch.zeros(robot.num_instances, len(wheel_ids), device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ep_step += 1

        if ep_step % 100 == 0:
            pos = robot.data.root_pos_w[0]
            print(f"  [Step {ep_step:5d}] pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.3, 0.3, 0.3], target=[0.0, 0.0, 0.05])

    scene_cfg = PinkyUsdSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 31강 완료")
