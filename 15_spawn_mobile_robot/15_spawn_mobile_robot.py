"""
15_spawn_mobile_robot.py - JetBot 스폰 및 차동 구동(Differential Drive) 제어

NVIDIA JetBot을 스폰하고 차동 구동(differential drive) 방식으로 제어합니다:
  - 유니사이클 모델: 선속도(v)와 각속도(ω)를 좌/우 바퀴 속도로 변환
  - 실제 바퀴 관절(left_wheel_joint, right_wheel_joint)을 사용
  - 바퀴와 지면의 물리적 접촉/마찰로 이동 (dummy joint 아님)
  - 다양한 모션 패턴: 전진, 후진, 제자리 회전, 원호 주행

JetBot 스펙:
  - 바퀴 반지름: 0.03 m
  - 바퀴 간 거리: 0.1125 m
  - 차동 구동 (differential drive): 좌/우 바퀴 속도 차이로 방향 제어

실행:
    source env_isaaclab/bin/activate
    cd lectures/15_spawn_mobile_robot
    python 15_spawn_mobile_robot.py --num_envs 2

    # GUI 없이 실행
    python 15_spawn_mobile_robot.py --headless
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="15 - JetBot 모바일 로봇 스폰 튜토리얼")
parser.add_argument("--num_envs", type=int, default=2, help="생성할 환경 수")
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
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ── 3. JetBot 로봇 설정 ─────────────────────────────────────────────────
# JetBot: NVIDIA의 소형 차동 구동 로봇
# 바퀴 2개(좌/우) + 캐스터 1개 구조
# 바퀴 관절에 velocity target을 적용하여 구동

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
    ),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

# JetBot 물리 파라미터 (DifferentialController 기준)
WHEEL_RADIUS = 0.03       # 바퀴 반지름 (m)
WHEEL_BASE = 0.1125       # 좌우 바퀴 간 거리 (m)


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의
# ══════════════════════════════════════════════════════════════════════════


@configclass
class JetBotSceneCfg(InteractiveSceneCfg):
    """JetBot 스폰 씬 설정."""

    # ── 바닥면 ───────────────────────────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ── 돔 조명 ──────────────────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ── JetBot 로봇 ─────────────────────────────────────────────────────
    robot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ══════════════════════════════════════════════════════════════════════════
# 차동 구동 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v: float, omega: float) -> tuple:
    """유니사이클 모델 → 좌/우 바퀴 각속도 변환.

    차동 구동(differential drive) 로봇의 운동학:
      v_left  = (v - ω × L/2) / R
      v_right = (v + ω × L/2) / R

    Args:
        v: 선속도 (m/s). 양수=전진, 음수=후진.
        omega: 각속도 (rad/s). 양수=반시계 방향(좌회전).

    Returns:
        (left_wheel_vel, right_wheel_vel) 각속도 (rad/s)
    """
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """다양한 모션 패턴으로 JetBot을 구동합니다."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # ── 관절 구조 출력 ────────────────────────────────────────────────────
    print("=" * 70)
    print("  JetBot 관절 구조")
    print("=" * 70)
    print(f"총 관절 수: {robot.num_joints}")
    print(f"환경 수: {scene.num_envs}")
    for i, name in enumerate(robot.joint_names):
        print(f"  {i} | {name}")

    # ── 바퀴 관절 ID 조회 ────────────────────────────────────────────────
    left_ids, _ = robot.find_joints("left_wheel_joint")
    right_ids, _ = robot.find_joints("right_wheel_joint")
    wheel_ids = left_ids + right_ids
    print(f"\n  Left wheel ID:  {left_ids}")
    print(f"  Right wheel ID: {right_ids}")
    print("=" * 70)

    # ── 모션 패턴 정의 ───────────────────────────────────────────────────
    # (이름, 선속도 m/s, 각속도 rad/s, 지속 스텝 수)
    PATTERNS = [
        ("전진 (Forward)",       0.2,  0.0,  300),
        ("후진 (Backward)",     -0.2,  0.0,  300),
        ("좌회전 (Turn Left)",   0.0,  2.0,  200),
        ("우회전 (Turn Right)",  0.0, -2.0,  200),
        ("원호 주행 (Arc)",      0.2,  1.0,  400),
        ("정지 (Stop)",          0.0,  0.0,  200),
    ]

    total_pattern_steps = sum(p[3] for p in PATTERNS)
    ep_step = 0

    print(f"\n[INFO] dt={sim_dt:.4f}s, wheel_r={WHEEL_RADIUS}m, wheel_base={WHEEL_BASE}m")
    print(f"[INFO] 모션 패턴 {len(PATTERNS)}개, 총 {total_pattern_steps} 스텝")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 에피소드 리셋 ────────────────────────────────────────────
        if ep_step % (total_pattern_steps + 200) == 0:
            ep_step = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            scene.reset()

            pos = robot.data.root_pos_w[0]
            print(f"\n[에피소드 리셋] pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")

        # ── (2) 현재 패턴의 속도 명령 결정 ───────────────────────────────
        v, omega = 0.0, 0.0
        cumulative = 0
        for name, pv, pw, dur in PATTERNS:
            if cumulative <= ep_step < cumulative + dur:
                v, omega = pv, pw
                if ep_step == cumulative:
                    wl, wr = unicycle_to_wheel_vel(v, omega)
                    print(f"  [Phase] {name} (v={v:.2f}, ω={omega:.2f}) → "
                          f"wheel L={wl:.1f}, R={wr:.1f} rad/s")
                break
            cumulative += dur

        # ── (3) 유니사이클 → 바퀴 속도 변환 및 적용 ─────────────────────
        wl, wr = unicycle_to_wheel_vel(v, omega)

        wheel_vel = torch.zeros(robot.num_instances, 2, device=robot.device)
        wheel_vel[:, 0] = wl   # left wheel
        wheel_vel[:, 1] = wr   # right wheel
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        # ── (4) 시뮬레이션 스텝 ──────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ep_step += 1

        # ── (5) 상태 출력 (100 스텝마다) ─────────────────────────────────
        if ep_step % 100 == 0:
            pos = robot.data.root_pos_w[0]
            vel = robot.data.root_lin_vel_w[0]
            print(
                f"[Step {ep_step:5d}] "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}) | "
                f"vel=({vel[0]:.2f}, {vel[1]:.2f}) | "
                f"cmd=(v={v:.2f}, ω={omega:.2f})"
            )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """JetBot을 InteractiveScene으로 생성하고 시뮬레이션을 실행합니다."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.0], target=[0.0, 0.0, 0.0])

    scene_cfg = JetBotSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - {args_cli.num_envs}개 환경")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
