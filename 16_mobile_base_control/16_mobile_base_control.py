"""
16_mobile_base_control.py - JetBot 경로 주행 및 오도메트리

JetBot을 차동 구동으로 제어하여 다양한 경로를 주행하고,
바퀴 회전량으로부터 이동 거리를 추정하는 오도메트리를 구현합니다:
  - 사각형 경로 (Square): 직진 + 90도 회전 반복
  - 원형 경로 (Circle): 선속도 + 일정 각속도
  - 8자 경로 (Figure-8): 좌/우 원호를 번갈아 주행
  - 바퀴 오도메트리: 좌/우 바퀴 회전량 → (x, y, θ) 추정
  - 홀로노믹 vs 비홀로노믹 차이 설명

실행:
    source env_isaaclab/bin/activate
    cd lectures/16_mobile_base_control
    python 16_mobile_base_control.py --num_envs 2

    # GUI 없이 실행
    python 16_mobile_base_control.py --headless
"""

# ── 1. AppLauncher 패턴 ──────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="16 - JetBot 경로 주행 + 오도메트리")
parser.add_argument("--num_envs", type=int, default=2, help="생성할 환경 수")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ── 3. JetBot 설정 (15번과 동일) ─────────────────────────────────────────

JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

WHEEL_RADIUS = 0.03
WHEEL_BASE = 0.1125


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg
# ══════════════════════════════════════════════════════════════════════════


@configclass
class MobileBaseSceneCfg(InteractiveSceneCfg):
    """JetBot 경로 주행 씬."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ══════════════════════════════════════════════════════════════════════════
# 차동 구동 유틸리티
# ══════════════════════════════════════════════════════════════════════════


def unicycle_to_wheel_vel(v: float, omega: float) -> tuple:
    """유니사이클 (v, ω) → 좌/우 바퀴 각속도 (rad/s)."""
    v_left = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
# 경로 패턴 정의
# ══════════════════════════════════════════════════════════════════════════

# 사각형 경로: (이름, v, omega, 스텝 수)
# 직진 150스텝(1.5초) + 90도 회전(π/2 rad) 반복
TURN_90_OMEGA = math.pi / 2.0   # 1.5708 rad/s
TURN_90_STEPS = 100             # 100스텝 × 0.01s = 1.0초 → π/2 rad

SQUARE_PATTERN = [
    ("직진 1", 0.2, 0.0, 150),
    ("90도 좌회전", 0.0, TURN_90_OMEGA, TURN_90_STEPS),
    ("직진 2", 0.2, 0.0, 150),
    ("90도 좌회전", 0.0, TURN_90_OMEGA, TURN_90_STEPS),
    ("직진 3", 0.2, 0.0, 150),
    ("90도 좌회전", 0.0, TURN_90_OMEGA, TURN_90_STEPS),
    ("직진 4", 0.2, 0.0, 150),
    ("90도 좌회전", 0.0, TURN_90_OMEGA, TURN_90_STEPS),
]

# 원형 경로: v=0.2 m/s, ω=1.0 rad/s → 반지름 r = v/ω = 0.2m
CIRCLE_PATTERN = [
    ("원형 주행", 0.2, 1.0, 630),   # 약 1바퀴 (2π/ω ≈ 6.28초 = 628스텝)
]

# 8자 경로: 좌측 반원 + 우측 반원 반복
FIGURE8_PATTERN = [
    ("좌측 반원", 0.2, 1.0, 315),    # π/ω ≈ 3.14초 = 314스텝 (반원)
    ("우측 반원", 0.2, -1.0, 315),   # 반대 방향 반원
    ("좌측 반원", 0.2, 1.0, 315),
    ("우측 반원", 0.2, -1.0, 315),
]

# 전체 시퀀스: 사각형 → 정지 → 원형 → 정지 → 8자 → 정지
PAUSE = [("정지", 0.0, 0.0, 200)]
ALL_PATTERNS = (
    SQUARE_PATTERN + PAUSE +
    CIRCLE_PATTERN + PAUSE +
    FIGURE8_PATTERN + PAUSE
)


# ══════════════════════════════════════════════════════════════════════════
# 오도메트리 클래스
# ══════════════════════════════════════════════════════════════════════════


class WheelOdometry:
    """바퀴 회전량으로 로봇의 (x, y, θ)를 추정하는 오도메트리.

    차동 구동 오도메트리 공식:
      Δs_L = R × Δφ_L      (좌측 바퀴 이동 거리)
      Δs_R = R × Δφ_R      (우측 바퀴 이동 거리)
      Δs   = (Δs_R + Δs_L) / 2   (로봇 중심 이동 거리)
      Δθ   = (Δs_R - Δs_L) / L   (로봇 회전량)
      x   += Δs × cos(θ + Δθ/2)
      y   += Δs × sin(θ + Δθ/2)
      θ   += Δθ
    """

    def __init__(self, num_envs: int, device: str):
        self.x = torch.zeros(num_envs, device=device)
        self.y = torch.zeros(num_envs, device=device)
        self.theta = torch.zeros(num_envs, device=device)
        self.prev_left = torch.zeros(num_envs, device=device)
        self.prev_right = torch.zeros(num_envs, device=device)
        self.total_dist = torch.zeros(num_envs, device=device)

    def reset(self, left_pos, right_pos):
        """초기 바퀴 위치를 설정합니다."""
        self.x[:] = 0.0
        self.y[:] = 0.0
        self.theta[:] = 0.0
        self.prev_left = left_pos.clone()
        self.prev_right = right_pos.clone()
        self.total_dist[:] = 0.0

    def update(self, left_pos, right_pos):
        """바퀴 위치 변화로 오도메트리를 갱신합니다."""
        d_left = (left_pos - self.prev_left) * WHEEL_RADIUS
        d_right = (right_pos - self.prev_right) * WHEEL_RADIUS

        ds = (d_right + d_left) / 2.0
        dtheta = (d_right - d_left) / WHEEL_BASE

        mid_theta = self.theta + dtheta / 2.0
        self.x += ds * torch.cos(mid_theta)
        self.y += ds * torch.sin(mid_theta)
        self.theta += dtheta
        self.total_dist += torch.abs(ds)

        self.prev_left = left_pos.clone()
        self.prev_right = right_pos.clone()


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """다양한 경로 패턴으로 JetBot을 구동하고 오도메트리를 추적합니다."""

    robot = scene["robot"]
    sim_dt = sim.get_physics_dt()

    # ── 관절 ID 조회 ─────────────────────────────────────────────────────
    left_ids, _ = robot.find_joints("left_wheel_joint")
    right_ids, _ = robot.find_joints("right_wheel_joint")
    wheel_ids = left_ids + right_ids
    print(f"[INFO] Wheel IDs: left={left_ids}, right={right_ids}")

    # ── 오도메트리 초기화 ────────────────────────────────────────────────
    odom = WheelOdometry(scene.num_envs, robot.device)

    # ── 패턴 상태 ────────────────────────────────────────────────────────
    total_steps = sum(p[3] for p in ALL_PATTERNS)
    ep_step = 0
    current_pattern_name = ""

    print(f"[INFO] dt={sim_dt:.4f}s, 전체 패턴 {len(ALL_PATTERNS)}개, {total_steps} 스텝")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 에피소드 리셋 ────────────────────────────────────────────
        if ep_step % (total_steps + 300) == 0:
            ep_step = 0

            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()

            # 오도메트리 리셋
            left_pos = robot.data.joint_pos[:, left_ids[0]]
            right_pos = robot.data.joint_pos[:, right_ids[0]]
            odom.reset(left_pos, right_pos)

            print(f"\n[에피소드 리셋]")
            print(f"  ★ 사각형 경로 → 원형 경로 → 8자 경로 순서로 실행")

        # ── (2) 현재 패턴의 명령 결정 ────────────────────────────────────
        v, omega = 0.0, 0.0
        cumulative = 0
        for name, pv, pw, dur in ALL_PATTERNS:
            if cumulative <= ep_step < cumulative + dur:
                v, omega = pv, pw
                if ep_step == cumulative and name != current_pattern_name:
                    current_pattern_name = name
                    wl, wr = unicycle_to_wheel_vel(v, omega)
                    print(f"\n  [패턴] {name} (v={v:.2f} m/s, ω={omega:.2f} rad/s)"
                          f" → L={wl:.1f}, R={wr:.1f} rad/s, {dur} 스텝")
                break
            cumulative += dur

        # ── (3) 바퀴 속도 적용 ───────────────────────────────────────────
        wl, wr = unicycle_to_wheel_vel(v, omega)
        wheel_vel = torch.zeros(robot.num_instances, 2, device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        # ── (4) 시뮬레이션 스텝 ──────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        ep_step += 1

        # ── (5) 오도메트리 갱신 ──────────────────────────────────────────
        left_pos = robot.data.joint_pos[:, left_ids[0]]
        right_pos = robot.data.joint_pos[:, right_ids[0]]
        odom.update(left_pos, right_pos)

        # ── (6) 상태 출력 (100 스텝마다) ─────────────────────────────────
        if ep_step % 100 == 0:
            # 환경 로컬 좌표로 변환하여 오도메트리와 동일한 기준으로 비교
            pos_w = robot.data.root_pos_w[0]
            origin = scene.env_origins[0]
            pos_local = pos_w - origin       # 환경 원점 기준 로컬 좌표
            err = math.sqrt(
                (pos_local[0].item() - odom.x[0].item())**2
                + (pos_local[1].item() - odom.y[0].item())**2
            )
            print(
                f"[Step {ep_step:5d}] "
                f"실제=({pos_local[0]:.2f}, {pos_local[1]:.2f}) | "
                f"오도메트리=({odom.x[0]:.2f}, {odom.y[0]:.2f}, "
                f"θ={math.degrees(odom.theta[0].item()):+.1f}°) | "
                f"오차={err:.3f}m | 누적={odom.total_dist[0]:.2f}m"
            )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 2.0], target=[0.5, 0.5, 0.0])

    scene_cfg = MobileBaseSceneCfg(num_envs=args_cli.num_envs, env_spacing=5.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - {args_cli.num_envs}개 환경")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("\n[DONE] 시뮬레이션 종료")
