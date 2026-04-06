"""
12_forward_kinematics.py - 순운동학 (Forward Kinematics)

Franka Panda 로봇의 관절 각도를 변경하면서 말단 장치(end-effector)의
위치와 자세가 어떻게 변하는지 관찰합니다:
  - Forward Kinematics: 관절 각도(theta) → 말단 장치 포즈(x, y, z, qw, qx, qy, qz)
  - FRANKA_PANDA_HIGH_PD_CFG (중력 비활성, 높은 PD 게인)으로 정밀 위치 추종
  - body_pose_w로 EE 포즈 읽기
  - VisualizationMarkers로 EE 프레임 시각화
  - 여러 관절 구성(joint configuration)을 순환하며 FK 결과 비교

실행:
    source env_isaaclab/bin/activate
    python 12_forward_kinematics.py --num_envs 2

    # GUI 없이 실행
    python 12_forward_kinematics.py --headless

    # 환경 수 변경
    python 12_forward_kinematics.py --num_envs 4
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="12 - 순운동학 (Forward Kinematics) 튜토리얼")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# FRANKA_PANDA_HIGH_PD_CFG: 중력 비활성 + 높은 PD 게인
# → 관절 위치 명령에 정확히 추종하므로 FK 시연에 적합
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의
# ══════════════════════════════════════════════════════════════════════════


@configclass
class FKSceneCfg(InteractiveSceneCfg):
    """순운동학 시연을 위한 씬 설정.

    Franka Panda 로봇을 스탠드(마운트) 위에 배치합니다.
    FRANKA_PANDA_HIGH_PD_CFG는 중력이 꺼져 있고 PD 게인이 높아서
    목표 관절 각도에 정확하게 도달합니다.
    """

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # 지면을 스탠드 높이만큼 아래로 내려 로봇이 테이블 위에 있는 것처럼 보이게 함
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # ── 돔 조명 (DomeLight) ──────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ── 스탠드 / 마운트 (테이블 역할) ────────────────────────────────────
    # 로봇 베이스 아래에 배치하여 시각적으로 안정감 제공
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # ── Franka Panda 로봇 (고강성 PD, 중력 비활성) ──────────────────────
    # HIGH_PD_CFG 특징:
    #   - disable_gravity = True (중력 없음 → 관절이 처지지 않음)
    #   - stiffness = 400.0, damping = 80.0 (높은 PD 게인 → 빠르고 정확한 추종)
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """여러 관절 구성을 순환하며 FK 결과를 관찰합니다."""

    # ── 씬에서 엔티티 꺼내기 ─────────────────────────────────────────────
    robot = scene["robot"]

    # ── SceneEntityCfg로 관절/바디 인덱스 확보 ───────────────────────────
    # joint_names: panda_joint1 ~ panda_joint7 (7DOF 팔 관절)
    # body_names: panda_hand (말단 장치 = end-effector)
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)

    # EE body 인덱스 (body_pose_w에서 사용)
    ee_body_idx = robot_entity_cfg.body_ids[0]

    # ── VisualizationMarkers 설정 (EE 프레임 시각화) ─────────────────────
    # FRAME_MARKER_CFG: XYZ 축을 RGB 색상으로 표시하는 좌표 프레임 마커
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(
        frame_marker_cfg.replace(prim_path="/Visuals/ee_frame")
    )

    # ── 순운동학 시연용 관절 구성 정의 ───────────────────────────────────
    # 각 구성은 panda_joint1~7의 목표 각도 (라디안)
    # 서로 다른 관절 각도 → 서로 다른 EE 포즈 (이것이 FK의 핵심!)
    joint_configs = [
        # ── Config 0: 기본 자세 (홈 포지션) ──────────────────────────────
        # 팔이 앞으로 뻗은 기본 자세
        {
            "name": "Home (기본 자세)",
            "joints": [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
        },
        # ── Config 1: 팔을 높이 올린 자세 ────────────────────────────────
        # joint2를 0에 가깝게 → 팔이 위로 올라감
        {
            "name": "Arm Up (팔 올림)",
            "joints": [0.0, -0.3, 0.0, -1.5, 0.0, 2.0, 0.741],
        },
        # ── Config 2: 좌측으로 회전 ──────────────────────────────────────
        # joint1을 +1.0 rad → 베이스가 좌측으로 회전
        {
            "name": "Left Rotate (좌회전)",
            "joints": [1.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741],
        },
        # ── Config 3: 팔을 앞으로 뻗은 자세 ─────────────────────────────
        # joint4(elbow)를 펼치고, joint6(wrist)를 조정
        {
            "name": "Reach Forward (전방 도달)",
            "joints": [0.0, 0.2, 0.0, -1.0, 0.0, 1.8, 0.741],
        },
    ]

    # 관절 구성을 텐서로 변환 (num_envs 브로드캐스트를 위해)
    num_configs = len(joint_configs)
    config_tensors = []
    for cfg in joint_configs:
        # 7개 팔 관절 + 2개 핑거 관절(기본값 0.04)
        joints_7 = cfg["joints"]
        config_tensors.append(joints_7)

    # ── 시간 관련 변수 ───────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    step_count = 0
    current_config_idx = 0
    CONFIG_SWITCH_INTERVAL = 200  # 200스텝마다 관절 구성 변경

    # ── 초기 정보 출력 ───────────────────────────────────────────────────
    print(f"\n[INFO] Physics dt = {sim_dt:.6f}초")
    print(f"[INFO] 환경 수: {scene.num_envs}")
    print(f"[INFO] 로봇 관절 이름: {robot.data.joint_names}")
    print(f"[INFO] 로봇 body 이름: {robot.data.body_names}")
    print(f"[INFO] EE body: panda_hand (body index = {ee_body_idx})")
    print(f"[INFO] 관절 구성 {num_configs}개를 {CONFIG_SWITCH_INTERVAL}스텝마다 순환합니다.")
    print("=" * 70)

    # ── FK 개념 설명 출력 ────────────────────────────────────────────────
    print("\n[순운동학 (Forward Kinematics) 개요]")
    print("  FK란: 관절 각도(θ1, θ2, ..., θ7) → 말단 장치 포즈(x, y, z, qw, qx, qy, qz)")
    print("  같은 관절 각도 → 항상 같은 EE 포즈 (결정론적 매핑)")
    print("  IsaacLab에서는 robot.data.body_pose_w로 FK 결과를 직접 읽을 수 있습니다.")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션 정지/일시정지 처리
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 에피소드 리셋 (시작 시) ───────────────────────────────────
        if step_count == 0:
            # 관절 상태를 기본값으로 초기화
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            print("\n[에피소드 시작] 로봇을 기본 자세로 초기화")

        # ── (2) 관절 구성 전환 (200스텝마다) ─────────────────────────────
        if step_count % CONFIG_SWITCH_INTERVAL == 0:
            current_config_idx = (step_count // CONFIG_SWITCH_INTERVAL) % num_configs
            cfg = joint_configs[current_config_idx]

            print(f"\n{'─' * 70}")
            print(f"[Step {step_count:5d}] 관절 구성 #{current_config_idx}: {cfg['name']}")
            print(f"  목표 관절 각도 (rad): {cfg['joints']}")

        # ── (3) 위치 제어 명령 생성 ──────────────────────────────────────
        # 현재 관절 구성의 목표 각도를 PD 컨트롤러에 전달
        # HIGH_PD_CFG의 높은 강성(stiffness=400)으로 빠르게 목표에 도달
        cfg = joint_configs[current_config_idx]
        target_pos = robot.data.default_joint_pos.clone()

        # 팔 관절(panda_joint1~7)에 목표 각도 설정
        for i, angle in enumerate(cfg["joints"]):
            target_pos[:, robot_entity_cfg.joint_ids[i]] = angle

        robot.set_joint_position_target(target_pos)

        # ── (4) 씬 데이터를 시뮬레이션에 기록 ─────────────────────────────
        scene.write_data_to_sim()

        # ── (5) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()

        # ── (6) 스텝 카운터 증가 ──────────────────────────────────────────
        step_count += 1

        # ── (7) 씬 버퍼 업데이트 ──────────────────────────────────────────
        scene.update(sim_dt)

        # ── (8) FK 결과 읽기: body_pose_w에서 EE 포즈 추출 ────────────────
        # body_pose_w: (num_envs, num_bodies, 7) — [x, y, z, qw, qx, qy, qz]
        # 여기서 ee_body_idx번째 body가 panda_hand (EE)
        ee_pose_w = robot.data.body_pose_w[:, ee_body_idx]  # (num_envs, 7)
        ee_pos = ee_pose_w[:, 0:3]   # 위치: (num_envs, 3)
        ee_quat = ee_pose_w[:, 3:7]  # 자세(quaternion): (num_envs, 4)

        # ── (9) EE 프레임 마커 업데이트 ───────────────────────────────────
        # 모든 환경의 EE 위치에 좌표축 프레임 마커 표시
        ee_marker.visualize(ee_pos, ee_quat)

        # ── (10) 주기적 FK 결과 출력 (구성 전환 후 안정화된 시점) ──────────
        # 구성 전환 후 150스텝 뒤에 출력 (PD 컨트롤러가 충분히 수렴)
        step_in_config = step_count % CONFIG_SWITCH_INTERVAL
        if step_in_config == 150:
            cfg = joint_configs[current_config_idx]
            # 실제 관절 위치 읽기 (env 0 기준)
            actual_joints = robot.data.joint_pos[0, robot_entity_cfg.joint_ids]
            joints_str = ", ".join([f"{v.item():+7.3f}" for v in actual_joints])

            # EE 포즈 (env 0 기준) — 환경 원점 기준으로 오프셋 제거
            ee_pos_local = ee_pos[0] - scene.env_origins[0]
            x, y, z = ee_pos_local.tolist()
            qw, qx, qy, qz = ee_quat[0].tolist()

            print(f"\n  [FK 결과 — Config #{current_config_idx}: {cfg['name']}]")
            print(f"  실제 관절 각도  : [{joints_str}]")
            print(f"  EE 위치 (x,y,z): ({x:+.4f}, {y:+.4f}, {z:+.4f})")
            print(f"  EE 자세 (quat) : (qw={qw:+.4f}, qx={qx:+.4f}, qy={qy:+.4f}, qz={qz:+.4f})")

            # 추가: 모든 링크의 위치도 출력 (FK 체인 이해를 위해)
            print(f"\n  [링크 체인 (link chain) — 각 body의 z좌표]")
            body_pos_w = robot.data.body_pos_w[0]  # (num_bodies, 3)
            body_quat_w = robot.data.body_quat_w[0]  # (num_bodies, 4)
            for i, name in enumerate(robot.data.body_names):
                bx, by, bz = (body_pos_w[i] - scene.env_origins[0]).tolist()
                print(f"    {name:>25s}: ({bx:+.4f}, {by:+.4f}, {bz:+.4f})")

        # ── (11) 에피소드 종료 (4개 구성 모두 순환 후) ────────────────────
        if step_count >= num_configs * CONFIG_SWITCH_INTERVAL:
            step_count = 0
            print(f"\n{'=' * 70}")
            print("[에피소드 종료] 모든 관절 구성을 순환했습니다. 다시 시작합니다.")
            print(f"{'=' * 70}")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """순운동학 시연 씬을 생성하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 카메라 시점: Franka를 비스듬히 위에서 내려다보는 각도
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    scene_cfg = FKSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - {args_cli.num_envs}개 환경")

    # ── sim.reset() - 물리 엔진 초기화 ────────────────────────────────────
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 시작 준비 완료")

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────────
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
