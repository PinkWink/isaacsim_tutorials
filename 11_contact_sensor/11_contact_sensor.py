"""
11_contact_sensor.py - ContactSensor를 이용한 접촉력 측정

IsaacLab의 ContactSensor를 사용하여 로봇 발에 작용하는 접촉력을 측정하는 방법을 배웁니다:
  - ContactSensorCfg로 접촉 센서 설정
  - prim_path 와일드카드 패턴으로 여러 발(body)에 동시 부착
  - history_length로 과거 접촉력 이력 저장
  - net_forces_w로 월드 프레임 기준 순 접촉력 읽기
  - ANYmal-C 4족 보행 로봇의 발 접촉력 모니터링

실행:
    source env_isaaclab/bin/activate
    python 11_contact_sensor.py --num_envs 2

    # GUI 없이 실행
    python 11_contact_sensor.py --headless

    # 환경 수 변경
    python 11_contact_sensor.py --num_envs 4
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="11 - ContactSensor 접촉력 측정 튜토리얼")
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
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# ANYmal-C: ANYbotics의 4족 보행 로봇
# activate_contact_sensors=True가 이미 설정되어 있어 접촉 감지가 가능합니다.
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의 - 접촉 센서가 포함된 씬
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ContactSceneCfg(InteractiveSceneCfg):
    """ANYmal-C 로봇과 발 접촉 센서가 포함된 씬 설정.

    ContactSensorCfg를 사용하여 로봇의 발(FOOT)에 접촉력 센서를 부착합니다.
    prim_path에 와일드카드(.*_FOOT)를 사용하면 매칭되는 모든 body에 센서가 생성됩니다.

    ANYmal-C의 발 body 이름:
      - LF_FOOT (Left Front, 왼쪽 앞)
      - RF_FOOT (Right Front, 오른쪽 앞)
      - LH_FOOT (Left Hind, 왼쪽 뒤)
      - RH_FOOT (Right Hind, 오른쪽 뒤)
    """

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # ── 돔 조명 (DomeLight) ──────────────────────────────────────────────
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # ── ANYmal-C 로봇 (Articulation) ────────────────────────────────────
    # ANYMAL_C_CFG에는 init_state로 서 있는 자세가 이미 정의되어 있습니다.
    # - pos=(0, 0, 0.6): 바닥에서 0.6m 높이에 배치
    # - joint_pos: HAA, HFE, KFE 관절의 기본 각도 설정
    robot: ArticulationCfg = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # ── 접촉 센서 (ContactSensor) ────────────────────────────────────────
    # prim_path: ".*_FOOT" 패턴으로 4개의 발(LF, RF, LH, RH)에 동시 부착
    # update_period: 0.0이면 매 물리 스텝마다 업데이트
    # history_length: 과거 6스텝의 접촉력 이력 저장 (net_forces_w_history로 접근)
    # debug_vis: True이면 접촉 지점에 시각화 마커 표시
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_FOOT",
        update_period=0.0,
        history_length=6,
        debug_vis=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """접촉 센서 데이터를 읽으며 시뮬레이션을 실행합니다."""

    # ── 씬에서 엔티티 꺼내기 ─────────────────────────────────────────────
    robot = scene["robot"]
    contact_sensor = scene["contact_forces"]

    # ── 센서 정보 출력 ────────────────────────────────────────────────────
    # body_names: 센서가 부착된 body 이름 목록
    # 와일드카드 패턴 ".*_FOOT"에 매칭된 body들이 나열됩니다.
    print(f"[INFO] 접촉 센서 부착 body: {contact_sensor.body_names}")
    print(f"[INFO] 센서 body 개수: {contact_sensor.num_bodies}")

    # ── 시간 관련 변수 ───────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    step_count = 0

    print(f"[INFO] Physics dt = {sim_dt:.6f}초  (1 step = {sim_dt * 1000:.2f}ms)")
    print(f"[INFO] 환경 수: {scene.num_envs}")
    print("=" * 70)

    # ── ANYmal-C 발 이름 매핑 (출력 시 가독성을 위해) ─────────────────────
    foot_labels = {
        "LF_FOOT": "왼앞(LF)",
        "RF_FOOT": "오른앞(RF)",
        "LH_FOOT": "왼뒤(LH)",
        "RH_FOOT": "오른뒤(RH)",
    }

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션 정지/일시정지 처리
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 500 step마다 에피소드 리셋 ────────────────────────────────
        if step_count % 500 == 0:
            step_count = 0

            # root state 리셋 - 환경 원점 오프셋 필수
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])

            # joint state 리셋 (기본 서 있는 자세)
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)

            # 씬 내부 버퍼 초기화 (센서 이력 포함)
            scene.reset()
            print("[INFO] 에피소드 리셋 완료 - 로봇을 기본 자세로 초기화")

        # ── (2) 기본 자세 유지를 위한 위치 제어 ───────────────────────────
        # default_joint_pos를 목표로 설정하면 로봇이 서 있는 자세를 유지합니다.
        # 접촉력을 안정적으로 관찰하기 위해 정지 상태를 유지합니다.
        targets = robot.data.default_joint_pos
        robot.set_joint_position_target(targets)

        # ── (3) 씬 데이터를 시뮬레이션에 기록 ─────────────────────────────
        scene.write_data_to_sim()

        # ── (4) 물리 스텝 실행 ────────────────────────────────────────────
        sim.step()

        # ── (5) 스텝 카운터 증가 ──────────────────────────────────────────
        step_count += 1

        # ── (6) 씬 버퍼 업데이트 (센서 데이터 갱신 포함) ──────────────────
        scene.update(sim_dt)

        # ── (7) 접촉력 데이터 읽기 및 출력 (100 step마다) ─────────────────
        if step_count % 100 == 0:
            # ─────────────────────────────────────────────────────────────
            # net_forces_w: 월드 프레임 기준 순 접촉력 (법선 방향 합력)
            # shape: (num_envs, num_bodies, 3)
            #   - num_envs: 환경 개수
            #   - num_bodies: 센서가 부착된 body 수 (4개 발)
            #   - 3: (Fx, Fy, Fz) 월드 좌표계 기준 힘 벡터
            # ─────────────────────────────────────────────────────────────
            forces = contact_sensor.data.net_forces_w  # (num_envs, 4, 3)

            # 각 발의 접촉력 크기 계산: ||F|| = sqrt(Fx^2 + Fy^2 + Fz^2)
            force_magnitudes = torch.norm(forces, dim=-1)  # (num_envs, 4)

            # 전체 환경에서 최대 접촉력
            max_force = force_magnitudes.max().item()

            print(f"\n[Step {step_count}] 접촉력 측정 결과")
            print(f"  전체 최대 접촉력: {max_force:.2f} N")

            # env 0에 대해 발별 상세 출력
            for i, body_name in enumerate(contact_sensor.body_names):
                label = foot_labels.get(body_name, body_name)
                fx, fy, fz = forces[0, i].tolist()
                mag = force_magnitudes[0, i].item()
                print(f"  {label:>10s}: Fx={fx:7.2f}, Fy={fy:7.2f}, Fz={fz:7.2f}  |F|={mag:7.2f} N")

            # ─────────────────────────────────────────────────────────────
            # net_forces_w_history: 과거 접촉력 이력
            # shape: (num_envs, history_length, num_bodies, 3)
            #   - history_length=6: 최근 6스텝의 이력
            #   - index 0이 가장 최근, index 5가 가장 오래된 데이터
            # ─────────────────────────────────────────────────────────────
            history = contact_sensor.data.net_forces_w_history  # (num_envs, 6, 4, 3)
            # 가장 최근 이력의 Fz 값 (env 0 기준)
            recent_fz = history[0, 0, :, 2]  # 최근 1스텝의 Fz (4개 발)
            oldest_fz = history[0, -1, :, 2]  # 가장 오래된 Fz (4개 발)
            print(f"  이력 Fz (최근): {recent_fz.tolist()}")
            print(f"  이력 Fz (6스텝 전): {oldest_fz.tolist()}")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """ContactSensor가 포함된 InteractiveScene을 생성하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    # dt=0.005 (200Hz): 접촉 감지에 적합한 물리 시뮬레이션 주기
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점: ANYmal-C를 위에서 비스듬히 내려다보는 각도
    sim.set_camera_view(eye=[3.5, 3.5, 3.5], target=[0.0, 0.0, 0.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    scene_cfg = ContactSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
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
