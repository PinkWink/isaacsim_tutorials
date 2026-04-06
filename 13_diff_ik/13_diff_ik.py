"""
13_diff_ik.py - Differential IK 컨트롤러를 이용한 엔드이펙터 제어

Differential IK(미분 역운동학) 컨트롤러를 사용하여 Franka Panda 로봇의
엔드이펙터(panda_hand)를 목표 자세(pose)로 이동시키는 방법을 배웁니다:
  - DifferentialIKControllerCfg로 IK 컨트롤러 설정
  - DLS(Damped Least Squares) 방법으로 특이점(singularity) 안정적 처리
  - PhysX에서 Jacobian 행렬 가져오기: robot.root_physx_view.get_jacobians()
  - subtract_frame_transforms로 루트 프레임 기준 EE 자세 계산
  - FRAME_MARKER_CFG로 현재 EE 위치와 목표 위치 시각화

역운동학(Inverse Kinematics)이란?
  - 순운동학(FK): 관절 각도 → 엔드이펙터 위치/자세
  - 역운동학(IK): 엔드이펙터 목표 위치/자세 → 관절 각도
  - Differential IK: Jacobian 행렬을 이용해 미소 변화량(Δq)을 계산하는 반복적 방법

실행:
    source env_isaaclab/bin/activate
    python 13_diff_ik.py --num_envs 4

    # GUI 없이 실행
    python 13_diff_ik.py --headless

    # 환경 수 변경
    python 13_diff_ik.py --num_envs 8
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="13 - Differential IK 컨트롤러 튜토리얼")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# FRANKA_PANDA_HIGH_PD_CFG: 높은 PD 게인의 Franka Panda 로봇
# - 높은 stiffness/damping으로 IK 목표를 정밀하게 추종
# - disable_gravity=True로 중력 보상 없이도 안정적
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의 - 테이블 위 Franka Panda
# ══════════════════════════════════════════════════════════════════════════


@configclass
class DiffIKSceneCfg(InteractiveSceneCfg):
    """Differential IK 데모를 위한 씬 설정.

    Franka Panda 로봇이 테이블/스탠드 위에 고정된 형태입니다.
    바닥면은 z=-1.05로 내려서 스탠드 아래에 위치합니다.
    """

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # z=-1.05: 스탠드(높이 ~1.05m) 아래에 바닥이 위치하도록 내림
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

    # ── 스탠드/테이블 마운트 ─────────────────────────────────────────────
    # Franka Panda를 올려놓는 받침대 (stand_instanceable.usd)
    # scale=(2,2,2)로 확대하여 로봇이 안정적으로 배치되도록 함
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd",
            scale=(2.0, 2.0, 2.0),
        ),
    )

    # ── Franka Panda 로봇 (Articulation) ────────────────────────────────
    # FRANKA_PANDA_HIGH_PD_CFG 특징:
    # - 높은 PD 게인 (stiffness=400, damping=80) → IK 목표를 정밀하게 추종
    # - disable_gravity=True → 중력 보상 없이도 팔이 처지지 않음
    # - panda_shoulder (joint 1~4), panda_forearm (joint 5~7) 액추에이터
    robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Differential IK 컨트롤러로 엔드이펙터를 목표 자세로 이동시킵니다."""

    # ── 씬에서 로봇 꺼내기 ───────────────────────────────────────────────
    robot = scene["robot"]

    # ── Differential IK 컨트롤러 생성 ────────────────────────────────────
    # command_type="pose": 위치(xyz) + 자세(quaternion) 목표를 받음 (7차원)
    # use_relative_mode=False: 절대 목표 자세 사용 (True이면 현재 대비 상대 변위)
    # ik_method="dls": Damped Least Squares 방법
    #   - Jacobian의 의사역행렬에 감쇠항(λ)을 추가하여 특이점 근처 안정성 보장
    #   - Δq = J^T (J J^T + λ²I)^{-1} Δx
    #   - "pinv"(의사역행렬)보다 안정적, "svd"(특이값분해)보다 빠름
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )
    print(f"[INFO] DifferentialIKController 생성 완료")
    print(f"  - command_type: pose (7D = xyz + quaternion)")
    print(f"  - ik_method: dls (Damped Least Squares)")
    print(f"  - action_dim: {diff_ik_controller.action_dim}")

    # ── FRAME_MARKER_CFG로 시각화 마커 생성 ──────────────────────────────
    # EE 현재 위치 마커: 로봇 엔드이펙터의 실시간 위치를 표시
    # 목표 위치 마커: IK가 추종하는 목표 자세를 표시
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # ── EE 목표 자세 정의 ────────────────────────────────────────────────
    # 형식: [x, y, z, qw, qx, qy, qz]
    # - xyz: 루트(base) 프레임 기준 엔드이펙터 위치 (미터)
    # - qw,qx,qy,qz: 루트 프레임 기준 엔드이펙터 자세 (쿼터니언)
    #
    # 목표 1: 오른쪽 위 - 비스듬히 기울어진 자세
    # 목표 2: 왼쪽 앞 - 아래를 향하는 자세
    # 목표 3: 정면 중앙 - 수직 아래를 향하는 자세
    ee_goals = [
        [0.5, 0.5, 0.7, 0.707, 0.0, 0.707, 0.0],
        [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0],
    ]
    ee_goals = torch.tensor(ee_goals, device=sim.device)
    print(f"[INFO] EE 목표 자세 {len(ee_goals)}개 정의 완료")
    for i, goal in enumerate(ee_goals):
        print(f"  목표 {i}: pos=({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")

    # ── 목표 추적 변수 ──────────────────────────────────────────────────
    current_goal_idx = 0

    # ── IK 명령 버퍼 (모든 환경에 동일한 목표 설정) ──────────────────────
    # shape: (num_envs, 7) - 각 환경의 EE 목표 자세
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=sim.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    # ── 로봇별 파라미터 설정 (SceneEntityCfg) ────────────────────────────
    # SceneEntityCfg: 씬 내 엔티티의 관절/바디 이름을 지정하여
    #                 인덱스를 자동으로 해석(resolve)하는 유틸리티
    # joint_names: IK로 제어할 관절 이름 (정규식 지원)
    #   - "panda_joint.*": panda_joint1 ~ panda_joint7 (7개 관절)
    #   - 그리퍼 관절(panda_finger_joint)은 제외
    # body_names: 엔드이펙터로 사용할 body 이름
    #   - "panda_hand": Franka Panda의 그리퍼 마운트 프레임
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    # resolve()로 이름 → 인덱스 변환
    # 이후 robot_entity_cfg.joint_ids, robot_entity_cfg.body_ids 사용 가능
    robot_entity_cfg.resolve(scene)
    print(f"[INFO] 제어 관절 인덱스: {robot_entity_cfg.joint_ids}")
    print(f"[INFO] EE body 인덱스: {robot_entity_cfg.body_ids}")

    # ── Jacobian 인덱스 계산 ─────────────────────────────────────────────
    # PhysX의 get_jacobians()가 반환하는 텐서에서 EE에 해당하는 행 인덱스
    # 고정 베이스 로봇(fixed base): body_id - 1 (루트 body가 Jacobian에 미포함)
    # 이동 베이스 로봇(floating base): body_id 그대로 사용
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]
    print(f"[INFO] Jacobian EE 인덱스: {ee_jacobi_idx} (fixed_base={robot.is_fixed_base})")

    # ── 시간 관련 변수 ───────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    count = 0

    print(f"[INFO] Physics dt = {sim_dt:.4f}초")
    print(f"[INFO] 환경 수: {scene.num_envs}")
    print(f"[INFO] 150 step마다 목표 자세 변경")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # ── (1) 150 step마다 리셋 및 목표 변경 ───────────────────────────
        # Differential IK는 반복적(iterative) 방법이므로,
        # 초기 관절 상태에서 시작하여 매 스텝 Jacobian을 계산하고
        # 점진적으로 목표에 접근합니다.
        if count % 150 == 0:
            # 스텝 카운터 리셋
            count = 0

            # 관절 상태를 기본값으로 리셋
            # (홈 자세에서 시작해야 IK가 안정적으로 수렴)
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()

            # IK 목표 설정 - 다음 목표로 순환
            ik_commands[:] = ee_goals[current_goal_idx]

            # 현재 관절 위치를 IK 대상 관절만 추출하여 초기값으로 저장
            # (리셋 직후이므로 default_joint_pos의 관련 관절 부분)
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()

            # IK 컨트롤러 리셋 및 새 목표 설정
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)

            print(f"[목표 변경] 목표 {current_goal_idx}: "
                  f"pos=({ee_goals[current_goal_idx][0]:.2f}, "
                  f"{ee_goals[current_goal_idx][1]:.2f}, "
                  f"{ee_goals[current_goal_idx][2]:.2f})")

            # 다음 리셋 시 사용할 목표 인덱스 갱신 (순환)
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)

        else:
            # ── (2) Differential IK 계산 ─────────────────────────────────
            # IK 계산에 필요한 4가지 정보:
            #   1) Jacobian 행렬: EE 속도와 관절 속도의 관계 (ẋ = J * q̇)
            #   2) EE 현재 자세: 루트 프레임 기준 (position + quaternion)
            #   3) 관절 현재 위치: 제어 대상 관절의 현재 각도
            #   4) 목표 자세: 이미 set_command()로 설정됨

            # ── Jacobian 행렬 가져오기 ───────────────────────────────────
            # robot.root_physx_view.get_jacobians():
            #   shape = (num_envs, num_bodies-1, 6, num_dofs)
            #   - 6: 선속도(3) + 각속도(3)
            # 우리가 필요한 것: EE body의 Jacobian, 제어 대상 관절만 추출
            #   → [:, ee_jacobi_idx, :, joint_ids]
            #   결과 shape: (num_envs, 6, num_controlled_joints)
            jacobian = robot.root_physx_view.get_jacobians()[
                :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
            ]

            # ── EE 현재 자세 (월드 프레임) ───────────────────────────────
            # body_pose_w: 월드 프레임 기준 body 자세 (position + quaternion)
            ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]

            # ── 루트(base) 자세 (월드 프레임) ────────────────────────────
            root_pose_w = robot.data.root_pose_w

            # ── 제어 대상 관절의 현재 위치 ───────────────────────────────
            joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

            # ── EE 자세를 루트 프레임 기준으로 변환 ──────────────────────
            # subtract_frame_transforms(parent_pos, parent_quat, child_pos, child_quat)
            # → child의 자세를 parent 프레임 기준으로 변환
            # IK 컨트롤러는 루트 프레임 기준 자세를 기대하므로 변환 필요
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
            )

            # ── IK 계산: 목표 관절 위치 산출 ─────────────────────────────
            # diff_ik_controller.compute():
            #   입력: EE 현재 자세(body frame), Jacobian, 현재 관절 위치
            #   내부: Δx = x_goal - x_current  (자세 오차)
            #          Δq = J^T (J J^T + λ²I)^{-1} Δx  (DLS 방법)
            #          q_des = q_current + Δq  (관절 위치 목표)
            #   출력: 관절 위치 목표 (num_envs, num_controlled_joints)
            joint_pos_des = diff_ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, joint_pos
            )

        # ── (3) 관절 위치 목표 적용 ──────────────────────────────────────
        # joint_ids를 지정하여 IK 제어 대상 관절에만 목표 설정
        # (그리퍼 관절 등 다른 관절에는 영향 없음)
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)

        # ── (4) 씬 데이터를 시뮬레이션에 기록 ────────────────────────────
        scene.write_data_to_sim()

        # ── (5) 물리 스텝 실행 ───────────────────────────────────────────
        sim.step()

        # ── (6) 스텝 카운터 증가 ─────────────────────────────────────────
        count += 1

        # ── (7) 씬 버퍼 업데이트 ─────────────────────────────────────────
        scene.update(sim_dt)

        # ── (8) 시각화 마커 업데이트 ─────────────────────────────────────
        # EE 현재 위치 마커: 월드 프레임 기준
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])

        # 목표 위치 마커: IK 목표는 루트 프레임 기준이므로 env_origins를 더해
        # 월드 프레임으로 변환하여 표시
        goal_marker.visualize(
            ik_commands[:, 0:3] + scene.env_origins,
            ik_commands[:, 3:7],
        )

        # ── (9) 주기적 상태 출력 (50 step마다) ───────────────────────────
        if count % 50 == 0 and count > 0:
            # EE 현재 위치와 목표 위치 사이의 거리(오차) 계산
            ee_pos_error = torch.norm(
                ee_pose_w[:, 0:3] - (ik_commands[:, 0:3] + scene.env_origins),
                dim=-1,
            )
            mean_error = ee_pos_error.mean().item()
            print(f"  [Step {count:3d}] 평균 위치 오차: {mean_error:.4f} m")


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Differential IK 씬을 생성하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    # dt=0.01 (100Hz): IK 제어에 적합한 물리 시뮬레이션 주기
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    # 카메라 시점: 로봇을 비스듬히 내려다보는 각도
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    scene_cfg = DiffIKSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
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
