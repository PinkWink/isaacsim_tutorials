"""
14_pick_and_place.py - Pick and Place: IK + 그리퍼 제어 통합

Franka Panda 로봇으로 물체를 집어서 다른 위치에 놓는 Pick & Place 작업을 수행합니다:
  - DifferentialIKController로 로봇팔 EE(end-effector)를 목표 위치로 이동
  - 그리퍼(finger joint)의 position target으로 물체를 잡고 놓기
  - 상태 머신(State Machine)으로 APPROACH → LOWER → GRASP → LIFT → MOVE → RELEASE → RETREAT 순서 제어
  - RigidObject로 작은 큐브를 스폰하여 실제 집기 대상으로 활용

실행:
    source env_isaaclab/bin/activate
    python 14_pick_and_place.py --num_envs 2

    # GUI 없이 실행
    python 14_pick_and_place.py --headless

    # 환경 수 변경
    python 14_pick_and_place.py --num_envs 4
"""

# ── 1. AppLauncher 패턴 (import 전에 Omniverse 앱을 먼저 실행) ────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="14 - Pick and Place 튜토리얼")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Omniverse / IsaacLab import (AppLauncher 이후에만 가능) ─────────────
"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
# ISAAC_NUCLEUS_DIR는 더 이상 사용하지 않음 (테이블을 CuboidCfg로 직접 생성)
from isaaclab.utils.math import subtract_frame_transforms

# ── 3. 사전 정의된 로봇 설정 불러오기 ─────────────────────────────────────
# FRANKA_PANDA_HIGH_PD_CFG: 높은 PD 게인 + 중력 비활성화 → IK 제어에 적합
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
# 상태 머신 Phase 정의
# ══════════════════════════════════════════════════════════════════════════

# 각 Phase의 이름과 인덱스를 정의합니다.
# 상태 머신이 순서대로 진행하며, 각 Phase에서 일정 시간이 지나면 다음 Phase로 전환합니다.
PHASE_APPROACH = 0   # 물체 위로 EE 이동
PHASE_LOWER = 1      # 물체 높이까지 EE 하강
PHASE_GRASP = 2      # 그리퍼 닫기 (물체 잡기)
PHASE_LIFT = 3       # 물체를 들어올리기
PHASE_MOVE = 4       # 놓을 위치로 이동
PHASE_RELEASE = 5    # 그리퍼 열기 (물체 놓기)
PHASE_RETREAT = 6    # 위로 후퇴

PHASE_NAMES = [
    "APPROACH", "LOWER", "GRASP", "LIFT", "MOVE", "RELEASE", "RETREAT"
]


# ══════════════════════════════════════════════════════════════════════════
# InteractiveSceneCfg 정의
# ══════════════════════════════════════════════════════════════════════════


@configclass
class PickPlaceSceneCfg(InteractiveSceneCfg):
    """Pick & Place 작업을 위한 씬 설정.

    구성 요소:
      - Franka Panda (HIGH_PD, 중력 비활성화) → IK 기반 팔 제어에 적합
      - 스탠드/테이블 마운트 → 로봇이 올라갈 받침대
      - 빨간색 작은 큐브 (4cm) → 집기 대상 물체
      - 바닥면 (z=-1.05로 낮춤) → 테이블 높이에 맞추기 위해
      - 돔 조명
    """

    # ── 바닥면 (GroundPlane) ──────────────────────────────────────────────
    # z=-1.05로 낮추어 스탠드가 지면과 맞닿도록 배치
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

    # ── テーブル (단순 박스로 생성) ──────────────────────────────────────
    # Nucleus USD 대신 CuboidCfg로 테이블을 직접 생성합니다.
    # 높이 0.4m, 폭 0.8m의 회색 박스 — 로봇 앞에 배치
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.8, 0.8, 0.4),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.6, 0.5, 0.4),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.2)),
    )

    # ── Franka Panda 로봇 (HIGH_PD, 중력 비활성화) ──────────────────────
    # HIGH_PD: shoulder/forearm stiffness=400, damping=80
    # disable_gravity=True → IK로 정밀하게 제어할 때 안정적
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ── 집기 대상 큐브 (RigidObject) ─────────────────────────────────────
    # 4cm 크기의 빨간색 큐브, 질량 0.1kg
    # 테이블 위에 스폰 (x=0.5, z=0.02 → 바닥에서 큐브 중심까지 높이)
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.04, 0.04, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.42)),  # 테이블 상면(z=0.4) + 큐브 반높이(0.02)
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """상태 머신 기반 Pick & Place를 실행합니다."""

    # ── 씬에서 엔티티 꺼내기 ─────────────────────────────────────────────
    robot = scene["robot"]
    cube = scene["cube"]

    # ── DifferentialIK 컨트롤러 생성 ─────────────────────────────────────
    # command_type="pose": 절대 위치/자세를 목표로 설정
    # ik_method="dls": Damped Least Squares (특이점 근처에서 안정적)
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    diff_ik_controller = DifferentialIKController(
        diff_ik_cfg, num_envs=scene.num_envs, device=sim.device
    )

    # ── 로봇 엔티티 설정 (팔 관절 + EE body) ────────────────────────────
    # Franka Panda: panda_joint1~7이 팔 관절, panda_hand가 EE body
    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=["panda_joint.*"],
        body_names=["panda_hand"],
    )
    robot_entity_cfg.resolve(scene)

    # EE Jacobian 인덱스: 고정 베이스 로봇은 body_id - 1
    # (root body가 Jacobian에 포함되지 않으므로)
    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1

    # ── 그리퍼(finger) 관절 ID 가져오기 ──────────────────────────────────
    # panda_finger_joint1, panda_finger_joint2의 ID를 찾습니다.
    # find_joints()는 (joint_ids_list, joint_names_list) 튜플을 반환합니다.
    finger_joint_ids, finger_joint_names = robot.find_joints("panda_finger_joint.*")
    print(f"[INFO] 그리퍼 관절: {finger_joint_names} (IDs: {finger_joint_ids})")

    # ── 상태 머신 변수 초기화 ────────────────────────────────────────────
    # 각 환경별 현재 Phase (모두 APPROACH에서 시작)
    current_phase = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)
    # 각 Phase 내 스텝 카운터
    phase_step = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # ── Phase별 지속 시간 (스텝 수) ──────────────────────────────────────
    # 각 Phase가 몇 스텝 동안 지속되는지 정의
    PHASE_DURATIONS = {
        PHASE_APPROACH: 120,   # 접근: 1.2초 (dt=0.01)
        PHASE_LOWER: 80,       # 하강: 0.8초
        PHASE_GRASP: 60,       # 잡기: 0.6초 (그리퍼 닫히는 시간)
        PHASE_LIFT: 100,       # 들어올리기: 1.0초
        PHASE_MOVE: 120,       # 이동: 1.2초
        PHASE_RELEASE: 60,     # 놓기: 0.6초 (그리퍼 열리는 시간)
        PHASE_RETREAT: 80,     # 후퇴: 0.8초
    }

    # ── 목표 위치 정의 (로봇 로컬 프레임 기준) ────────────────────────────
    # EE 목표 자세: [x, y, z, qw, qx, qy, qz]
    # 쿼터니언 (0, 1, 0, 0)은 EE가 아래를 바라보는 자세 (그리퍼가 아래로 향함)
    PICK_POS = [0.5, 0.0, 0.0]       # 물체 위치 (x, y, z) — z는 Phase별로 다름
    PLACE_POS = [0.5, 0.3, 0.0]      # 놓을 위치 (y 방향으로 0.3m 이동)
    EE_QUAT = [0.0, 1.0, 0.0, 0.0]   # EE 자세: 아래를 바라봄 (w, x, y, z)

    # 높이 파라미터 (테이블 상면 z=0.4 기준)
    APPROACH_HEIGHT = 0.55   # 접근 높이 (테이블 위 15cm)
    GRASP_HEIGHT = 0.42      # 잡기 높이 (큐브 중심 = 테이블 상면 + 2cm)
    LIFT_HEIGHT = 0.60       # 들어올린 높이 (테이블 위 20cm)
    RETREAT_HEIGHT = 0.65    # 후퇴 높이

    # ── 시간 관련 변수 ───────────────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    step_count = 0

    # IK 목표 명령 버퍼 (7D: pos(3) + quat(4))
    ik_commands = torch.zeros(scene.num_envs, 7, device=sim.device)
    # IK로 계산된 목표 관절 위치
    joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()

    print(f"[INFO] Physics dt = {sim_dt:.4f}초")
    print(f"[INFO] 환경 수: {scene.num_envs}")
    print(f"[INFO] Phase 순서: {' → '.join(PHASE_NAMES)}")
    print("=" * 70)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    while simulation_app.is_running():
        # 시뮬레이션 정지/일시정지 처리
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 주기적 에피소드 리셋 (전체 Phase 완료 후) ─────────────────
        # RETREAT Phase가 완료된 환경을 리셋합니다.
        reset_mask = (current_phase >= len(PHASE_NAMES))
        if reset_mask.any() or step_count == 0:
            # 리셋 대상 환경의 인덱스
            if step_count == 0:
                reset_ids = torch.arange(scene.num_envs, device=sim.device)
            else:
                reset_ids = reset_mask.nonzero(as_tuple=False).squeeze(-1)

            # 로봇 root state 리셋 (환경 원점 오프셋 포함)
            root_state = robot.data.default_root_state[reset_ids].clone()
            root_state[:, :3] += scene.env_origins[reset_ids]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=reset_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids=reset_ids)

            # 로봇 관절 상태 리셋 (기본 자세: 그리퍼 열림 0.04)
            joint_pos_reset = robot.data.default_joint_pos[reset_ids].clone()
            joint_vel_reset = robot.data.default_joint_vel[reset_ids].clone()
            robot.write_joint_state_to_sim(joint_pos_reset, joint_vel_reset, env_ids=reset_ids)

            # 큐브 위치 리셋 (테이블 위)
            cube_state = cube.data.default_root_state[reset_ids].clone()
            cube_state[:, :3] += scene.env_origins[reset_ids]
            cube.write_root_pose_to_sim(cube_state[:, :7], env_ids=reset_ids)
            cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids=reset_ids)

            # 상태 머신 초기화
            current_phase[reset_ids] = PHASE_APPROACH
            phase_step[reset_ids] = 0

            # 컨트롤러 리셋
            diff_ik_controller.reset(env_ids=reset_ids)

            # IK 목표 관절 위치 초기화
            joint_pos_des[reset_ids] = robot.data.default_joint_pos[reset_ids][:, robot_entity_cfg.joint_ids].clone()

            # 씬 내부 버퍼 초기화
            scene.reset()

            print(f"[Step {step_count:5d}] 에피소드 리셋 (환경 {reset_ids.tolist()})")

        # ── (2) Phase별 EE 목표 위치 설정 및 그리퍼 제어 ──────────────────
        for env_idx in range(scene.num_envs):
            phase = current_phase[env_idx].item()

            if phase == PHASE_APPROACH:
                # 물체 위로 접근 (높은 위치)
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PICK_POS[0], PICK_POS[1], APPROACH_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # 그리퍼 열림 유지 (0.04)
                robot.set_joint_position_target(
                    torch.full((1, len(finger_joint_ids)), 0.04, device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_LOWER:
                # 잡기 높이까지 하강
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PICK_POS[0], PICK_POS[1], GRASP_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # 그리퍼 열림 유지
                robot.set_joint_position_target(
                    torch.full((1, len(finger_joint_ids)), 0.04, device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_GRASP:
                # 현재 위치 유지 + 그리퍼 닫기
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PICK_POS[0], PICK_POS[1], GRASP_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # ★ 그리퍼 닫기: finger joint target = 0.0 (완전히 닫힘)
                robot.set_joint_position_target(
                    torch.zeros(1, len(finger_joint_ids), device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_LIFT:
                # 물체를 잡은 채로 위로 들어올리기
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PICK_POS[0], PICK_POS[1], LIFT_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # 그리퍼 닫힘 유지
                robot.set_joint_position_target(
                    torch.zeros(1, len(finger_joint_ids), device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_MOVE:
                # 놓을 위치로 이동 (y 방향으로 이동, 높이 유지)
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PLACE_POS[0], PLACE_POS[1], LIFT_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # 그리퍼 닫힘 유지
                robot.set_joint_position_target(
                    torch.zeros(1, len(finger_joint_ids), device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_RELEASE:
                # 현재 위치 유지 + 그리퍼 열기
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PLACE_POS[0], PLACE_POS[1], LIFT_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # ★ 그리퍼 열기: finger joint target = 0.04 (완전히 열림)
                robot.set_joint_position_target(
                    torch.full((1, len(finger_joint_ids)), 0.04, device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

            elif phase == PHASE_RETREAT:
                # 위로 후퇴
                ik_commands[env_idx, 0:3] = torch.tensor(
                    [PLACE_POS[0], PLACE_POS[1], RETREAT_HEIGHT], device=sim.device
                )
                ik_commands[env_idx, 3:7] = torch.tensor(EE_QUAT, device=sim.device)
                # 그리퍼 열림 유지
                robot.set_joint_position_target(
                    torch.full((1, len(finger_joint_ids)), 0.04, device=sim.device),
                    joint_ids=finger_joint_ids,
                    env_ids=torch.tensor([env_idx], device=sim.device),
                )

        # ── (3) DifferentialIK로 관절 위치 계산 ──────────────────────────
        # IK 컨트롤러에 목표 명령 설정
        diff_ik_controller.set_command(ik_commands)

        # 로봇의 현재 상태에서 Jacobian과 EE 위치 가져오기
        jacobian = robot.root_physx_view.get_jacobians()[
            :, ee_jacobi_idx, :, robot_entity_cfg.joint_ids
        ]
        ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w
        joint_pos_current = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        # EE 위치를 로봇 로컬 프레임으로 변환
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        # IK 계산 → 목표 관절 위치
        joint_pos_des = diff_ik_controller.compute(
            ee_pos_b, ee_quat_b, jacobian, joint_pos_current
        )

        # ── (4) 팔 관절에 목표 위치 적용 ─────────────────────────────────
        robot.set_joint_position_target(
            joint_pos_des, joint_ids=robot_entity_cfg.joint_ids
        )

        # ── (5) 씬 데이터를 시뮬레이션에 기록 ────────────────────────────
        scene.write_data_to_sim()

        # ── (6) 물리 스텝 실행 ───────────────────────────────────────────
        sim.step()

        # ── (7) 스텝 카운터 증가 ─────────────────────────────────────────
        step_count += 1
        phase_step += 1

        # ── (8) 씬 버퍼 업데이트 ─────────────────────────────────────────
        scene.update(sim_dt)

        # ── (9) Phase 전환 판정 ──────────────────────────────────────────
        for env_idx in range(scene.num_envs):
            phase = current_phase[env_idx].item()
            if phase < len(PHASE_NAMES):
                duration = PHASE_DURATIONS.get(phase, 100)
                if phase_step[env_idx].item() >= duration:
                    # 다음 Phase로 전환
                    current_phase[env_idx] += 1
                    phase_step[env_idx] = 0
                    next_phase = current_phase[env_idx].item()
                    if next_phase < len(PHASE_NAMES):
                        print(
                            f"  [Env {env_idx}] Phase 전환: "
                            f"{PHASE_NAMES[phase]} → {PHASE_NAMES[next_phase]}"
                        )
                    else:
                        print(f"  [Env {env_idx}] 전체 Pick & Place 완료! 리셋 예정.")

        # ── (10) 주기적 상태 출력 (50 step마다) ──────────────────────────
        if step_count % 50 == 0:
            # env 0의 현재 EE 위치 (월드 프레임)
            ee_pos_w = robot.data.body_pose_w[0, robot_entity_cfg.body_ids[0], 0:3]
            phase_idx = current_phase[0].item()
            phase_name = PHASE_NAMES[phase_idx] if phase_idx < len(PHASE_NAMES) else "DONE"
            print(
                f"[Step {step_count:5d}] Phase={phase_name:>8s} | "
                f"EE pos=({ee_pos_w[0]:.3f}, {ee_pos_w[1]:.3f}, {ee_pos_w[2]:.3f})"
            )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """Pick & Place InteractiveScene을 생성하고 시뮬레이션을 실행합니다."""

    # ── SimulationContext 생성 ────────────────────────────────────────────
    # dt=0.01 (100Hz): IK 제어에 적합한 물리 시뮬레이션 주기
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # 카메라 시점: 테이블을 비스듬히 내려다보는 각도
    sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.5, 0.0, 0.0])

    # ── InteractiveScene 생성 ─────────────────────────────────────────────
    scene_cfg = PickPlaceSceneCfg(num_envs=args_cli.num_envs, env_spacing=3.0)
    scene = InteractiveScene(scene_cfg)
    print(f"[INFO] InteractiveScene 생성 완료 - {args_cli.num_envs}개 환경")

    # ── sim.reset() — 물리 엔진 초기화 ────────────────────────────────────
    sim.reset()
    print("[INFO] sim.reset() 완료 - 시뮬레이션 시작 준비 완료")

    # ── 시뮬레이션 실행 ──────────────────────────────────────────────────
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    # 시뮬레이션 앱 종료
    simulation_app.close()
