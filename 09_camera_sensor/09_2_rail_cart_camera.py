"""
09_2_rail_cart_camera.py - 원형 레일 위 카메라 카트: 인물 중심 궤도 촬영

반지름 4m의 원형 레일을 만들고, 그 위를 도는 카트에 카메라를 장착합니다.
레일 한가운데에는 Isaac Sim 기본 제공 사람 캐릭터(공사장 작업자)를 세워두고,
카트가 레일을 한 바퀴 도는 동안 카메라가 항상 사람을 바라보도록 합니다.
(영화 촬영에서 쓰는 '서클 돌리(circle dolly)' 샷과 같은 구성)

동작 구성:
  - 원형 레일: 침목(sleeper) + 안쪽/바깥쪽 레일을 작은 박스 조각으로 구성 (시각용)
  - 카트: 중력 비활성 강체 → 매 스텝 write_root_pose_to_sim()으로 원 궤도를 따라 이동
  - 카메라: 카트 위 마스트 높이에서 set_world_poses_from_view()로
    항상 사람(원 중심)을 바라보게 갱신
  - 사람: ISAAC_NUCLEUS_DIR의 People/Characters 에셋 (정적 배치)

필수 플래그: --enable_cameras (카메라 렌더링 활성화)
필수 패키지: python3-tk (또는 python3.11-tk)

실행:
    source env_isaaclab/bin/activate
    cd lectures/09_camera_sensor

    # 권장: headless + matplotlib 조합 (IsaacSim GUI 부하 제거)
    python 09_2_rail_cart_camera.py --headless --enable_cameras

    # IsaacSim GUI도 함께 보기 (GPU 부하 높음)
    python 09_2_rail_cart_camera.py --enable_cameras
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="09_2 - Rail Cart Camera (Circle Dolly)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import os
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# ── 3. Matplotlib ────────────────────────────────────────────────────────
import matplotlib

if os.environ.get("DISPLAY"):
    try:
        matplotlib.use("TkAgg")
        INTERACTIVE = True
    except Exception:
        matplotlib.use("Agg")
        INTERACTIVE = False
else:
    matplotlib.use("Agg")
    INTERACTIVE = False
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 공통 파라미터
# ══════════════════════════════════════════════════════════════════════════

RAIL_RADIUS = 4.0     # 레일 반지름 (m)
CART_Z = 0.20         # 카트 중심 높이 (레일 상면 0.09 + 카트 반높이 0.11)
CAM_HEIGHT = 0.65     # 카메라(마스트 꼭대기) 높이 (m)
TARGET_POS = (0.0, 0.0, 0.9)   # 카메라가 바라볼 지점 (사람의 몸통 높이)
ORBIT_PERIOD = 12.0   # 레일 한 바퀴 도는 데 걸리는 시간 (초)

# 원 중심에 세울 사람 캐릭터 (Isaac Sim 기본 제공 People 에셋)
PERSON_USD = (
    f"{ISAAC_NUCLEUS_DIR}/People/Characters/"
    "male_adult_construction_05_new/male_adult_construction_05_new.usd"
)


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성 (InteractiveSceneCfg)
# ══════════════════════════════════════════════════════════════════════════


@configclass
class RailCartSceneCfg(InteractiveSceneCfg):
    """원형 레일 + 카메라 카트 + 사람 씬 설정.

    구성 요소:
      - 바닥면 + 돔 조명
      - 사람 캐릭터 — 원 중심(원점)에 정적 배치
      - 카트 — 중력 비활성 강체 박스 (매 스텝 코드로 pose를 갱신)
      - 카메라 — 독립 프림, 매 스텝 카트 위 마스트 위치로 이동
      - 레일 자체는 시각용 프림이라 main()에서 별도로 스폰
    """

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

    # ── 사람 캐릭터 (원 중심, 정적) ──────────────────────────────────────
    person = AssetBaseCfg(
        prim_path="/World/Person",
        spawn=sim_utils.UsdFileCfg(usd_path=PERSON_USD),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )

    # ── 카메라 카트 (중력 비활성 강체) ───────────────────────────────────
    # 매 스텝 write_root_pose_to_sim()으로 pose를 직접 지정해 움직인다.
    # (GPU 파이프라인에서는 kinematic 강체에 pose를 쓰면 CUDA 오류가 나므로,
    #  중력을 끈 dynamic 강체에 pose+속도를 매 스텝 덮어쓰는 방식을 사용)
    cart: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cart",
        spawn=sim_utils.CuboidCfg(
            size=(0.5, 0.3, 0.22),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.1, 0.3, 0.8), roughness=0.4
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(RAIL_RADIUS, 0.0, CART_Z)),
    )

    # ── 레일 카메라 (2D RGB) ─────────────────────────────────────────────
    # 로봇 링크에 부착하는 대신 독립 프림으로 두고,
    # 매 스텝 set_world_poses_from_view()로 위치/시선을 직접 갱신한다.
    rail_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/RailCam",
        update_period=0,       # 매 물리 스텝마다 업데이트
        height=240,            # 실시간 표시를 위해 해상도를 낮춤
        width=320,
        data_types=["rgb"],    # 2D 카메라: RGB만 사용
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(RAIL_RADIUS, 0.0, CAM_HEIGHT),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
# 원형 레일 스폰 (시각용 프림)
# ══════════════════════════════════════════════════════════════════════════


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    """z축 회전 yaw(rad)의 쿼터니언 (w, x, y, z)."""
    return (math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0))


def spawn_circular_rail(radius: float) -> None:
    """작은 박스 조각들로 원형 레일(침목 + 안/바깥 레일)을 만듭니다.

    물리와는 무관한 시각용 프림이라 collision 없이 스폰합니다.
    (카트 pose는 코드로 직접 지정하므로 레일 위에 '보이도록' 높이만 맞추면 된다)
    """
    rail_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.35, 0.35, 0.40), metallic=0.8, roughness=0.35
    )
    sleeper_mat = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.40, 0.26, 0.13), roughness=0.9
    )

    # 침목: 반지름 방향으로 긴 나무 박스 (약 35cm 간격으로 배치)
    num_sleepers = round(2.0 * math.pi * radius / 0.35)
    sleeper_cfg = sim_utils.CuboidCfg(size=(0.5, 0.14, 0.04), visual_material=sleeper_mat)
    for i in range(num_sleepers):
        th = 2.0 * math.pi * i / num_sleepers
        sleeper_cfg.func(
            f"/World/Rail/sleeper_{i:02d}",
            sleeper_cfg,
            translation=(radius * math.cos(th), radius * math.sin(th), 0.02),
            orientation=yaw_quat(th),
        )

    # 레일: 접선 방향으로 긴 금속 박스 조각(약 17cm) × 2줄 (안쪽/바깥쪽)
    num_segments = round(2.0 * math.pi * radius / 0.17)
    for r_offset, name in ((-0.08, "inner"), (+0.08, "outer")):
        r = radius + r_offset
        seg_len = 2.0 * math.pi * r / num_segments * 1.05  # 살짝 겹치게
        rail_cfg = sim_utils.CuboidCfg(size=(0.05, seg_len, 0.05), visual_material=rail_mat)
        for i in range(num_segments):
            th = 2.0 * math.pi * i / num_segments
            rail_cfg.func(
                f"/World/Rail/{name}_{i:02d}",
                rail_cfg,
                translation=(r * math.cos(th), r * math.sin(th), 0.065),
                orientation=yaw_quat(th),
            )

    print(f"[INFO] Circular rail spawned: radius={radius}m, "
          f"{num_sleepers} sleepers + {num_segments}x2 rail segments")


def spawn_cart_mast(cart_prim_path: str) -> None:
    """카트 프림 아래에 카메라 마스트(기둥)와 하우징(박스)을 붙입니다.

    강체 프림의 자식 시각 프림은 카트와 함께 움직이므로,
    GUI에서 '카메라가 실려 있는 카트'처럼 보이게 된다.
    """
    dark_mat = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.15))

    mast_cfg = sim_utils.CylinderCfg(radius=0.02, height=0.3, visual_material=dark_mat)
    mast_cfg.func(f"{cart_prim_path}/mast", mast_cfg, translation=(0.0, 0.0, 0.26))

    housing_cfg = sim_utils.CuboidCfg(size=(0.12, 0.08, 0.08), visual_material=dark_mat)
    housing_cfg.func(f"{cart_prim_path}/cam_housing", housing_cfg, translation=(0.0, 0.0, 0.45))


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 실행 함수
# ══════════════════════════════════════════════════════════════════════════


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """카트를 원 궤도로 움직이며 카메라 영상을 실시간 표시합니다."""

    cart = scene["cart"]
    camera = scene["rail_cam"]

    sim_dt = sim.get_physics_dt()
    omega = 2.0 * math.pi / ORBIT_PERIOD  # 각속도 (rad/s)

    # 매 스텝 재사용할 텐서 버퍼
    cart_pose = torch.zeros(scene.num_envs, 7, device=sim.device)
    cart_zero_vel = torch.zeros(scene.num_envs, 6, device=sim.device)
    cam_eye = torch.zeros(scene.num_envs, 3, device=sim.device)
    cam_target = torch.tensor([TARGET_POS], device=sim.device).repeat(scene.num_envs, 1)

    print(f"[INFO] Physics dt: {sim_dt:.4f}s")
    print(f"[INFO] Camera resolution: {camera.image_shape[0]} x {camera.image_shape[1]} (H x W)")
    print(f"[INFO] Rail radius: {RAIL_RADIUS}m | Orbit period: {ORBIT_PERIOD}s")
    print(f"[INFO] Camera target (person): {TARGET_POS}")

    def apply_orbit_step(t: float) -> None:
        """시각 t의 원 궤도 위 카트 pose와 카메라 시선을 적용합니다."""
        th = omega * t

        # (1) 카트: 원 위의 위치 + 진행(접선) 방향을 향하는 yaw
        cart_pose[:, 0] = RAIL_RADIUS * math.cos(th)
        cart_pose[:, 1] = RAIL_RADIUS * math.sin(th)
        cart_pose[:, 2] = CART_Z
        qw, _, _, qz = yaw_quat(th + math.pi / 2.0)
        cart_pose[:, 3] = qw
        cart_pose[:, 6] = qz
        cart.write_root_pose_to_sim(cart_pose)
        cart.write_root_velocity_to_sim(cart_zero_vel)  # 잔여 속도 제거

        # (2) 카메라: 카트 마스트 꼭대기에서 원 중심의 사람을 바라보기
        cam_eye[:, 0] = cart_pose[:, 0]
        cam_eye[:, 1] = cart_pose[:, 1]
        cam_eye[:, 2] = CAM_HEIGHT
        camera.set_world_poses_from_view(cam_eye, cam_target)

    # ── 워밍업: 렌더러가 사람 에셋 텍스처를 로드할 시간 확보 ─────────────
    WARMUP_STEPS = 30
    print(f"\n[INFO] Warming up for {WARMUP_STEPS} steps (loading assets)...")
    for step in range(WARMUP_STEPS):
        if not simulation_app.is_running():
            return
        apply_orbit_step(t=0.0)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    # ── matplotlib 실시간 창 설정 ────────────────────────────────────────
    DISPLAY_INTERVAL = 10   # N스텝마다 화면 갱신
    TOTAL_STEPS = 1200      # 12초 (dt=0.01) = 레일 한 바퀴

    if INTERACTIVE:
        plt.ion()

    fig, ax_rgb = plt.subplots(1, 1, figsize=(6, 4.5))
    fig.suptitle("Rail Cart Camera (Circle Dolly) — Orbiting a Person", fontsize=11)

    rgb_data = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
    im_rgb = ax_rgb.imshow(rgb_data)
    ax_rgb.set_title("Rail Camera RGB")
    ax_rgb.axis("off")
    plt.tight_layout()

    if INTERACTIVE:
        fig.canvas.draw()
        fig.canvas.flush_events()

    print(f"\n[INFO] Running {TOTAL_STEPS} steps — one full orbit around the person...")
    print("=" * 60)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        # (1) 원 궤도 위 카트 pose + 카메라 시선 적용
        apply_orbit_step(t=step * sim_dt)

        # (2) 물리 스텝 + 씬(카메라 포함) 업데이트
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

        # (3) 실시간 표시
        if INTERACTIVE and step % DISPLAY_INTERVAL == 0:
            rgb_np = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
            im_rgb.set_data(rgb_np)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # (4) 콘솔 출력 (100 step마다)
        if (step + 1) % 100 == 0:
            deg = math.degrees(omega * step * sim_dt) % 360.0
            print(
                f"  [Step {step + 1:4d}] "
                f"orbit angle={deg:6.1f} deg | "
                f"cart pos=({cart_pose[0, 0]:.2f}, {cart_pose[0, 1]:.2f}) | "
                f"RGB shape={list(camera.data.output['rgb'].shape)}"
            )

    print(f"\n[INFO] Simulation complete — {TOTAL_STEPS} steps")

    # ── 최종 이미지 저장 ─────────────────────────────────────────────────
    save_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "09_2_rail_cart_result.png"
    )
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Final frame saved: {save_path}")

    if INTERACTIVE:
        plt.ioff()
        print("[INFO] Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """씬을 생성하고 레일 카트 카메라 시뮬레이션을 실행합니다."""

    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    # GUI 시점: 레일 전체와 중앙의 사람이 한눈에 보이도록
    view_dist = RAIL_RADIUS * 2.2
    sim.set_camera_view(
        eye=[view_dist, view_dist, RAIL_RADIUS * 1.5], target=[0.0, 0.0, 0.8]
    )

    scene_cfg = RailCartSceneCfg(num_envs=1, env_spacing=6.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] InteractiveScene 생성 완료 (Person + Cart + Rail Camera)")

    # 시각용 프림 추가 스폰 (레일, 카트 마스트) — sim.reset() 전에 수행
    spawn_circular_rail(RAIL_RADIUS)
    spawn_cart_mast(scene.env_prim_paths[0] + "/Cart")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)

    simulation_app.close()
    print("[DONE] Simulation ended")


if __name__ == "__main__":
    main()
