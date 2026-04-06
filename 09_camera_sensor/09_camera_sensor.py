"""
09_camera_sensor.py - 카메라 센서로 RGB/Depth 이미지 실시간 표시

IsaacLab의 Camera 센서로 RGB와 Depth 이미지를 획득하고,
matplotlib 창에서 실시간으로 표시합니다.

필수 플래그: --enable_cameras (카메라 렌더링 활성화)
필수 패키지: python3-tk (또는 python3.11-tk)

실행:
    source env_isaaclab/bin/activate
    cd lectures/09_camera_sensor

    # 권장: headless + matplotlib 조합 (IsaacSim GUI 부하 제거)
    python 09_camera_sensor.py --headless --enable_cameras

    # IsaacSim GUI도 함께 보기 (GPU 부하 높음, not responding 발생 가능)
    python 09_camera_sensor.py --enable_cameras
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="09 - Camera Sensor (real-time display)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import os
import torch
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.sensors.camera import Camera, CameraCfg

# ── 3. Matplotlib ────────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use("TkAgg")
    INTERACTIVE = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE = False
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성
# ══════════════════════════════════════════════════════════════════════════


def design_scene() -> Camera:
    """바닥, 조명, 색상별 오브젝트, 카메라 센서를 구성합니다."""

    # ── 바닥 + 조명 ──────────────────────────────────────────────────────
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(1.0, 1.0, 1.0))
    cfg.func("/World/DistantLight", cfg, translation=(1.0, 1.0, 10.0))

    # ── 색상별 오브젝트 ──────────────────────────────────────────────────
    # 빨간 큐브
    cfg = sim_utils.CuboidCfg(
        size=(0.3, 0.3, 0.3),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
    )
    cfg.func("/World/Objects/RedCube", cfg, translation=(1.0, 0.0, 0.15))

    # 초록 원뿔
    cfg = sim_utils.ConeCfg(
        radius=0.15, height=0.5,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.1)),
    )
    cfg.func("/World/Objects/GreenCone", cfg, translation=(1.0, -0.6, 0.25))

    # 파란 원기둥
    cfg = sim_utils.CylinderCfg(
        radius=0.12, height=0.4,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.2, 0.9)),
    )
    cfg.func("/World/Objects/BlueCylinder", cfg, translation=(1.2, 0.5, 0.2))

    # 노란 구
    cfg = sim_utils.SphereCfg(
        radius=0.2,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.9, 0.1)),
    )
    cfg.func("/World/Objects/YellowSphere", cfg, translation=(1.5, 0.0, 0.2))

    # ── 카메라 센서 ──────────────────────────────────────────────────────
    camera_cfg = CameraCfg(
        prim_path="/World/CameraSensor",
        update_period=0,       # 매 물리 스텝마다 업데이트
        height=240,            # 실시간 표시를 위해 해상도를 낮춤
        width=320,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
    )
    camera = Camera(cfg=camera_cfg)

    return camera


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════


def main() -> None:
    """카메라 센서 영상을 matplotlib에서 실시간 표시합니다."""

    # ── SimulationContext ─────────────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[5.0, 3.0, 3.0], target=[1.0, 0.0, 0.0])

    camera = design_scene()

    sim.reset()
    camera.reset()

    # ── 카메라 위치 설정 ─────────────────────────────────────────────────
    eye = torch.tensor([[3.0, 0.0, 1.5]], device=sim.device)
    target = torch.tensor([[1.0, 0.0, 0.2]], device=sim.device)
    camera.set_world_poses_from_view(eye, target)

    sim_dt = sim.get_physics_dt()

    print("[INFO] sim.reset() complete")
    print(f"[INFO] Camera resolution: {camera.image_shape[0]} x {camera.image_shape[1]} (H x W)")
    print(f"[INFO] Data types: {camera.cfg.data_types}")
    print(f"[INFO] Physics dt: {sim_dt:.6f}s")

    # ── 첫 프레임 렌더링 (초기 이미지 획득) ──────────────────────────────
    for _ in range(5):
        sim.step()
        camera.update(dt=sim_dt)

    # ── matplotlib 실시간 창 설정 ────────────────────────────────────────
    DISPLAY_INTERVAL = 10  # N스텝마다 화면 갱신 (낮출수록 부하 감소)
    TOTAL_STEPS = 300      # 5초 (60Hz)

    if INTERACTIVE:
        plt.ion()

    fig, (ax_rgb, ax_depth) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Camera Sensor — RGB & Depth (real-time)", fontsize=13)

    # 초기 이미지로 imshow 생성
    rgb_data = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()  # (H, W, 3)
    depth_data = camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()  # (H, W)
    depth_display = np.clip(depth_data, 0, 5.0)  # 0~5m 범위로 클리핑

    im_rgb = ax_rgb.imshow(rgb_data)
    ax_rgb.set_title("RGB Image")
    ax_rgb.axis("off")

    im_depth = ax_depth.imshow(depth_display, cmap="turbo", vmin=0, vmax=5.0)
    ax_depth.set_title("Depth Image (0~5m)")
    ax_depth.axis("off")
    fig.colorbar(im_depth, ax=ax_depth, label="Distance (m)", shrink=0.8)

    plt.tight_layout()

    if INTERACTIVE:
        fig.canvas.draw()
        fig.canvas.flush_events()

    print(f"\n[INFO] Running {TOTAL_STEPS} steps with real-time display...")
    print("=" * 60)

    # ── 시뮬레이션 루프 ──────────────────────────────────────────────────
    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        # (1) 물리 스텝
        sim.step()

        # (2) 카메라 업데이트
        camera.update(dt=sim_dt)

        # (3) 실시간 표시
        if INTERACTIVE and step % DISPLAY_INTERVAL == 0:
            # RGB: (1, H, W, 4) → (H, W, 3)
            rgb_np = camera.data.output["rgb"][0, :, :, :3].cpu().numpy()
            im_rgb.set_data(rgb_np)

            # Depth: (1, H, W, 1) → (H, W), clip to 0~5m
            depth_np = camera.data.output["distance_to_image_plane"][0, :, :, 0].cpu().numpy()
            depth_np = np.clip(depth_np, 0, 5.0)
            im_depth.set_data(depth_np)

            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # (4) 콘솔 출력 (100 step마다)
        if (step + 1) % 100 == 0:
            rgb_img = camera.data.output["rgb"]
            depth_img = camera.data.output["distance_to_image_plane"]
            valid_depth = depth_img[depth_img < 1e4]
            depth_str = f"{valid_depth.min().item():.2f}~{valid_depth.max().item():.2f}m" if valid_depth.numel() > 0 else "N/A"
            print(
                f"  [Step {step + 1:4d}] "
                f"RGB shape={list(rgb_img.shape)} | "
                f"Depth range={depth_str}"
            )

    print(f"\n[INFO] Simulation complete — {TOTAL_STEPS} steps")

    # ── 최종 이미지 저장 ─────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "09_camera_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] Final frame saved: {save_path}")

    if INTERACTIVE:
        plt.ioff()
        print("[INFO] Close the plot window to exit.")
        plt.show()
    else:
        plt.close(fig)

    simulation_app.close()
    print("[DONE] Simulation ended")


if __name__ == "__main__":
    main()
