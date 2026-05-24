"""
18_lidar_maze.py - JetBot LiDAR(RayCaster) + 미로 + matplotlib 실시간 스캔 시각화

JetBot 한 대를 4 × 4 m 미로 안에 두고, chassis 위에 부착한 2D LiDAR
(RayCaster + LidarPatternCfg) 의 360° 스캔을 matplotlib 으로 실시간
표시한다. JetBot 은 17번과 같은 P-제어 waypoint 추종으로 미로 안을 한 바퀴
돈다.

구성 요점:
  - 미로 벽 전체를 **하나의 UsdGeom.Mesh** 로 spawn (10번 ray_caster 의
    단일-메시 제약과 동일). 외벽 4개 + 내부 가로 막대 1개 + 우측 상단 기둥 1개.
  - LiDAR: channels=1, horizontal_fov_range=(-180°, +180°), horizontal_res=1°
    → 360 rays. JetBot chassis 위 0.06 m 지점에 offset 으로 부착.
  - matplotlib 2-패널:
      * 좌: top-down 지도 — 벽 윤곽, 로봇 위치/방향(화살표), ray endpoints
      * 우: polar plot — sensor frame 기준 각도 vs 거리

실행:
    source env_isaaclab/bin/activate
    cd lectures/18_lidar_maze

    # 권장: GUI + matplotlib
    python 18_lidar_maze.py

    # GUI 없이 (matplotlib 만)
    python 18_lidar_maze.py --headless
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="18 - JetBot LiDAR 미로 스캔")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import math
import os
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import euler_xyz_from_quat

from pxr import UsdGeom, Sdf, Gf
import omni.usd

# ── 3. Matplotlib ────────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use("TkAgg")
    INTERACTIVE_PLOT = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE_PLOT = False
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ══════════════════════════════════════════════════════════════════════════
# 미로 정의 — 모든 벽을 단일 mesh 로 합쳐서 spawn
# ══════════════════════════════════════════════════════════════════════════
# 각 벽: (center_x, center_y, size_x, size_y)  — z 는 공통 (WALL_Z_LO ~ WALL_Z_HI)
WALL_Z_LO = 0.0
WALL_Z_HI = 0.30
MAZE_HALF = 2.0       # 미로 외벽 절반 길이 (전체 4 m × 4 m)
WALL_THICK = 0.08

WALLS: list[tuple[float, float, float, float]] = [
    # ── 외벽 4개 (정사각형) ──────────────────────────────────────────────
    ( 0.0, -MAZE_HALF, 2 * MAZE_HALF + WALL_THICK, WALL_THICK),  # 남쪽
    ( 0.0, +MAZE_HALF, 2 * MAZE_HALF + WALL_THICK, WALL_THICK),  # 북쪽
    (-MAZE_HALF,  0.0, WALL_THICK, 2 * MAZE_HALF + WALL_THICK),  # 서쪽
    (+MAZE_HALF,  0.0, WALL_THICK, 2 * MAZE_HALF + WALL_THICK),  # 동쪽
    # ── 내부 가로 막대 (y = 0 부근, x ∈ [-0.8, 0.8]) ────────────────────
    ( 0.0, 0.0, 1.6, WALL_THICK),
    # ── 우측 상단 기둥 ──────────────────────────────────────────────────
    ( 1.2, 1.2, 0.30, 0.30),
]

MAZE_MESH_PARENT = "/World/Maze"
MAZE_MESH_PATH   = "/World/Maze/MazeMesh"


def _box_geom(cx: float, cy: float, sx: float, sy: float,
              z_lo: float, z_hi: float
              ) -> tuple[list[Gf.Vec3f], list[tuple[int, int, int]]]:
    """직육면체 한 개의 8 정점 + 12 삼각형 (CCW from outside) 을 반환.

    좌표계: z-up. 박스는 [cx±sx/2, cy±sy/2, z_lo..z_hi].
    """
    x0, x1 = cx - sx / 2, cx + sx / 2
    y0, y1 = cy - sy / 2, cy + sy / 2
    z0, z1 = z_lo, z_hi

    # 정점 8개 (bot 4 + top 4)
    pts = [
        Gf.Vec3f(x0, y0, z0),  # 0 bot-SW
        Gf.Vec3f(x1, y0, z0),  # 1 bot-SE
        Gf.Vec3f(x1, y1, z0),  # 2 bot-NE
        Gf.Vec3f(x0, y1, z0),  # 3 bot-NW
        Gf.Vec3f(x0, y0, z1),  # 4 top-SW
        Gf.Vec3f(x1, y0, z1),  # 5 top-SE
        Gf.Vec3f(x1, y1, z1),  # 6 top-NE
        Gf.Vec3f(x0, y1, z1),  # 7 top-NW
    ]
    # 12 삼각형 (외향 normal). 각 면 = 사각형 2장.
    tris = [
        # 하면 (z = z0, normal -z)
        (0, 2, 1), (0, 3, 2),
        # 상면 (z = z1, normal +z)
        (4, 5, 6), (4, 6, 7),
        # 남면 (y = y0, normal -y)
        (0, 1, 5), (0, 5, 4),
        # 북면 (y = y1, normal +y)
        (2, 3, 7), (2, 7, 6),
        # 서면 (x = x0, normal -x)
        (3, 0, 4), (3, 4, 7),
        # 동면 (x = x1, normal +x)
        (1, 2, 6), (1, 6, 5),
    ]
    return pts, tris


def spawn_maze_mesh() -> None:
    """모든 벽을 합쳐 단일 UsdGeom.Mesh 로 만든다.

    10번 ray_caster 와 동일한 이유:
        IsaacLab RayCaster 는 mesh_prim_paths 의 mesh 의 face indices 를
        3개씩 묶어 삼각형으로 해석한다. 따라서 미로의 모든 벽을 **삼각형
        분할된 하나의 mesh** 로 만들어야 ray-cast 가 일관되게 동작한다.
    """
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, Sdf.Path(MAZE_MESH_PARENT))
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(MAZE_MESH_PATH))

    points: list[Gf.Vec3f] = []
    fv_counts: list[int] = []
    fv_indices: list[int] = []
    base = 0
    for (cx, cy, sx, sy) in WALLS:
        pts, tris = _box_geom(cx, cy, sx, sy, WALL_Z_LO, WALL_Z_HI)
        points.extend(pts)
        for (a, b, c) in tris:
            fv_counts.append(3)
            fv_indices.extend([base + a, base + b, base + c])
        base += len(pts)

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(fv_counts)
    mesh.CreateFaceVertexIndicesAttr(fv_indices)
    # 회색 벽
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.58, 0.62)])


# ══════════════════════════════════════════════════════════════════════════
# JetBot 설정 (15/16/17/19 번 공통)
# ══════════════════════════════════════════════════════════════════════════
JETBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/NVIDIA/Jetbot/jetbot.usd",
    ),
    init_state=ArticulationCfg.InitialStateCfg(pos=(-1.3, -1.3, 0.0)),
    actuators={
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["left_wheel_joint", "right_wheel_joint"],
            stiffness=0.0,
            damping=10.0,
        ),
    },
)

WHEEL_RADIUS = 0.03
WHEEL_BASE   = 0.1125

# 미로 안 waypoint (외벽/내벽 안 부딪히도록)
WAYPOINTS: list[tuple[float, float]] = [
    (-1.3, -1.3),   # 시작 = 좌하단
    ( 1.3, -1.3),   # 우하단 (가로 벽 아래)
    ( 1.3,  0.6),   # 우중단 (기둥 옆 통과)
    (-1.3,  1.3),   # 좌상단
    (-1.3, -1.3),   # 복귀
]


# ══════════════════════════════════════════════════════════════════════════
# Scene
# ══════════════════════════════════════════════════════════════════════════
@configclass
class LidarMazeSceneCfg(InteractiveSceneCfg):
    """JetBot + LiDAR (RayCaster). 미로 mesh 는 main() 에서 직접 spawn."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.9, 0.9, 0.95)),
    )
    robot: ArticulationCfg = JETBOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # LiDAR — JetBot chassis 에 부착. offset_pos 로 차체 위로 살짝 띄움.
    # channels=1: 수평 1열만 (2D LiDAR). 360 / 1° = 360 rays.
    lidar = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/chassis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.06)),
        attach_yaw_only=True,        # roll/pitch 무시 → 수평면 유지
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=1.0,
        ),
        mesh_prim_paths=[MAZE_MESH_PARENT],
        debug_vis=True,
        update_period=0.0,
    )


# ══════════════════════════════════════════════════════════════════════════
# 차동 구동 유틸
# ══════════════════════════════════════════════════════════════════════════
def unicycle_to_wheel_vel(v, omega):
    v_left  = (v - omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    v_right = (v + omega * WHEEL_BASE / 2.0) / WHEEL_RADIUS
    return v_left, v_right


# ══════════════════════════════════════════════════════════════════════════
# matplotlib 설정 — top-down 지도 + polar plot
# ══════════════════════════════════════════════════════════════════════════
def _setup_plot():
    """2-패널 figure 와 그릴 객체들을 만들어 반환."""
    if INTERACTIVE_PLOT:
        plt.ion()
    fig = plt.figure(figsize=(11, 5.2))
    fig.suptitle("JetBot LiDAR in Maze — top-down map (left) + polar scan (right)",
                 fontsize=12)

    # ── 좌: top-down ─────────────────────────────────────────────────────
    ax_map = fig.add_subplot(1, 2, 1)
    ax_map.set_aspect("equal")
    pad = 0.3
    ax_map.set_xlim(-MAZE_HALF - pad, MAZE_HALF + pad)
    ax_map.set_ylim(-MAZE_HALF - pad, MAZE_HALF + pad)
    ax_map.set_xlabel("x  [m]")
    ax_map.set_ylabel("y  [m]")
    ax_map.set_title("Map view")
    ax_map.grid(alpha=0.3)

    # 벽 그리기 (정적)
    for (cx, cy, sx, sy) in WALLS:
        ax_map.add_patch(Rectangle(
            (cx - sx / 2, cy - sy / 2), sx, sy,
            facecolor="#7c8087", edgecolor="black", lw=0.5,
        ))

    # waypoints
    wp_xy = list(zip(*WAYPOINTS))
    ax_map.plot(wp_xy[0], wp_xy[1], "g--", lw=0.8, alpha=0.5, label="planned path")
    ax_map.scatter(wp_xy[0], wp_xy[1], c="green", s=40, alpha=0.6,
                   marker="o", zorder=3, label="waypoints")

    # 동적 객체
    hits_scatter = ax_map.scatter([], [], c="tab:red", s=4, alpha=0.7,
                                  label="LiDAR hits", zorder=4)
    (robot_dot,) = ax_map.plot([], [], "o", color="tab:blue", ms=10,
                               zorder=5, label="JetBot")
    heading_quiver = ax_map.quiver(
        [0], [0], [0], [0],
        angles="xy", scale_units="xy", scale=1.0,
        color="tab:blue", width=0.008, zorder=6,
    )
    ax_map.legend(loc="upper right", fontsize=8)

    # ── 우: polar (sensor frame 기준) ────────────────────────────────────
    ax_polar = fig.add_subplot(1, 2, 2, projection="polar")
    ax_polar.set_title("Polar scan (sensor frame)", pad=14)
    ax_polar.set_theta_zero_location("N")   # 0° = 로봇 전방 = 그래프 위쪽
    # 반시계방향 (+) — lidar_pattern 의 azimuth 가 표준 CCW (+90° = sensor +y = LEFT)
    # 이므로 polar 도 CCW 로 두어야 로봇의 왼쪽이 그래프 왼쪽에 표시됨 (ROS rviz convention).
    ax_polar.set_theta_direction(1)
    ax_polar.set_rlim(0, max(MAZE_HALF * 2.0, 5.0))
    ax_polar.grid(alpha=0.4)
    (polar_line,) = ax_polar.plot([], [], color="tab:red", lw=1.2)
    polar_fill = [None]  # mutable container 로 fill 객체 보관

    plt.tight_layout()
    return {
        "fig": fig,
        "ax_map": ax_map,
        "ax_polar": ax_polar,
        "hits_scatter": hits_scatter,
        "robot_dot": robot_dot,
        "heading_quiver": heading_quiver,
        "polar_line": polar_line,
        "polar_fill": polar_fill,
    }


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션 루프
# ══════════════════════════════════════════════════════════════════════════
def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["robot"]
    lidar: RayCaster = scene["lidar"]
    sim_dt = sim.get_physics_dt()

    # 바퀴 관절 ID
    left_ids,  _ = robot.find_joints("left_wheel_joint")
    right_ids, _ = robot.find_joints("right_wheel_joint")
    wheel_ids = left_ids + right_ids

    waypoints = torch.tensor(WAYPOINTS, device=sim.device)
    num_wp = waypoints.shape[0]

    # P-제어기
    Kp_lin, Kp_ang = 1.2, 4.0
    v_max, w_max   = 0.25, 2.5
    arrive_thr     = 0.10

    current_wp = torch.zeros(1, dtype=torch.long, device=sim.device)
    step_count = 0
    laps       = 0

    print("=" * 70)
    print("[INFO] LiDAR 미로 데모")
    print(f"  미로: {2*MAZE_HALF:.1f} × {2*MAZE_HALF:.1f} m, 벽 {len(WALLS)} 개")
    print(f"  LiDAR rays: {lidar.num_rays} (360° / 1°)")
    print(f"  waypoints : {WAYPOINTS}")
    print("=" * 70)

    # plot 준비 (1 step 시뮬 후 LiDAR 데이터 한 번 확보한 뒤 좌표 알게 되므로,
    # plot 객체는 미리 만든다.)
    P = _setup_plot()

    # LiDAR 의 ray 각도 — sensor frame 기준 (수평면 360°, 360 rays).
    # lidar_pattern 코드에 따르면 -180°~+180°, 360 rays (마지막 점 제외).
    ray_thetas = torch.linspace(-math.pi, math.pi, lidar.num_rays + 1)[:-1]
    ray_thetas_np = ray_thetas.cpu().numpy()

    DISPLAY_INTERVAL = max(1, int(round(0.05 / sim_dt)))   # ~20 Hz 갱신

    while simulation_app.is_running():
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 초기 리셋 ────────────────────────────────────────────────
        if step_count == 0:
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim(root_state[:, :7])
            robot.write_root_velocity_to_sim(root_state[:, 7:])
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            scene.reset()
            current_wp[:] = 0
            print("[INFO] 리셋 완료 — 주행 시작")

        # ── (2) 로봇 상태 ───────────────────────────────────────────────
        pos_w = robot.data.root_pos_w[:, :2]
        origin_xy = scene.env_origins[:, :2]
        pos_local = pos_w - origin_xy
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w)

        # ── (3) P-제어 (17번과 동일 패턴) ───────────────────────────────
        target = waypoints[current_wp]
        dx = target[:, 0] - pos_local[:, 0]
        dy = target[:, 1] - pos_local[:, 1]
        distance = torch.sqrt(dx**2 + dy**2)
        desired_yaw = torch.atan2(dy, dx)
        heading_err = torch.atan2(
            torch.sin(desired_yaw - yaw),
            torch.cos(desired_yaw - yaw),
        )
        v = torch.clamp(Kp_lin * distance, max=v_max)
        omega = torch.clamp(Kp_ang * heading_err, -w_max, w_max)
        v = v * torch.clamp(1.0 - torch.abs(heading_err) / math.pi, min=0.1)

        wl, wr = unicycle_to_wheel_vel(v, omega)
        wheel_vel = torch.zeros(robot.num_instances, 2, device=robot.device)
        wheel_vel[:, 0] = wl
        wheel_vel[:, 1] = wr
        robot.set_joint_velocity_target(wheel_vel, joint_ids=wheel_ids)

        # ── (4) 스텝 ─────────────────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step_count += 1

        # ── (5) waypoint 도달 처리 ──────────────────────────────────────
        if distance[0] < arrive_thr:
            old = current_wp[0].item()
            current_wp[0] = (current_wp[0] + 1) % num_wp
            new = current_wp[0].item()
            if new == 0:
                laps += 1
                print(f"  WP {old} 도달 → 1 lap 완료 (lap {laps})")
            else:
                print(f"  WP {old} 도달 → 다음 WP {new} ({WAYPOINTS[new]})")

        # ── (6) LiDAR 데이터 → plot 갱신 ────────────────────────────────
        if step_count % DISPLAY_INTERVAL != 0:
            continue

        hits_w   = lidar.data.ray_hits_w[0]            # (num_rays, 3) world
        sensor_pos_w = lidar.data.pos_w[0]             # (3,) world
        sensor_quat_w = lidar.data.quat_w[0]           # (4,) wxyz

        # 거리 = ||hit - sensor||. 무한대(=miss) 처리.
        delta = hits_w - sensor_pos_w
        dist = torch.norm(delta, dim=1)
        dist[~torch.isfinite(dist)] = float("inf")
        dist_np = dist.cpu().numpy()

        # (a) top-down: hit 점들을 그대로 (world xy). miss 는 제외.
        finite_mask = torch.isfinite(dist)
        hits_xy = hits_w[finite_mask, :2].cpu().numpy()
        P["hits_scatter"].set_offsets(hits_xy if hits_xy.size else [[0, 0]])

        # 로봇 위치/방향
        rx, ry = pos_local[0, 0].item(), pos_local[0, 1].item()
        ryaw = yaw[0].item()
        P["robot_dot"].set_data([rx], [ry])
        head_len = 0.25
        P["heading_quiver"].set_offsets([[rx, ry]])
        P["heading_quiver"].set_UVC(
            [head_len * math.cos(ryaw)],
            [head_len * math.sin(ryaw)],
        )

        # (b) polar plot — sensor frame 기준 (각도는 ray pattern 그대로).
        #   distance 가 무한대인 곳은 sensor 최대 사거리(예: 8 m) 로 클립해 표시.
        R_MAX = 8.0
        polar_r = dist.clone()
        polar_r[~torch.isfinite(polar_r)] = R_MAX
        polar_r_np = polar_r.cpu().numpy()
        # 폐곡선으로 보이게 끝점 한 번 더 추가
        theta_closed = torch.cat([ray_thetas, ray_thetas[:1]]).cpu().numpy()
        r_closed = torch.cat([polar_r, polar_r[:1]]).cpu().numpy()
        P["polar_line"].set_data(theta_closed, r_closed)
        # fill 갱신 (이전 fill 제거)
        if P["polar_fill"][0] is not None:
            P["polar_fill"][0].remove()
        P["polar_fill"][0] = P["ax_polar"].fill(
            theta_closed, r_closed, color="tab:red", alpha=0.12,
        )[0]

        if INTERACTIVE_PLOT:
            P["fig"].canvas.draw_idle()
            P["fig"].canvas.flush_events()

        # ── (7) 콘솔 로그 (1 초 주기) ───────────────────────────────────
        if step_count % int(round(1.0 / sim_dt)) == 0:
            n_finite = int(finite_mask.sum().item())
            d_min = float(dist[finite_mask].min().item()) if n_finite else float("nan")
            d_mean = float(dist[finite_mask].mean().item()) if n_finite else float("nan")
            print(f"[t={step_count*sim_dt:5.1f}s] "
                  f"pos=({rx:+.2f},{ry:+.2f}) yaw={math.degrees(ryaw):+5.0f}° | "
                  f"WP {current_wp[0].item()} dist={distance[0].item():.2f} | "
                  f"LiDAR hits={n_finite}/{lidar.num_rays} "
                  f"d_min={d_min:.2f} d_mean={d_mean:.2f}")

        # 종료 조건: 2 lap 이후
        if laps >= 2:
            print("[INFO] 2 lap 완료 — 종료")
            break

    # ── 최종 저장 ────────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "18_lidar_maze_result.png")
    P["fig"].savefig(save_path, dpi=140, bbox_inches="tight")
    print(f"[INFO] result saved: {save_path}")
    if INTERACTIVE_PLOT:
        plt.ioff()
    plt.close(P["fig"])


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.0, -4.5, 5.0], target=[0.0, 0.0, 0.0])

    # 미로 mesh 는 InteractiveScene 생성 전에 stage 에 spawn 해야
    # RayCaster 가 초기화 시점에 mesh_prim_paths 를 찾을 수 있다.
    spawn_maze_mesh()
    print(f"[INFO] 미로 mesh 생성 완료: {MAZE_MESH_PATH} ({len(WALLS)} walls)")

    scene_cfg = LidarMazeSceneCfg(num_envs=1, env_spacing=8.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] InteractiveScene 생성 완료")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("[DONE] 18 (lidar-maze) 종료")
