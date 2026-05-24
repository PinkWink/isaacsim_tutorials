"""
10_ray_caster.py - 공중 레일 스캐너로 계단 단면 스캔

공중에 떠 있는 레일 위 스캐너가 좌우로 왕복하며 아래쪽 계단을 향해
ray를 발사합니다. 측정한 바닥 높이를 스캐너 x 위치별로 누적하면
계단 단면이 그래프 위에 점점 드러납니다.

요점:
  - RayCasterCfg / GridPatternCfg 로 5×5 격자 광선 패턴 생성
  - 계단을 하나의 UsdGeom.Mesh 로 만들어 mesh_prim_paths 단일-메시 제약을 만족
  - 스캐너는 kinematic rigid body 로 만들어 매 스텝 write_root_pose_to_sim 으로 이동
  - ray_hits_w 의 z 평균을 모아 실시간 그래프에 누적

실행:
    source env_isaaclab/bin/activate
    cd lectures/10_ray_caster

    # 권장: headless + matplotlib
    python 10_ray_caster.py --headless

    # IsaacSim GUI 도 함께 보기
    python 10_ray_caster.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="10 - RayCaster 로 계단 스캔")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import ────────────────────────────────────────────────────────────
import os
import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sensors.ray_caster import RayCaster, RayCasterCfg, patterns

from pxr import UsdGeom, Sdf, Gf
import omni.usd

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
# 계단/스캔 사양
# ══════════════════════════════════════════════════════════════════════════
NUM_STEPS    = 6
STEP_DEPTH   = 0.4    # x 방향 한 단 깊이
STEP_RISE    = 0.15   # 한 단 높이
STEP_WIDTH_Y = 2.0    # y 방향 폭 (가로)
PRE_FLAT     = 1.0    # 계단 시작 전 평지 길이 (x<0 쪽)
POST_FLAT    = 1.0    # 계단 끝 후 평지 길이

RAIL_Z       = 2.5    # 스캐너가 매달린 레일 높이
SWEEP_CENTER = (NUM_STEPS * STEP_DEPTH) / 2.0          # 1.2
SWEEP_AMPL   = 2.0                                      # ±2 m 왕복
SWEEP_PERIOD = 20.0                                     # 주기 (sec) — 천천히 한 바퀴

STAIRS_MESH_PATH = "/World/Stairs/StairsMesh"
STAIRS_PARENT    = "/World/Stairs"


# ══════════════════════════════════════════════════════════════════════════
# 계단 단면(profile) → 하나의 USD Mesh
# ══════════════════════════════════════════════════════════════════════════
def _build_profile() -> list[tuple[float, float]]:
    """xz 평면에서 계단의 옆모습 윤곽 점들을 순서대로 반환.

    구성: [PRE_FLAT 평지] → [6단 계단] → [POST_FLAT 평지]
    각 단은 (수평 tread, 수직 riser) 한 쌍으로 표현된다.
    """
    pts: list[tuple[float, float]] = []
    pts.append((-PRE_FLAT, 0.0))   # 시작 평지 좌측 끝
    pts.append((0.0, 0.0))         # 계단 시작점
    for i in range(NUM_STEPS):
        x_left  = i * STEP_DEPTH
        x_right = (i + 1) * STEP_DEPTH
        z_top   = (i + 1) * STEP_RISE
        # riser: (x_left, prev_z) → (x_left, z_top)
        pts.append((x_left, z_top))
        # tread: (x_left, z_top) → (x_right, z_top)
        pts.append((x_right, z_top))
    # 끝 평지
    pts.append((NUM_STEPS * STEP_DEPTH + POST_FLAT, NUM_STEPS * STEP_RISE))
    return pts


def spawn_stairs_mesh() -> None:
    """단일 UsdGeom.Mesh 로 계단 형상을 생성한다.

    각 profile 세그먼트를 y축으로 STEP_WIDTH_Y 만큼 압출해 quad 를 만들고,
    그 quad 를 **삼각형 2개로 분할**해 fv_indices 에 넣는다.

    IsaacLab RayCaster 는 USD 메시를 Warp 로 변환할 때 face_vertex_counts 는
    무시하고 GetFaceVertexIndicesAttr 를 곧바로 3개씩 묶어 삼각형으로 해석한다
    (ray_caster.py:196 참고). 따라서 메시는 **반드시 삼각형 분할**되어 있어야
    하며, 그렇지 않으면 인덱스가 엉뚱하게 묶여 깨진 메시가 만들어진다.
    """
    stage = omni.usd.get_context().get_stage()

    # 부모 Xform 생성 (RayCaster 가 자식에서 Mesh 를 찾아냄)
    UsdGeom.Xform.Define(stage, Sdf.Path(STAIRS_PARENT))
    mesh = UsdGeom.Mesh.Define(stage, Sdf.Path(STAIRS_MESH_PATH))

    profile = _build_profile()
    y0, y1 = -STEP_WIDTH_Y / 2.0, STEP_WIDTH_Y / 2.0

    points: list[Gf.Vec3f] = []
    fv_counts: list[int] = []
    fv_indices: list[int] = []
    idx = 0
    for (x0, z0), (x1, z1) in zip(profile[:-1], profile[1:]):
        if abs(z1 - z0) < 1e-6:
            # 수평면 (tread / ground / top): 위에서 보이도록 normal = +z
            p = [
                Gf.Vec3f(x0, y0, z0),     # 0
                Gf.Vec3f(x1, y0, z0),     # 1
                Gf.Vec3f(x1, y1, z0),     # 2
                Gf.Vec3f(x0, y1, z0),     # 3
            ]
            tri_order = [(0, 1, 2), (0, 2, 3)]   # CCW from +z
        else:
            # 수직 riser: 정면(-x) 에서 보이도록 normal = -x
            z_lo, z_hi = min(z0, z1), max(z0, z1)
            p = [
                Gf.Vec3f(x0, y0, z_lo),   # 0
                Gf.Vec3f(x0, y1, z_lo),   # 1
                Gf.Vec3f(x0, y1, z_hi),   # 2
                Gf.Vec3f(x0, y0, z_hi),   # 3
            ]
            tri_order = [(0, 3, 2), (0, 2, 1)]   # CCW from -x
        points.extend(p)
        for tri in tri_order:
            fv_counts.append(3)
            fv_indices.extend(idx + k for k in tri)
        idx += 4

    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr(fv_counts)
    mesh.CreateFaceVertexIndicesAttr(fv_indices)
    # 단순 회색 표면. PreviewSurface 없이도 RayCaster 동작에는 문제 없다.
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.58, 0.62)])


# ══════════════════════════════════════════════════════════════════════════
# 씬: 바닥/조명/계단/레일/스캐너
# ══════════════════════════════════════════════════════════════════════════
def design_scene() -> RigidObject:
    """씬을 구성하고 스캐너 RigidObject 를 반환한다."""

    # 바닥 (시각용 — RayCaster 타겟은 계단 메시만)
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # 조명
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.95, 0.95, 1.0))
    cfg.func("/World/DistantLight", cfg, translation=(2.0, 2.0, 10.0))

    # 계단 (단일 Mesh)
    spawn_stairs_mesh()

    # 공중 레일 (시각용 얇은 막대 — 물리 없음)
    rail_len = 2 * SWEEP_AMPL + 1.0
    rail_cfg = sim_utils.CuboidCfg(
        size=(rail_len, 0.04, 0.04),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.15, 0.18)),
    )
    rail_cfg.func(
        "/World/Rail", rail_cfg,
        translation=(SWEEP_CENTER, 0.0, RAIL_Z + 0.15),
    )

    # 스캐너 — 중력만 끈 일반 dynamic rigid body. 매 스텝 pose+velocity 0 으로
    # 덮어써서 "테레포트" 시킨다. (kinematic_enabled 는 GPU PhysX 백엔드에서
    # 매 스텝 write 와 함께 쓰면 illegal-memory-access 가 발생하는 케이스가 있음)
    scanner_cfg = RigidObjectCfg(
        prim_path="/World/Scanner",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.45, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(SWEEP_CENTER, 0.0, RAIL_Z)),
    )
    return RigidObject(scanner_cfg)


# ══════════════════════════════════════════════════════════════════════════
# 메인
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    # ── SimulationContext ────────────────────────────────────────────────
    sim_cfg = SimulationCfg(dt=1.0 / 60.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[4.5, 5.5, 3.5], target=[1.2, 0.0, 0.5])

    scanner = design_scene()

    # ── RayCaster ────────────────────────────────────────────────────────
    ray_caster_cfg = RayCasterCfg(
        prim_path="/World/Scanner",
        mesh_prim_paths=[STAIRS_PARENT],   # 계단 단일 메시
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(0.2, 0.2),               # 20×20 cm 풋프린트 → 5×5 = 25 rays
        ),
        debug_vis=not args_cli.headless,
    )
    ray_caster = RayCaster(cfg=ray_caster_cfg)

    sim.reset()
    print(f"[INFO] sim.reset() complete | num_rays = {ray_caster.num_rays}")

    # ── matplotlib 실시간 창 ────────────────────────────────────────────
    if INTERACTIVE:
        plt.ion()
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.5))
    fig.suptitle("RayCaster — Overhead Rail Scanner Reads the Staircase Profile",
                 fontsize=13)

    # Ground-truth staircase profile
    truth = _build_profile()
    ax.plot([p[0] for p in truth], [p[1] for p in truth],
            "k--", lw=1.2, alpha=0.7, label="Ground truth (staircase)")
    measured_dot, = ax.plot([], [], "o", color="tab:orange", ms=4, alpha=0.8,
                            label="RayCaster measured z (mean of 25 rays)")
    scanner_dot, = ax.plot([], [], "v", color="tab:red", ms=12,
                           label="Scanner position")

    ax.set_xlim(-PRE_FLAT - 0.3, NUM_STEPS * STEP_DEPTH + POST_FLAT + 0.3)
    ax.set_ylim(-0.15, RAIL_Z + 0.2)
    ax.axhline(RAIL_Z, color="gray", ls=":", lw=0.8, alpha=0.5)
    ax.text(ax.get_xlim()[0] + 0.1, RAIL_Z + 0.05, "rail height", fontsize=8, color="gray")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("z (m)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()

    sim_dt = sim.get_physics_dt()
    TOTAL_STEPS = 1200          # 20초 = 정확히 한 사이클 (천천히 좌-우-좌 완주)
    DISPLAY_INTERVAL = 6

    measured_x: list[float] = []
    measured_z: list[float] = []

    # 첫 프레임 — pose + 0 속도 로 초기화한 뒤 한 번 step 해서 view 안정화
    zero_vel = torch.zeros(1, 6, device=sim.device)
    init_pose = torch.tensor([[SWEEP_CENTER, 0.0, RAIL_Z, 1.0, 0.0, 0.0, 0.0]],
                             device=sim.device)
    scanner.write_root_pose_to_sim(init_pose)
    scanner.write_root_velocity_to_sim(zero_vel)
    sim.step()
    ray_caster.update(dt=sim_dt, force_recompute=True)

    print(f"[INFO] running {TOTAL_STEPS} steps — scanner sweep ±{SWEEP_AMPL:.1f} m, "
          f"period {SWEEP_PERIOD:.0f} s")
    print("=" * 70)

    for step in range(TOTAL_STEPS):
        if not simulation_app.is_running():
            break

        # 스캐너 좌우 sin 왕복 — pose 와 함께 0 속도 도 매번 덮어쓴다
        t = step * sim_dt
        x = SWEEP_CENTER + SWEEP_AMPL * math.sin(2.0 * math.pi * t / SWEEP_PERIOD)
        pose = torch.tensor([[x, 0.0, RAIL_Z, 1.0, 0.0, 0.0, 0.0]],
                            device=sim.device)
        scanner.write_root_pose_to_sim(pose)
        scanner.write_root_velocity_to_sim(zero_vel)

        sim.step()
        ray_caster.update(dt=sim_dt, force_recompute=True)

        # 25 ray 의 z 평균을 한 점으로 기록. miss 한 ray 는 +inf 로 표시되므로
        # (raycast_mesh 기본값) finite 한 ray 만 골라 평균을 낸다.
        hits = ray_caster.data.ray_hits_w[0]
        z = hits[:, 2]
        valid = z[torch.isfinite(z)]
        if valid.numel() > 0:
            z_mean = valid.mean().item()
            measured_x.append(x)
            measured_z.append(z_mean)

        # 실시간 업데이트
        if INTERACTIVE and step % DISPLAY_INTERVAL == 0:
            measured_dot.set_data(measured_x, measured_z)
            scanner_dot.set_data([x], [RAIL_Z])
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        if (step + 1) % 200 == 0:
            print(f"  [Step {step + 1:4d}]  scanner x = {x:+.2f} m   "
                  f"측정 z = {z_mean:.3f} m")

    print(f"\n[INFO] complete — {TOTAL_STEPS} steps, {len(measured_x)} 측정점")

    # ── 결과 저장 후 자동 종료 ───────────────────────────────────────────
    # plt.show() 는 호출하지 않는다 — 블로킹 호출이라 IsaacSim Kit 과
    # GUI 리소스가 충돌해 창을 닫지 못하는 경우가 있다. PNG 로 저장하고
    # 곧바로 창을 닫는다. 결과는 저장된 PNG 로 확인할 것.
    measured_dot.set_data(measured_x, measured_z)
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "10_ray_caster_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] result saved: {save_path}")

    if INTERACTIVE:
        plt.ioff()
    plt.close(fig)

    simulation_app.close()
    print("[DONE] simulation ended")


if __name__ == "__main__":
    main()
