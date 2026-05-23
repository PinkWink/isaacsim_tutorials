# 03_galileo_experiment.py
# Galileo's Leaning Tower of Pisa Experiment in IsaacLab
#
# 갈릴레오가 16세기 피사의 사탑에서 했다고 전해지는 실험을 IsaacLab/PhysX 로 재현합니다.
# "무게가 다른 두 물체를 같은 높이에서 떨어뜨리면 동시에 바닥에 도착한다"
#
# 1 kg 가벼운 공(빨강)과 10 kg 무거운 공(파랑)을 피사의 사탑 실제 높이(약 57 m)에서
# 동시에 떨어뜨리고, z 좌표와 착지 시점을 출력해 동시 도달을 수치로 확인합니다.
#
# 물리적 근거:
#   F = m·g  →  a = F/m = g  (질량 소거)
#   진공(공기 저항 없음)에서 모든 물체는 동일한 g 로 가속합니다.
#   PhysX 는 기본적으로 공기 저항을 모델링하지 않으므로 진공과 동일.
#
# 실행 방법:
#   source env_isaaclab/bin/activate
#   python 03_galileo_experiment.py
#   python 03_galileo_experiment.py --headless
#
# 사용법:
#   실행하면 두 공이 사탑 꼭대기 옆에 멈춰 있습니다.
#   'Simulation Control' 윈도우의 ▶ Play 버튼을 눌러 동시에 떨어뜨립니다.
#   ⏹ Reset 버튼을 누르면 두 공이 원래 높이로 되돌아갑니다.

# ──────────────────────────────────────────────
# 1) argparse + AppLauncher (IsaacLab 표준 패턴)
# ──────────────────────────────────────────────
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="IsaacLab Tutorial: Galileo's Leaning Tower of Pisa Experiment"
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""AppLauncher 초기화 이후에 나머지 모듈을 import 합니다."""

# ──────────────────────────────────────────────
# 2) IsaacLab / USD / UI import
# ──────────────────────────────────────────────
import math

import omni.kit.app
import omni.ui as ui

# Isaac Sim 표준 Play/Pause/Stop 미니바(viewport 하단)를 활성화합니다.
omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
    "omni.kit.timeline.minibar", True
)

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationCfg, SimulationContext


# ══════════════════════════════════════════════════════════════════════════
# 실험 파라미터
# ══════════════════════════════════════════════════════════════════════════

# 피사의 사탑 실제 정보 (단순화)
PISA_HEIGHT = 56.0          # 사탑 본체 높이 (m) — 실제 약 56 m
PISA_TILT_DEG = 4.0         # 사탑 기울기 (도) — 실제 약 3.97°
PISA_BASE_RADIUS = 7.5      # 기단 반지름 (m)

# 공이 떨어지는 높이 — 사탑 종루 위 살짝 위 (대략 57 m)
DROP_HEIGHT = PISA_HEIGHT + 1.0
SPHERE_RADIUS = 0.5         # 공 반지름 (m) — 멀리서도 보이게 크게

# 두 공의 위치 — 사탑이 +x 방향으로 기울어져 있으므로, 그 기울어진 위쪽 끝 바깥에서
# 떨어뜨립니다 (갈릴레오가 사탑의 leaning side 에서 떨어뜨렸다는 설). 사탑 꼭대기는
# 약 x ≈ +3.9, 종루 반지름 ≈ 5 → 꼭대기 외곽은 x ≈ +8.9. 공은 그 바깥(x ≥ 9.5).
# 두 공은 같은 y=0 평면에 두고 x 로만 약간 떨어뜨려서 옆에서 봤을 때 평행 낙하가 보이게.
BALLS = [
    # (이름, prim 경로, x 좌표, 질량 kg, 색상 RGB)
    ("Light Ball (1 kg)",  "/World/Ball_light", 10.5,  1.0, (1.0, 0.2, 0.2)),
    ("Heavy Ball (10 kg)", "/World/Ball_heavy", 12.5, 10.0, (0.2, 0.5, 1.0)),
]


# ══════════════════════════════════════════════════════════════════════════
# 씬 구성
# ══════════════════════════════════════════════════════════════════════════

def build_pisa_tower():
    """단순화된 피사의 사탑 — 8 층 원기둥 + 층간 띠를 4° 기울여서 쌓습니다.

    각 층(원기둥)의 중심은 기울어진 축 위에 위치하고, 원기둥 자체도 같은 각도로 회전.
    """
    tilt_rad = math.radians(PISA_TILT_DEG)
    cos_t = math.cos(tilt_rad)
    sin_t = math.sin(tilt_rad)

    # y축 기준으로 PISA_TILT_DEG° 회전하는 쿼터니언 (w, x, y, z)
    tilt_quat = (math.cos(tilt_rad / 2), 0.0, math.sin(tilt_rad / 2), 0.0)

    cream = (0.92, 0.87, 0.75)         # 본체 색 (대리석 베이지)
    ring_color = (0.85, 0.78, 0.65)    # 층간 띠 색 (살짝 어두운 베이지)
    bell_color = (0.95, 0.92, 0.85)    # 종루 색

    # (axis_z 시작점, 높이, 반지름, 색) — 본체 + 띠 + 종루
    levels = [
        ( 0.0,  8.0, PISA_BASE_RADIUS,        cream),       # 기단 (조금 큼)
        ( 8.0,  0.6, PISA_BASE_RADIUS + 0.2,  ring_color),  # 띠
        ( 8.6,  6.5, PISA_BASE_RADIUS - 0.3,  cream),       # 2층
        (15.1,  0.6, PISA_BASE_RADIUS - 0.1,  ring_color),
        (15.7,  6.5, PISA_BASE_RADIUS - 0.5,  cream),       # 3층
        (22.2,  0.6, PISA_BASE_RADIUS - 0.3,  ring_color),
        (22.8,  6.5, PISA_BASE_RADIUS - 0.7,  cream),       # 4층
        (29.3,  0.6, PISA_BASE_RADIUS - 0.5,  ring_color),
        (29.9,  6.5, PISA_BASE_RADIUS - 0.9,  cream),       # 5층
        (36.4,  0.6, PISA_BASE_RADIUS - 0.7,  ring_color),
        (37.0,  6.5, PISA_BASE_RADIUS - 1.1,  cream),       # 6층
        (43.5,  0.6, PISA_BASE_RADIUS - 0.9,  ring_color),
        (44.1,  6.5, PISA_BASE_RADIUS - 1.3,  cream),       # 7층
        (50.6,  0.6, PISA_BASE_RADIUS - 1.1,  ring_color),
        (51.2,  4.8, PISA_BASE_RADIUS - 2.5,  bell_color),  # 종루 (꼭대기, 좁음)
    ]

    for i, (z_local, h, r, color) in enumerate(levels):
        axis_z = z_local + h / 2  # 층의 중심을 사탑 축 좌표로
        x = axis_z * sin_t
        z = axis_z * cos_t

        cyl_cfg = sim_utils.CylinderCfg(
            radius=r,
            height=h,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        cyl_cfg.func(
            f"/World/PisaTower/Level{i:02d}",
            cyl_cfg,
            translation=(x, 0.0, z),
            orientation=tilt_quat,
        )


def design_scene():
    """씬 구성 — 바닥, 조명, 사탑, 두 공."""

    # 바닥 (피사의 사탑 광장이 넓도록 크게)
    cfg_ground = sim_utils.GroundPlaneCfg(size=(200.0, 200.0))
    cfg_ground.func("/World/GroundPlane", cfg_ground)

    # 조명 (해 같은 평행광, 따뜻한 색감)
    cfg_light = sim_utils.DistantLightCfg(
        intensity=3500.0,
        color=(1.0, 0.97, 0.9),
    )
    cfg_light.func("/World/DistantLight", cfg_light, translation=(0.0, 0.0, 60.0))

    # 피사의 사탑
    build_pisa_tower()

    # 두 개의 공 — 같은 z(높이), 같은 y, 다른 x(좌우 1 m 간격), 다른 질량
    for name, prim_path, x_pos, mass, color in BALLS:
        sphere_cfg = sim_utils.SphereCfg(
            radius=SPHERE_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        )
        sphere_cfg.func(
            prim_path,
            sphere_cfg,
            translation=(x_pos, 0.0, DROP_HEIGHT),
        )


# ══════════════════════════════════════════════════════════════════════════
# 메인 함수
# ══════════════════════════════════════════════════════════════════════════

def main():
    # SimulationContext
    sim_cfg = SimulationCfg(dt=1.0 / 60.0)
    sim = SimulationContext(sim_cfg)

    # 카메라 — y축 옆에서 보는 시점 (사탑의 4° 기울기 + 두 공의 평행 낙하가
    # 한 화면에 모두 잘 보입니다). 사탑은 왼쪽, 떨어지는 공들은 오른쪽에 위치.
    sim.set_camera_view(
        eye=(8.0, -55.0, 30.0),
        target=(8.0, 0.0, 25.0),
    )

    design_scene()
    sim.reset()

    # 타임라인 — 자동 정지/루프 방지
    sim._timeline.set_looping(False)
    sim._timeline.set_start_time(0.0)
    sim._timeline.set_end_time(1e9)

    # ──────────────────────────────────────────
    # 사용자 컨트롤 윈도우
    # ──────────────────────────────────────────
    dropping = [False]
    reset_requested = [False]

    def on_play():
        dropping[0] = True
        print("[INFO] ▶ Play — 갈릴레오 실험 시작! 두 공이 동시에 떨어집니다")

    def on_reset():
        reset_requested[0] = True

    control_window = ui.Window("Simulation Control", width=260, height=110)
    with control_window.frame:
        with ui.HStack(spacing=10):
            ui.Button("▶ Play", clicked_fn=on_play, height=60)
            ui.Button("⏹ Reset", clicked_fn=on_reset, height=60)

    # 두 공의 위치를 한 view 로 관리
    balls_view = sim_utils.XformPrimView("/World/Ball_.*", device=sim.device)

    # 이론적 낙하 시간 (자유낙하)
    g = 9.81
    fall_distance = DROP_HEIGHT - SPHERE_RADIUS  # 공 중심이 닿는 z = 반지름
    t_theory = math.sqrt(2 * fall_distance / g)

    print("=" * 78)
    print("[INFO] Galileo's Leaning Tower of Pisa Experiment")
    print(f"[INFO] 사탑 높이 = {PISA_HEIGHT:.1f} m,  기울기 = {PISA_TILT_DEG:.1f}°")
    print(f"[INFO] 낙하 시작 높이 = {DROP_HEIGHT:.1f} m")
    for name, _, x, mass, _ in BALLS:
        print(f"        {name}  x={x:+.1f}  mass={mass:5.1f} kg")
    print(f"[INFO] 이론상 낙하 시간 = √(2·{fall_distance:.1f}/g) = {t_theory:.3f} s")
    print(f"[INFO] (질량과 무관 — 두 공이 동시에 도달해야 함)")
    print("[INFO] >>> 'Simulation Control' 윈도우의 ▶ Play 버튼을 눌러 시작 <<<")
    print("=" * 78)

    # ──────────────────────────────────────────
    # 시뮬레이션 루프
    # ──────────────────────────────────────────
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step_count = 0
    LANDING_Z = SPHERE_RADIUS + 0.02  # 공 중심이 이 이하로 내려오면 착지
    landed = {name: None for name, *_ in BALLS}

    while simulation_app.is_running():
        # Reset 처리 (프레임 경계에서 안전하게)
        if reset_requested[0]:
            dropping[0] = False
            sim.reset()
            sim._timeline.set_looping(False)
            sim._timeline.set_end_time(1e9)
            sim_time = 0.0
            step_count = 0
            landed = {name: None for name, *_ in BALLS}
            print(f"[INFO] ⏹ Reset — 두 공이 z = {DROP_HEIGHT:.1f} m 로 복귀")
            reset_requested[0] = False

        if dropping[0]:
            sim.step()
            sim_time += sim_dt
            step_count += 1

            positions, _ = balls_view.get_world_poses()
            z_values = positions[:, 2].cpu().numpy()

            # 매 20 step (≈ 0.33 s) 마다 z 좌표 출력
            if step_count % 20 == 0:
                z_str = "   ".join(
                    f"{name}: z={z:6.2f} m"
                    for (name, *_), z in zip(BALLS, z_values)
                )
                print(f"  [t={sim_time:5.3f} s]  {z_str}")

            # 착지 시점 기록
            for (name, *_), z in zip(BALLS, z_values):
                if landed[name] is None and z <= LANDING_Z:
                    landed[name] = sim_time
                    print(f"  >>> {name} 착지: t = {sim_time:.3f} s "
                          f"(이론값 {t_theory:.3f} s)")
                    if all(t is not None for t in landed.values()):
                        diff_ms = (max(landed.values()) - min(landed.values())) * 1000
                        print()
                        print(f"  ★ 두 공의 착지 시간 차이: {diff_ms:.1f} ms")
                        print(f"  ★ 이론상 0 ms — 무게가 달라도 동시에 도달 (갈릴레오)")
                        print()
        else:
            sim.render()

    print("[INFO] 시뮬레이션 종료.")


if __name__ == "__main__":
    main()
