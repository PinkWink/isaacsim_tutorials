"""
11_torque_lever_contact.py - 모터 + 빔(레버) + ContactSensor 로 토크(Nm) 시각화

두 개의 동일 모터에 빔 길이만 다른 레버(L = 0.4, 0.6 m)를 달고,
각 빔의 끝이 sensor platform 을 누르도록 배치한다. 모든 모터에 동일한
토크 τ 를 인가했을 때 ContactSensor 가 측정하는 힘 F 가 빔 길이에 따라
서로 다르게 나오는 것을 보여, "Nm = N × m" 이라는 토크의 정의를 직관적으로
설명한다.

물리:
    레버 평형식 (피벗 기준 토크 합 = 0):
        τ_motor + m·g·(L/2) = F_contact · L
    →   F_contact = τ_motor / L  +  m·g / 2

    빔 질량 m 을 모두 동일하게 잡으면 두 번째 항(중력 기여)은 L 과 무관한
    상수이고, 첫 번째 항만이 빔 길이에 따라 1/L 로 작아진다.
    같은 τ 를 줄 때
        L = 0.4 m → F ≈ τ/0.4 + 2.45 N
        L = 0.6 m → F ≈ τ/0.6 + 2.45 N
    두 곡선이 1/L 비율로 갈라지는 것이 핵심 관전 포인트.

    참고: L = 0.2 m 도 처음엔 함께 시도했으나, 짧은 빔은 회전관성이 작아
    PhysX contact 응답이 불안정(chattering)했다. 이는 "짧은 레버로 큰
    힘을 내려고 하면 시스템이 거칠어진다"는 실제 기계 설계 교훈과도 통한다.

구현 요점:
  - 빔 길이만 다른 URDF 3 개를 런타임에 생성 → UrdfConverter 로 USD 변환
  - ImplicitActuatorCfg(stiffness=0, damping=0) 로 순수 토크 입력 모드 구성
  - 매 스텝 set_joint_effort_target(τ) 로 같은 토크 인가
  - ContactSensor 를 빔 링크에 부착 → platform 과의 net contact force 측정
  - matplotlib 실시간 그래프: 인가 τ 와 측정 F (세 길이) 동시 표시

실행:
    source env_isaaclab/bin/activate
    cd lectures/11_contact_sensor

    # 권장: headless + matplotlib 그래프 자동 저장
    python 11_torque_lever_contact.py --headless

    # IsaacSim GUI 도 함께 보기
    python 11_torque_lever_contact.py
"""

# ── 1. AppLauncher ────────────────────────────────────────────────────────
import argparse
import math
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="11 - 모터+레버 토크 / 접촉 힘 데모")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Import (AppLauncher 이후에만 가능) ─────────────────────────────────
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg
from isaaclab.utils import configclass

# ── 3. Matplotlib ────────────────────────────────────────────────────────
import matplotlib
try:
    matplotlib.use("TkAgg")
    INTERACTIVE_PLOT = True
except Exception:
    matplotlib.use("Agg")
    INTERACTIVE_PLOT = False
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════════
# 설정값 — 빔 사양 / 모터 토크 프로파일
# ══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_CACHE = os.path.join(SCRIPT_DIR, "lever_assets")
os.makedirs(URDF_CACHE, exist_ok=True)

PIVOT_HEIGHT = 0.5      # 모터 피벗 z (m)
BEAM_SIDE    = 0.04     # 빔 단면 한 변 (m, 정사각 단면)
BEAM_MASS    = 0.5      # 모든 빔 질량 동일 (kg)
BASE_MASS    = 5.0      # 모터 하우징 질량 (fix_base=True 라 운동학상 의미 없음)

# 두 가지 빔 길이 (이름, 길이 m, 색)
# (L=0.2 m 는 contact chattering 이 심해 제외 — 짧은 빔은 회전관성이 작아
#  같은 dt 에서 PhysX contact 응답이 불안정해진다.)
LEVERS = [
    ("medium", 0.4, "tab:green"),
    ("long",   0.6, "tab:blue"),
]
LEVER_X_OFFSET = 0.7    # 두 모터 피벗의 좌우 거리 (±LEVER_X_OFFSET)

# 토크 프로파일 (sin 으로 0 ↔ TORQUE_AMPL 진동, 한쪽 방향만 사용)
TORQUE_AMPL  = 2.0      # 최대 인가 토크 (Nm)
TORQUE_PERIOD = 6.0     # 주기 (sec)


# 모든 platform 에 공통으로 쓰는 spawn 설정 — kinematic rigid cuboid.
# (InteractiveSceneCfg 클래스 본문에 두면 "Unknown asset config type" 에러가 나서
#  모듈 레벨에 둔다. 클래스 본문의 필드는 전부 entity 후보로 해석되기 때문.)
_PLATFORM_SPAWN = sim_utils.CuboidCfg(
    size=(0.10, 0.10, 0.10),
    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    collision_props=sim_utils.CollisionPropertiesCfg(),
    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.6, 0.95)),
)


# ══════════════════════════════════════════════════════════════════════════
# URDF 생성 — 빔 길이만 다른 2-링크 articulation
# ══════════════════════════════════════════════════════════════════════════
def _beam_inertia(length: float, side: float, mass: float) -> tuple[float, float, float]:
    """직육면체 빔의 (Ixx, Iyy, Izz) — 빔 중심 기준, 길이축 = x.

    Ixx (긴 축 회전) = (1/12) m (side² + side²)
    Iyy = Izz       = (1/12) m (length² + side²)
    """
    ixx = (1.0 / 12.0) * mass * (side**2 + side**2)
    iyy = (1.0 / 12.0) * mass * (length**2 + side**2)
    izz = iyy
    return ixx, iyy, izz


def _write_lever_urdf(name: str, length: float) -> str:
    """빔 길이 length 의 레버 URDF 파일을 만들고 경로를 반환.

    구조:
      world ──(fixed, fix_base=True 가 자동 생성)── base_link (모터 하우징)
      base_link ──(motor_joint, revolute, axis=Y)── beam_link

    beam_link 는 자기 원점에서 +x 방향으로 length 만큼 뻗는 직육면체.
    원점 = 피벗(모터 축) 위치. 따라서 inertial/visual/collision 의 origin 을
    (length/2, 0, 0) 으로 두어 빔 중심에 위치시킨다.
    """
    ixx, iyy, izz = _beam_inertia(length, BEAM_SIDE, BEAM_MASS)
    urdf = f"""<?xml version="1.0"?>
<robot name="lever_{name}">

  <!-- ===== 모터 하우징 (root) ===== -->
  <link name="base_link">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.12"/></geometry>
      <material name="motor_gray"><color rgba="0.25 0.25 0.28 1"/></material>
    </visual>
    <collision>
      <origin xyz="0 0 0"/>
      <geometry><cylinder radius="0.05" length="0.12"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="{BASE_MASS}"/>
      <inertia ixx="0.01" iyy="0.01" izz="0.01" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- ===== 빔 (length = {length} m) ===== -->
  <link name="beam_link">
    <visual>
      <origin xyz="{length / 2.0} 0 0"/>
      <geometry><box size="{length} {BEAM_SIDE} {BEAM_SIDE}"/></geometry>
      <material name="beam_orange"><color rgba="1.0 0.55 0.1 1"/></material>
    </visual>
    <collision>
      <origin xyz="{length / 2.0} 0 0"/>
      <geometry><box size="{length} {BEAM_SIDE} {BEAM_SIDE}"/></geometry>
    </collision>
    <inertial>
      <origin xyz="{length / 2.0} 0 0"/>
      <mass value="{BEAM_MASS}"/>
      <inertia ixx="{ixx}" iyy="{iyy}" izz="{izz}" ixy="0" ixz="0" iyz="0"/>
    </inertial>
  </link>

  <!-- ===== 모터 관절 (revolute, axis = y) ===== -->
  <joint name="motor_joint" type="revolute">
    <parent link="base_link"/>
    <child  link="beam_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <!-- 빔이 위/아래로 ±90° 정도만 움직이게 제한 -->
    <limit lower="-1.5708" upper="1.5708" effort="50.0" velocity="20.0"/>
    <dynamics damping="0.01" friction="0.0"/>
  </joint>

</robot>
"""
    path = os.path.join(URDF_CACHE, f"lever_{name}.urdf")
    with open(path, "w") as f:
        f.write(urdf)
    return path


# ══════════════════════════════════════════════════════════════════════════
# URDF → USD 변환 + ArticulationCfg 생성
# ══════════════════════════════════════════════════════════════════════════
def _make_lever_cfg(name: str, length: float, x_pos: float) -> ArticulationCfg:
    """레버 하나에 대한 ArticulationCfg 를 만든다.

    핵심:
      - stiffness=0, damping=0 의 ImplicitActuator → 외부에서 주는
        set_joint_effort_target(τ) 만이 토크 입력. PD 제어가 끼어들지 않아
        "순수 토크 = τ" 가 그대로 들어간다.
      - effort_limit=10 Nm 로 안전 한도. TORQUE_AMPL=2 보다 충분히 큰 값.
      - activate_contact_sensors=True 로 빔 링크에 PhysX 접촉 리포터 부착.
    """
    urdf_path = _write_lever_urdf(name, length)

    # URDF → USD 변환
    converter = UrdfConverter(
        UrdfConverterCfg(
            asset_path=urdf_path,
            usd_dir=URDF_CACHE,
            usd_file_name=f"lever_{name}.usd",
            fix_base=True,                  # 모터 하우징을 월드에 고정
            merge_fixed_joints=False,       # base/beam 분리 유지
            self_collision=False,
            collision_from_visuals=False,
            collider_type="convex_hull",
            # joint_drive 는 USD 안의 기본 드라이브. 실제 제어는 아래 actuator 가 결정.
            joint_drive=UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=0.0, damping=0.0,
                ),
            ),
            force_usd_conversion=True,      # 빔 길이 바뀌면 무조건 재변환
            make_instanceable=False,
        )
    )

    return ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=converter.usd_path,
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(x_pos, 0.0, PIVOT_HEIGHT),
            joint_pos={"motor_joint": 0.0},   # 빔 수평 자세에서 시작
        ),
        actuators={
            "motor": ImplicitActuatorCfg(
                joint_names_expr=["motor_joint"],
                stiffness=0.0,                # PD 끔 → 순수 토크 모드
                damping=0.0,
                effort_limit=10.0,
            ),
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# Scene 설정 — 3 개의 레버 + 3 개의 sensor platform + 3 개의 ContactSensor
# ══════════════════════════════════════════════════════════════════════════
@configclass
class TorqueLeverSceneCfg(InteractiveSceneCfg):
    """모터+레버 3개를 나란히, 각 빔 끝 아래에 platform 을 둔다."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.85, 0.85, 0.9)),
    )

    # ── 레버 2개 ────────────────────────────────────────────────────────
    # 모터 피벗을 x = ±LEVER_X_OFFSET 에 좌우 대칭 배치.
    lever_medium: ArticulationCfg = _make_lever_cfg(
        "medium", LEVERS[0][1], -LEVER_X_OFFSET,
    ).replace(prim_path="{ENV_REGEX_NS}/LeverMedium")
    lever_long:   ArticulationCfg = _make_lever_cfg(
        "long",   LEVERS[1][1], +LEVER_X_OFFSET,
    ).replace(prim_path="{ENV_REGEX_NS}/LeverLong")

    # ── Sensor platform 2개 ─────────────────────────────────────────────
    # 빔이 수평일 때 자연스럽게 접촉하도록 platform top 이 z ≈ 0.48 이 되게 배치
    # (platform 크기 0.10×0.10×0.10, 중심 z = PIVOT_HEIGHT - 0.07 = 0.43 → top = 0.48).
    platform_medium = AssetBaseCfg(
        prim_path="/World/PlatformMedium",
        spawn=_PLATFORM_SPAWN,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-LEVER_X_OFFSET + LEVERS[0][1], 0.0, PIVOT_HEIGHT - 0.07),
        ),
    )
    platform_long = AssetBaseCfg(
        prim_path="/World/PlatformLong",
        spawn=_PLATFORM_SPAWN,
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(+LEVER_X_OFFSET + LEVERS[1][1], 0.0, PIVOT_HEIGHT - 0.07),
        ),
    )

    # ── ContactSensor 2개 ───────────────────────────────────────────────
    # 각 빔 링크에 부착 → platform 과 닿을 때 net contact force 측정.
    contact_medium = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeverMedium/beam_link",
        update_period=0.0, history_length=1, debug_vis=True,
    )
    contact_long = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/LeverLong/beam_link",
        update_period=0.0, history_length=1, debug_vis=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# 시뮬레이션
# ══════════════════════════════════════════════════════════════════════════
def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    """세 레버에 동일 토크를 주며 ContactSensor 의 힘을 기록·표시한다."""

    # 엔티티 꺼내기 — name, length, color, articulation, sensor 묶음
    arms = []
    for (name, length, color), art_key, sen_key in [
        (LEVERS[0], "lever_medium", "contact_medium"),
        (LEVERS[1], "lever_long",   "contact_long"),
    ]:
        arms.append({
            "name": name,
            "length": length,
            "color": color,
            "art": scene[art_key],
            "sen": scene[sen_key],
        })

    sim_dt = sim.get_physics_dt()
    print("=" * 70)
    print("[INFO] 모터+레버 토크/접촉력 데모")
    for a in arms:
        print(f"  {a['name']:>6s}: L = {a['length']:.2f} m, "
              f"빔 질량 = {BEAM_MASS} kg, 단면 = {BEAM_SIDE} m")
    print(f"  토크 프로파일: 0 ~ {TORQUE_AMPL} Nm,  주기 {TORQUE_PERIOD} s")
    print("=" * 70)

    # ── matplotlib ───────────────────────────────────────────────────────
    if INTERACTIVE_PLOT:
        plt.ion()
    fig, (ax_tau, ax_f) = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
    fig.suptitle("Torque (Nm) → Contact Force (N) for three beam lengths",
                 fontsize=13)

    # 상단: 인가 토크
    (tau_line,) = ax_tau.plot([], [], "k-", lw=1.6, label="motor torque τ")
    ax_tau.set_ylabel("τ  [Nm]")
    ax_tau.set_ylim(-0.1, TORQUE_AMPL * 1.1)
    ax_tau.grid(alpha=0.3)
    ax_tau.legend(loc="upper right")

    # 하단: 측정 힘 + 이론값 (점선)
    force_lines = []
    theory_lines = []
    for a in arms:
        (mline,) = ax_f.plot([], [], color=a["color"], lw=1.6,
                             label=f"L = {a['length']:.1f} m  (measured)")
        (tline,) = ax_f.plot([], [], color=a["color"], lw=1.0, ls="--", alpha=0.55,
                             label=f"L = {a['length']:.1f} m  (theory τ/L + mg/2)")
        force_lines.append(mline)
        theory_lines.append(tline)
    ax_f.set_xlabel("time  [s]")
    ax_f.set_ylabel("contact force  [N]")
    # y 한도: 가장 짧은 (= 큰 힘) 빔의 이론 피크 + 여유
    _min_L = min(L for _, L, _ in LEVERS)
    ax_f.set_ylim(0.0, TORQUE_AMPL / _min_L + BEAM_MASS * 9.81 / 2.0 + 2.0)
    ax_f.grid(alpha=0.3)
    ax_f.legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()

    # ── 데이터 버퍼 ──────────────────────────────────────────────────────
    t_hist: list[float] = []
    tau_hist: list[float] = []
    f_hist: list[list[float]] = [[] for _ in arms]
    theory_hist: list[list[float]] = [[] for _ in arms]

    # 시뮬 길이: 정확히 한 사이클 + 약간 (관찰 편의)
    TOTAL_STEPS = int(round((TORQUE_PERIOD * 1.5) / sim_dt))
    DISPLAY_INTERVAL = max(1, int(round(0.05 / sim_dt)))   # ~20 Hz 갱신

    # 안정화: 초기 100 스텝은 토크 0 으로 두고 빔이 platform 에 가만히 안착하게.
    SETTLE_STEPS = 100

    print(f"[INFO] 총 {TOTAL_STEPS} step ({TOTAL_STEPS * sim_dt:.1f} s), "
          f"settle {SETTLE_STEPS} step")

    step = 0
    while simulation_app.is_running() and step < TOTAL_STEPS + SETTLE_STEPS:
        if sim.is_stopped():
            break
        if not sim.is_playing():
            sim.step(render=True)
            continue

        # ── (1) 토크 계산 ────────────────────────────────────────────────
        if step < SETTLE_STEPS:
            tau = 0.0
        else:
            t = (step - SETTLE_STEPS) * sim_dt
            # 양의 반파(sin²) — 항상 빔을 platform 쪽으로만 누르도록
            tau = TORQUE_AMPL * (math.sin(math.pi * t / TORQUE_PERIOD)) ** 2

        # ── (2) 세 레버에 모두 같은 토크 인가 ────────────────────────────
        tau_tensor = torch.tensor([[tau]], device=sim.device)
        for a in arms:
            a["art"].set_joint_effort_target(tau_tensor)

        # ── (3) 씬 데이터 → 시뮬 → 버퍼 업데이트 ─────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        step += 1

        # ── (4) 측정 (settle 끝난 뒤부터 기록) ──────────────────────────
        if step <= SETTLE_STEPS:
            continue

        t_now = (step - SETTLE_STEPS) * sim_dt
        t_hist.append(t_now)
        tau_hist.append(tau)

        for i, a in enumerate(arms):
            # net_forces_w: (num_envs, num_bodies=1, 3). 크기로 환산.
            f_vec = a["sen"].data.net_forces_w[0, 0]
            f_mag = torch.norm(f_vec).item()
            f_hist[i].append(f_mag)
            # 이론값: τ/L + mg/2 (정적 평형 가정)
            theory_hist[i].append(tau / a["length"] + BEAM_MASS * 9.81 / 2.0)

        # ── (5) 그래프 갱신 ──────────────────────────────────────────────
        if INTERACTIVE_PLOT and step % DISPLAY_INTERVAL == 0:
            tau_line.set_data(t_hist, tau_hist)
            for i, (mline, tline) in enumerate(zip(force_lines, theory_lines)):
                mline.set_data(t_hist, f_hist[i])
                tline.set_data(t_hist, theory_hist[i])
            ax_tau.set_xlim(0, max(1.0, t_hist[-1]))
            ax_f.set_xlim(0, max(1.0, t_hist[-1]))
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

        # ── (6) 콘솔 로그 (약 1초 주기) ──────────────────────────────────
        if step % int(round(1.0 / sim_dt)) == 0:
            line = f"[t={t_now:5.2f}s] τ={tau:5.2f} Nm  | "
            for a, fl in zip(arms, f_hist):
                f = fl[-1] if fl else 0.0
                theory = tau / a["length"] + BEAM_MASS * 9.81 / 2.0
                line += f"L={a['length']:.1f}: F={f:5.2f}N (이론 {theory:5.2f})  "
            print(line)

    print("=" * 70)
    print(f"[INFO] 시뮬 종료 — {len(t_hist)} 측정점 기록")

    # ── 최종 그래프 저장 후 종료 ────────────────────────────────────────
    tau_line.set_data(t_hist, tau_hist)
    for i, (mline, tline) in enumerate(zip(force_lines, theory_lines)):
        mline.set_data(t_hist, f_hist[i])
        tline.set_data(t_hist, theory_hist[i])
    ax_tau.set_xlim(0, t_hist[-1] if t_hist else 1.0)
    ax_f.set_xlim(0, t_hist[-1] if t_hist else 1.0)

    save_path = os.path.join(SCRIPT_DIR, "11_torque_lever_result.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[INFO] result saved: {save_path}")

    if INTERACTIVE_PLOT:
        plt.ioff()
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════════
def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 200.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 3.5, 1.6], target=[0.0, 0.0, 0.45])

    scene_cfg = TorqueLeverSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    print("[INFO] InteractiveScene 생성 완료")

    sim.reset()
    print("[INFO] sim.reset() 완료")

    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()
    print("[DONE] 11 (torque-lever-contact) 종료")
