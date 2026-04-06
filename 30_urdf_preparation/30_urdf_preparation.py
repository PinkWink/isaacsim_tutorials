"""
30_urdf_preparation.py - ROS2 패키지에서 URDF 준비 및 구조 분석

ROS2 패키지(pinky_pro)에서 xacro → URDF 변환을 수행하고,
URDF의 link/joint 구조를 분석하여 Isaac Sim 변환을 준비합니다:
  - xacro → URDF 변환 (ROS2 Jazzy 환경 사용)
  - package:// 경로를 절대 경로로 치환
  - URDF link/joint 트리 구조 시각화
  - 메시 파일 존재 여부 검증
  - 로봇 물리 파라미터(질량, 관성) 요약

실행:
    source env_isaaclab/bin/activate
    cd lectures/30_urdf_preparation
    python 30_urdf_preparation.py
"""

import os
import re
import subprocess
import xml.etree.ElementTree as ET


# ══════════════════════════════════════════════════════════════════════════
#  경로 설정
# ══════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PINKY_REPO = os.path.join(SCRIPT_DIR, "pinky_pro")
PINKY_DESC = os.path.join(PINKY_REPO, "pinky_description")
XACRO_FILE = os.path.join(PINKY_DESC, "urdf", "robot.urdf.xacro")
URDF_OUTPUT = os.path.join(SCRIPT_DIR, "pinky_pro.urdf")


# ══════════════════════════════════════════════════════════════════════════
#  1단계: xacro → URDF 변환
# ══════════════════════════════════════════════════════════════════════════


def convert_xacro_to_urdf():
    """xacro 파일을 URDF로 변환합니다."""

    print("=" * 70)
    print("[1단계] xacro → URDF 변환")
    print("=" * 70)

    if not os.path.exists(XACRO_FILE):
        print(f"  ERROR: xacro 파일을 찾을 수 없습니다: {XACRO_FILE}")
        print(f"  먼저 pinky_pro 저장소를 clone하세요:")
        print(f"    cd {SCRIPT_DIR}")
        print(f"    git clone https://github.com/pinklab-art/pinky_pro.git")
        return False

    print(f"  xacro 입력: {XACRO_FILE}")
    print(f"  URDF 출력:  {URDF_OUTPUT}")

    # ament_index가 pinky_description 패키지를 찾을 수 있도록 임시 워크스페이스 구성
    fake_prefix = "/tmp/pinky_ws"
    os.makedirs(f"{fake_prefix}/share/pinky_description", exist_ok=True)

    # 심볼릭 링크로 urdf, meshes 디렉토리 연결
    for subdir in ["urdf", "meshes"]:
        link_path = f"{fake_prefix}/share/pinky_description/{subdir}"
        target_path = os.path.join(PINKY_DESC, subdir)
        if os.path.exists(link_path):
            os.remove(link_path)
        os.symlink(target_path, link_path)

    # ament marker 파일 생성
    marker_dir = f"{fake_prefix}/share/ament_index/resource_index/packages"
    os.makedirs(marker_dir, exist_ok=True)
    with open(f"{marker_dir}/pinky_description", "w") as f:
        f.write("pinky_description")

    # xacro 실행
    cmd = (
        f"source /opt/ros/jazzy/setup.bash && "
        f"export AMENT_PREFIX_PATH='{fake_prefix}:/opt/ros/jazzy' && "
        f"xacro {XACRO_FILE} "
        f"namespace:='' is_sim:=false cam_tilt_deg:=8 screen_tilt_deg:=-25 "
        f"-o {URDF_OUTPUT}"
    )

    result = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ERROR: xacro 변환 실패!")
        print(f"  stderr: {result.stderr}")
        return False

    # package:// 경로를 절대 경로로 치환
    with open(URDF_OUTPUT, "r") as f:
        content = f.read()

    content = content.replace("package://pinky_description/", f"{PINKY_DESC}/")

    with open(URDF_OUTPUT, "w") as f:
        f.write(content)

    line_count = content.count("\n")
    print(f"  변환 완료! ({line_count} 줄)")
    print(f"  package:// 경로 → 절대 경로 치환 완료")
    return True


# ══════════════════════════════════════════════════════════════════════════
#  2단계: URDF 구조 분석
# ══════════════════════════════════════════════════════════════════════════


def analyze_urdf():
    """URDF 파일의 link/joint 구조를 분석합니다."""

    print(f"\n{'=' * 70}")
    print("[2단계] URDF 구조 분석")
    print("=" * 70)

    tree = ET.parse(URDF_OUTPUT)
    root = tree.getroot()
    robot_name = root.attrib.get("name", "unknown")

    # ── Link 목록 ────────────────────────────────────────────────────────
    links = root.findall("link")
    print(f"\n  로봇 이름: {robot_name}")
    print(f"  총 Link 수: {len(links)}")
    print()

    total_mass = 0.0
    for link in links:
        name = link.attrib["name"]
        visual = link.find("visual")
        collision = link.find("collision")
        inertial = link.find("inertial")

        has_visual = "V" if visual is not None else "-"
        has_collision = "C" if collision is not None else "-"
        has_inertial = "I" if inertial is not None else "-"

        mass_val = 0.0
        if inertial is not None:
            mass_el = inertial.find("mass")
            if mass_el is not None:
                mass_val = float(mass_el.attrib.get("value", 0))
                total_mass += mass_val

        mesh_name = ""
        if visual is not None:
            geom = visual.find("geometry")
            if geom is not None:
                mesh = geom.find("mesh")
                if mesh is not None:
                    mesh_name = os.path.basename(mesh.attrib.get("filename", ""))

        print(f"  {name:<30s} [{has_visual}{has_collision}{has_inertial}] "
              f"mass={mass_val:.3f}kg  mesh={mesh_name}")

    print(f"\n  총 질량: {total_mass:.3f} kg")

    # ── Joint 목록 ───────────────────────────────────────────────────────
    joints = root.findall("joint")
    print(f"\n  총 Joint 수: {len(joints)}")
    print()

    movable_joints = []
    for joint in joints:
        name = joint.attrib["name"]
        jtype = joint.attrib["type"]
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        origin = joint.find("origin")

        xyz = origin.attrib.get("xyz", "0 0 0") if origin is not None else "0 0 0"

        type_icon = "🔄" if jtype == "continuous" else "🔒" if jtype == "fixed" else "↔"
        print(f"  {type_icon} {name:<40s} [{jtype:>10s}] {parent} → {child}")

        if jtype in ("continuous", "revolute", "prismatic"):
            axis_el = joint.find("axis")
            axis = axis_el.attrib.get("xyz", "0 0 0") if axis_el is not None else "0 0 0"
            movable_joints.append((name, jtype, axis))

    print(f"\n  움직이는 관절 ({len(movable_joints)}개):")
    for name, jtype, axis in movable_joints:
        print(f"    {name:<35s} type={jtype:<12s} axis=({axis})")

    # ── Link 트리 구조 ───────────────────────────────────────────────────
    print(f"\n  Link 트리:")
    parent_map = {}
    for joint in joints:
        parent = joint.find("parent").attrib["link"]
        child = joint.find("child").attrib["link"]
        jtype = joint.attrib["type"]
        parent_map[child] = (parent, jtype)

    def print_tree(link_name, depth=0):
        prefix = "    " + "  " * depth
        jinfo = ""
        if link_name in parent_map:
            _, jt = parent_map[link_name]
            jinfo = f" ({'fixed' if jt == 'fixed' else jt})"
        print(f"{prefix}├─ {link_name}{jinfo}")
        children = [(j.find("child").attrib["link"], j.attrib["type"])
                     for j in joints if j.find("parent").attrib["link"] == link_name]
        for child, _ in children:
            print_tree(child, depth + 1)

    # 루트 링크 찾기
    all_children = {j.find("child").attrib["link"] for j in joints}
    all_parents = {j.find("parent").attrib["link"] for j in joints}
    root_links = all_parents - all_children
    for rl in root_links:
        print_tree(rl)


# ══════════════════════════════════════════════════════════════════════════
#  3단계: 메시 파일 검증
# ══════════════════════════════════════════════════════════════════════════


def verify_meshes():
    """URDF에서 참조하는 메시 파일의 존재 여부를 확인합니다."""

    print(f"\n{'=' * 70}")
    print("[3단계] 메시 파일 검증")
    print("=" * 70)

    tree = ET.parse(URDF_OUTPUT)
    root = tree.getroot()

    mesh_files = set()
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename", "")
        if filename:
            mesh_files.add(filename)

    ok_count = 0
    missing_count = 0
    for mesh_path in sorted(mesh_files):
        exists = os.path.exists(mesh_path)
        status = "OK" if exists else "MISSING"
        if exists:
            ok_count += 1
            size = os.path.getsize(mesh_path)
            print(f"  [{status:7s}] {os.path.basename(mesh_path):30s} ({size:,} bytes)")
        else:
            missing_count += 1
            print(f"  [{status:7s}] {mesh_path}")

    print(f"\n  결과: {ok_count}개 OK, {missing_count}개 MISSING")
    if missing_count == 0:
        print("  모든 메시 파일이 정상입니다!")
    return missing_count == 0


# ══════════════════════════════════════════════════════════════════════════
#  메인
# ══════════════════════════════════════════════════════════════════════════


def main():
    print("\n" + "=" * 70)
    print("  30강: ROS2 패키지에서 URDF 준비 (Pinky Pro)")
    print("=" * 70)

    # 1단계: xacro → URDF
    if not os.path.exists(URDF_OUTPUT):
        if not convert_xacro_to_urdf():
            return
    else:
        print(f"\n[INFO] URDF 파일이 이미 존재합니다: {URDF_OUTPUT}")
        print("  (재변환하려면 파일을 삭제 후 다시 실행하세요)")

    # 2단계: URDF 구조 분석
    analyze_urdf()

    # 3단계: 메시 파일 검증
    verify_meshes()

    # 요약
    print(f"\n{'=' * 70}")
    print("[완료] URDF 준비 완료!")
    print(f"  URDF 파일: {URDF_OUTPUT}")
    print(f"  → 31강에서 이 URDF를 USD로 변환합니다.")
    print("=" * 70)


if __name__ == "__main__":
    main()
