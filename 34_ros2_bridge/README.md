# 34강 — Pinky Pro × ROS2 Jazzy (IsaacSim ROS2 Bridge)

Gazebo 대신 IsaacSim을 ROS2 시뮬레이터로 사용하는 최소 셋업입니다. 한 스크립트로
표준 게시 토픽(`/clock`, `/tf`, `/tf_static`, `/joint_states`, `/odom`, `/scan`)과
구독 토픽(`/cmd_vel`)을 모두 띄워서, 외부의 ROS2 도구(RViz, teleop, slam_toolbox,
nav2 등)와 그대로 연결합니다.

---

## 사전 준비

- Ubuntu 24.04 + ROS2 Jazzy 설치 (`/opt/ros/jazzy/setup.bash` 존재).
- IsaacSim 4.5+ / IsaacLab 2.3+ (Jazzy 지원 라인업).
- 30강(URDF 준비) → 31강(URDF→USD 변환) 완료.
  `31_urdf_to_usd/usd_output/pinky_pro.usd` 파일이 있어야 합니다.

---

## 운영 모드 두 가지

본 강의는 두 가지 운영 방식을 지원하며, RViz에서 무엇을 보고 싶은지에 따라 고릅니다.

| | **모드 ① 기본** | **모드 ② `--with_rsp`** |
|---|---|---|
| IsaacSim 인자 | (없음) | `--with_rsp` |
| 정적 TF (base_footprint→base_link→rplidar_link) | IsaacSim 게시 | robot_state_publisher 게시 |
| 휠 회전 TF (base_link→l_wheel 등) | 없음 | robot_state_publisher 게시 |
| RViz RobotModel (메시 표시) | ❌ | ✅ |
| 호스트에서 띄울 노드 수 | RViz, teleop만 | RViz, teleop, **robot_state_publisher** |
| 셋업 단순성 | ⭐⭐⭐ | ⭐⭐ |

---

## 빠른 실행

### 공통 — 모든 셸에서 먼저

```bash
source /opt/ros/jazzy/setup.bash         # ★ 반드시 venv보다 먼저
```

> IsaacSim 번들 ROS2 라이브러리(`isaacsim/exts/.../jazzy/lib/`)는 `libament_index_cpp.so`가
> 빠져 있어 단독으로 동작하지 않습니다. 시스템 ROS2를 source해야 IsaacSim 브릿지가
> startup에 성공하고, 나아가 `/scan`을 게시할 LaserScan 라이터도 등록됩니다.

### 모드 ① — 기본 (LaserScan만 보면 됨)

```bash
# 터미널 1: IsaacSim
source /opt/ros/jazzy/setup.bash
source env_isaaclab/bin/activate
python lectures/34_ros2_bridge/34_ros2_bridge.py

# 터미널 2: 토픽 확인
source /opt/ros/jazzy/setup.bash
ros2 topic list
# /clock /cmd_vel /joint_states /odom /scan /tf /tf_static 모두 보여야 정상

# 터미널 3: 키보드 주행 (※ 이 창에 포커스가 있어야 키 캡처됨)
source /opt/ros/jazzy/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args -p use_sim_time:=true

# 터미널 4: RViz
source /opt/ros/jazzy/setup.bash
rviz2 --ros-args -p use_sim_time:=true
```

RViz 디스플레이 설정은 아래 **"RViz 셋업"** 섹션 참조.

### 모드 ② — `--with_rsp` (RobotModel까지 보기)

추가로 `robot_state_publisher`를 띄우면 됩니다.

```bash
# 터미널 1: IsaacSim
source /opt/ros/jazzy/setup.bash
source env_isaaclab/bin/activate
python lectures/34_ros2_bridge/34_ros2_bridge.py --with_rsp

# 터미널 2: robot_state_publisher
#   ※ sed로 URDF의 mesh 절대 경로를 file:// URI로 변환합니다.
#     URDF에 "/home/.../mesh.dae"가 박혀 있는데, RViz의 resource_retriever는
#     URI 스킴이 없는 절대 경로를 거부합니다.
source /opt/ros/jazzy/setup.bash
ros2 run robot_state_publisher robot_state_publisher \
  --ros-args -p use_sim_time:=true \
  -p robot_description:="$(sed 's|filename=\"/|filename=\"file:///|g' \
    ~/isaac/isaacsim_tutorials/30_urdf_preparation/pinky_pro.urdf)"

# 터미널 3, 4: 모드 ①과 동일 (teleop, RViz)
```

---

## RViz 셋업

- **Fixed Frame**: `odom`
- **LaserScan** 디스플레이 — Topic: `/scan`
  - ⚠ Topic 펼쳐서 **Reliability Policy = Best Effort** 설정 필수
    (RTX LiDAR는 Best Effort, RViz 기본은 Reliable. 미일치 시 표시 안 됨.)
- **TF** 디스플레이
- **Odometry** — Topic: `/odom`
- **RobotModel** (모드 ②만) — Description Source = `Topic`, Topic = `/robot_description`

---

## TF 토폴로지

### 모드 ① 토폴로지

```
/tf         odom ──(동적, ComputeOdom)── base_footprint
/tf_static  base_footprint ── base_link ── rplidar_link
```

세 개의 `ROS2PublishRawTransformTree` 노드로 명시 게시:
- `odom → base_footprint` (동적) — `ComputeOdom`의 position/orientation 그대로
- `base_footprint → base_link` (정적) — URDF `base_link_fixed_joint` xyz=(0,0,0.028)
- `base_link → rplidar_link` (정적, **URDF 두 fixed joint 합성**)
  - `base_link → rplidar_mount`: xyz=(-0.017, 0, 0.067)
  - `rplidar_mount → rplidar_link`: xyz=(0, 0, 0.030), Z축 180°
  - 합성: T=(-0.017, 0, 0.097), 쿼터니언 (x,y,z,w)=(0,0,1,0)

`ROS2PublishTransformTree`(USD prim 트리 기반)는 RigidBodyAPI가 없는 빈 링크
(`<link name="rplidar_link"/>` 같은)를 건너뛰므로, 정적 TF는 모두
`RawTransformTree`로 직접 발행합니다.

### 모드 ② 토폴로지

```
/tf         odom ──(동적, ComputeOdom)── base_footprint
            (그 외 모든 TF는 robot_state_publisher가 URDF에서 생성)
/tf_static  base_footprint → base_link → l_wheel, r_wheel, rplidar_mount → rplidar_link, ...
```

`--with_rsp` 플래그가 켜지면 IsaacSim 쪽 두 정적 TF 노드(`PubTfFpBase`,
`PubTfBaseLidar`)가 그래프에 추가조차 되지 않습니다 → TF "multiple parents"
충돌 회피.

---

## Gazebo ↔ IsaacSim 대응표

기존 `gazebo_ros` 플러그인을 쓰던 사용자라면 1:1 대응:

| 역할 | Gazebo | IsaacSim |
|---|---|---|
| 시뮬 시간 (`/clock`) | `<plugin name="ros_clock">` | OmniGraph `ROS2PublishClock` |
| 차동 구동 (`/cmd_vel` → 바퀴) | `gazebo_ros_diff_drive` | `ROS2SubscribeTwist` → `BreakVector3` ×2 → `DifferentialController` → `IsaacArticulationController` |
| 오도메트리 (`/odom`) | `gazebo_ros_p3d` 또는 diff_drive 옵션 | `IsaacComputeOdometry` → `ROS2PublishOdometry` |
| TF 트리 (`/tf`, `/tf_static`) | `robot_state_publisher` | `ROS2PublishRawTransformTree` × N (또는 모드 ②에서 rsp 사용) |
| 관절 상태 (`/joint_states`) | `gazebo_ros_joint_state_publisher` | `ROS2PublishJointState` |
| LiDAR (`/scan`) | `gazebo_ros_ray_sensor` | RTX 2D LiDAR + `rep.writers["RtxLidarROS2PublishLaserScan"]` |
| `use_sim_time` | 동일 | 동일 |
| URDF 플러그인 | `<gazebo><plugin>` 블록 | URDF에는 없음. OmniGraph로 별도 |

---

## 코드 구조 (`34_ros2_bridge.py`)

```
0. ROS_DISTRO=jazzy 사전 점검 (안 되면 친절히 종료)
1. AppLauncher → enable_extension("isaacsim.ros2.bridge")
2. Pinky USD 로드 → stage traverse로 articulation root 자동 발견
3. 휠 조인트(l/r_wheel_joint)에 UsdPhysics.DriveAPI 명시 설정
   - stiffness=0, damping=10  (32강 ImplicitActuatorCfg와 동일)
4. 미로(4m × 4m 외벽 + 내부 장애물 2개) 배치 — LiDAR가 맞힐 표적
5. RTX 2D LiDAR 부착 → rep.writers["RtxLidarROS2PublishLaserScan"]로 /scan 파이프라인
6. OmniGraph 구성 (with_rsp 여부에 따라 정적 TF 노드 포함/제외):
   - /clock, /tf(동적 odom→base_footprint), /tf_static ×2(모드①만), /joint_states, /odom
   - /cmd_vel → BreakVector3 ×2 → DifferentialController → IsaacArticulationController
7. world.reset() + world.play()  (둘 다 필수)
8. 메인 루프 (world.step(render=True))
```

---

## 자주 빠지는 함정과 해결

### 1. 시작 직후 브릿지 실패 — `libament_index_cpp.so` 못 찾음

**증상**: 콘솔에 `Could not load librmw_implementation.so`, 이어서
`RtxLidarROS2PublishLaserScan was found in registry` 에러 줄줄이.

**원인**: 시스템 ROS2 Jazzy를 source 안 함. IsaacSim 번들 jazzy lib에는
`libament_index_cpp.so`가 빠져 있어, 단독으로는 못 뜸. 브릿지가 startup 직후
shutdown되면서 RTX LiDAR 라이터까지 등록 안 됨.

**해결**: 매 셸에서 `source /opt/ros/jazzy/setup.bash`를 venv 활성화보다 먼저.
스크립트는 `ROS_DISTRO=jazzy`가 아니면 시작 시 멈추도록 사전 점검합니다.

---

### 2. `ros2 topic list`에 아무것도 안 보임

**원인**: 호스트와 IsaacSim의 **RMW(미들웨어) 불일치** 또는 **`ROS_DOMAIN_ID` 차이**.
IsaacSim 브릿지 기본은 FastRTPS.

**해결**:
```bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# ROS_DOMAIN_ID도 양쪽 동일해야 (기본 0)
ros2 topic list
```

---

### 3. `/scan` 토픽이 *아예* 보이지 않음

**원인**: `world.play()`를 호출 안 하면 timeline이 정지 상태라 RTX LiDAR 렌더
파이프라인이 안 돔 → LaserScan 라이터 미발화 → publisher가 ROS 그래프에 등록조차
안 됨. (`/odom`은 OmniGraph가 직접 발화하므로 timeline 무관하게 보이는 비대칭이
여기서 발생.)

**해결**: 본 스크립트는 `world.reset()` 직후 `world.play()`를 호출합니다.
`/scan`이 보이는데 값이 모두 `inf`라면 단지 LiDAR 주변에 장애물이 없는 정상 상태
— 미로 자동 배치로 해결.

---

### 4. `Failed to find articulation at '/World/pinky'` 가 줄줄이 찍힘

**원인**: URDF→USD 변환기는 `ArticulationRootAPI`를 보통 `/World/pinky` 자체가
아니라 그 아래 `base_link` 또는 `base_footprint`에 부착합니다. articulation 관련
OmniGraph 노드(`ArticulationController`, `IsaacComputeOdometry`,
`PublishJointState`)의 `targetPrim`은 정확한 root를 가리켜야 함.

**해결**: 본 스크립트는 시작 시 stage를 traverse해서 `ArticulationRootAPI`가 붙은
prim을 자동 발견하고 `[INFO] 발견된 articulation root: ...`를 출력합니다.
그 경로를 모든 articulation 관련 노드에 자동 사용.

---

### 5. RViz `Fixed Frame [odom] does not exist` 또는 TF lookup 실패

**원인 A**: 호스트의 RViz/teleop 노드들에 `use_sim_time:=true`가 없음. 그러면
호스트가 wall clock으로 TF 타임스탬프를 비교해 항상 실패.

**해결**: 모든 호스트 노드에 `--ros-args -p use_sim_time:=true` 명시.
먼저 `/clock`이 흐르는지 확인: `ros2 topic echo /clock --once`.

**원인 B**: TF 트리에 `odom` 프레임 자체가 없음. `ROS2PublishOdometry`는 메시지만
발행하고 TF는 안 만듭니다. `ROS2PublishRawTransformTree`로 `odom→base_footprint`를
명시 발행해야 함.

**해결**: 본 스크립트가 `PubTfOdomFp` 노드로 자동 처리.
진단: `ros2 run tf2_ros tf2_echo odom base_footprint`.

---

### 6. `ros2 topic echo /scan`은 흐르는데 RViz에서만 안 보임

**원인 A (가장 흔함)**: TF에 `rplidar_link`가 없음.
URDF의 `<link name="rplidar_link"/>`처럼 RigidBody가 없는 빈 링크는
`ROS2PublishTransformTree`가 건너뜁니다.

**해결**: 본 스크립트는 `ROS2PublishRawTransformTree`로 `base_link→rplidar_link`를
정적으로 직접 발행. 진단: `ros2 run tf2_ros tf2_echo odom rplidar_link`.

**원인 B (QoS 미스매치)**: RTX LaserScan 퍼블리셔는 `BEST_EFFORT`인데 RViz는 기본
`RELIABLE`. `echo`는 자동 협상되지만 RViz는 수동 설정 필요.

**해결**: RViz LaserScan 디스플레이의 Topic 펼쳐서 **Reliability Policy =
Best Effort**, Durability=Volatile, History=Keep Last로 변경.
진단: `ros2 topic info /scan --verbose | grep Reliability`.

---

### 7. `/cmd_vel`을 보내도 로봇이 안 움직임

`ros2 topic info /cmd_vel --verbose`로 `_ActionGraph_SubTwist`가 subscriber로
잡혀 있는지 먼저 확인. 그래도 안 움직이면 아래 셋 중 하나:

**원인 A (가장 흔함)**: 휠 조인트에 `UsdPhysics.DriveAPI`가 없음.
URDF→USD 변환기가 continuous 조인트는 만들지만 DriveAPI는 자동으로 안 채워줍니다
→ ArticulationController.velocityCommand가 어디로도 안 감. IsaacLab의
`ImplicitActuatorCfg`는 이걸 자동 처리하지만, 본 강의는 raw IsaacSim API라
직접 박아야 함.

**해결**: 본 스크립트는 시작 시 `_setup_wheel_velocity_drive()`로 l/r_wheel_joint에
`stiffness=0, damping=10, type="force"`를 명시 설정 (32강 값과 동일).
콘솔에 `[INFO] l_wheel_joint 휠 drive 설정: ...` 두 줄이 나오면 정상.

**원인 B**: `SubTwist.outputs:linearVelocity`(vec3)와
`DiffDrive.inputs:linearVelocity`(scalar)는 타입이 달라 직접 연결하면 OmniGraph가
silently 거부합니다.

**해결**: 본 스크립트는 사이에 `omni.graph.nodes.BreakVector3` 두 개를 끼워
`.x`(전진), `.z`(회전)만 뽑아 넘김.

**원인 C**: teleop_twist_keyboard는 **그 터미널 창에 포커스**가 있어야 키를 캡처.
포커스 없으면 `linear: x: 0.0`만 흐름.

**우회 테스트**:
```bash
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2}, angular: {x: 0.0, y: 0.0, z: 0.0}}" --rate 10
```

---

### 8. RViz RobotModel — `Could not load resource ... Unable to open file`

**원인**: Pinky URDF에는 `<mesh filename="/home/.../base_link.dae">`처럼 그냥
절대 경로가 박혀 있지만, RViz의 `resource_retriever`는 **URI 스킴**(`file://` 또는
`package://`)이 없는 순수 절대 경로를 거부합니다. 파일이 실재해도 열지 못함.

**해결**: `robot_state_publisher`에 URDF를 넘기기 전에 `sed`로 한 번 변환.

```bash
sed 's|filename="/|filename="file:///|g' pinky_pro.urdf
```

모든 `filename="/...` → `filename="file:///...`로 바꿔서 robot_description
파라미터에 그 결과를 넣으면 RViz가 정상적으로 메시를 로드합니다.

---

## 다음 단계 (강의 외 확장)

- **SLAM**: `slam_toolbox`로 매핑
  ```bash
  ros2 launch slam_toolbox online_async_launch.py use_sim_time:=true
  ```
- **자율 주행**: `nav2_bringup`로 nav2 스택. `/scan`/`/odom`/`/tf`만 있으면 그대로 동작.
- **카메라 추가**: `ROS2CameraHelper`로 `/image_raw` 게시 (33강의 카메라 prim 활용).
- **3D 포인트클라우드**: LiDAR config를 `Example_Rotary` (3D)로 바꾸고
  `RtxLidarROS2PublishPointCloud` writer 추가.
