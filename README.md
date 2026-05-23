# IsaacSim 기초 강의용 코드

NVIDIA Isaac Sim / Isaac Lab을 활용한 로봇 시뮬레이션 기초 강의 예제 코드 모음입니다.

## 환경 정보

| 항목 | 버전 |
|------|------|
| **OS** | Ubuntu 24.04 LTS |
| **Isaac Sim** | 4.5 (isaacsim 5.1.0) |
| **Isaac Lab** | 2.3.2 |
| **Python** | 3.10+ |

## 실행 방법

각 예제는 IsaacLab 환경에서 실행합니다.

```bash
# IsaacLab 디렉토리에서 실행
cd <IsaacLab 경로>

# 개별 예제 실행 (예: 01번)
./isaaclab.sh -p <예제 경로>/01_launch_sim/01_launch_sim.py

# 또는 직접 python으로 실행
python <예제 경로>/01_launch_sim/01_launch_sim.py
```

> **참고**: `--headless` 옵션을 추가하면 GUI 없이 실행할 수 있습니다.

## Jupyter 실행 (`jupyter/` 폴더)

`jupyter/` 폴더의 `.ipynb` 튜토리얼은 별도 사전 준비가 필요합니다.

### ⚠️ ipykernel 6.x 필수

Isaac Sim의 `omni.kit.async_engine`은 자체 asyncio 이벤트 루프를 패치해서 사용합니다.
하지만 **ipykernel 7+** 는 셀을 async 컨텍스트에서 실행하기 때문에, `AppLauncher(...)` 호출
시점에 이미 이벤트 루프가 돌고 있어 다음 에러가 발생합니다.

```
RuntimeError: This event loop is already running
AttributeError: '_UnixSelectorEventLoop' object has no attribute '_old_agen_hooks'
```

따라서 ipykernel은 반드시 **6.x** 를 사용해야 합니다.

### 설치 절차 (한 번만)

**1) Isaac Lab venv에 ipykernel 6.x 설치**

```bash
~/isaac/env_isaaclab/bin/python -m pip install "ipykernel<7" "jupyter_client<8.7"
```

**2) Jupyter 커널 등록**

```bash
~/isaac/env_isaaclab/bin/python -m ipykernel install --user \
    --name isaaclab --display-name "Python (isaaclab)"
```

**3) Jupyter Lab 실행**

```bash
cd ~/isaac/isaacsim_tutorials/jupyter
~/isaac/env_isaaclab/bin/jupyter lab
```

노트북을 열고 우상단 커널 선택에서 **Python (isaaclab)** 을 고르세요.

### 사용 규칙

- 셀은 **위에서 아래로 한 번씩 순서대로** 실행하세요.
- 한 노트북당 Isaac Sim 앱은 **한 번만** 띄울 수 있습니다. 다시 처음부터 하려면 커널 재시작 후 진행하세요.
- `AppLauncher` 셀 직후의 **GUI 펌프 셀** (Tornado `PeriodicCallback`) 을 반드시 실행해야 뷰포트 휠 줌/카메라 조작이 멈추지 않습니다.
  asyncio task 방식(`asyncio.ensure_future`)은 ipykernel의 dispatch task와 충돌해서 `Cannot enter into task` 에러를 일으키므로 사용 금지입니다.
- 시뮬레이션 리셋은 반드시 **`await sim.reset_async()`** 를 사용하세요.
  동기 버전 `sim.reset()` 은 Kit 이벤트 루프와 충돌해 셀이 무한 대기 상태로 멈춥니다
  (Isaac Sim 공식 문서: "Extensions/Jupyter 워크플로우에서는 async 버전 사용").

## 강의 목차

### Part 1. 시뮬레이션 기초 (01~04)

| # | 폴더 | 내용 |
|---|------|------|
| 01 | `01_launch_sim` | 시뮬레이터 실행 - SimulationApp 초기화, 지면/조명 설정, 기본 시뮬레이션 루프 |
| 02 | `02_spawn_primitives` | 기본 도형 생성 - Cube, Sphere, Cone, Cylinder 등 USD 프리미티브 스폰 |
| 03 | `03_galileo_experiment` | 갈릴레오 자유낙하 실험 — 피사의 사탑에서 무게가 다른 두 공을 동시에 떨어뜨려 질량과 낙하 시간의 무관성 확인 |
| 04 | `04_simulation_loop` | 시뮬레이션 루프 - step(), reset(), physics dt, 에피소드 관리 |

### Part 2. 로봇 제어 및 센서 (05~14)

| # | 폴더 | 내용 |
|---|------|------|
| 05 | `05_spawn_robot` | 로봇 스폰 - Franka Panda 로봇 로드, 랜덤 토크 적용, 조인트 상태 읽기 |
| 06 | `06_robot_joint_control` | 조인트 제어 - 사인파 위치 제어, 중력 보상, matplotlib 실시간 플롯 |
| 07 | `07_scene_design` | 씬 디자인 - InteractiveScene 패턴, @configclass, 멀티 환경 구성 |
| 08 | `08_multi_robot` | 멀티 로봇 - Cartpole + Franka 병렬 환경에서 독립 제어 |
| 09 | `09_camera_sensor` | 카메라 센서 - RGB/Depth 이미지 캡처, 실시간 시각화 |
| 10 | `10_ray_caster` | 레이 캐스터 - LiDAR 유사 센서, 그리드 패턴 레이캐스팅 |
| 11 | `11_contact_sensor` | 접촉 센서 - 충돌 감지 및 접촉 힘 측정 |
| 12 | `12_forward_kinematics` | 순운동학 (FK) - 조인트 각도 -> End-Effector 위치 계산, 마커 시각화 |
| 13 | `13_diff_ik` | 역운동학 (Diff IK) - DLS 기반 미분 역운동학 컨트롤러, 목표 자세 추종 |
| 14 | `14_pick_and_place` | Pick & Place - 상태 머신 기반 물체 집기/놓기 태스크 |

### Part 3. 모바일 로봇 (15~19)

| # | 폴더 | 내용 |
|---|------|------|
| 15 | `15_spawn_mobile_robot` | 모바일 로봇 스폰 - JetBot 차동 구동, 유니사이클 모델 |
| 16 | `16_mobile_base_control` | 경로 추종 - 사각형/원형/8자 경로, 휠 오도메트리 |
| 17 | `17_mobile_navigation` | 목표점 내비게이션 - P 제어 기반 헤딩/거리 제어, 웨이포인트 추종 |
| 19 | `19_obstacle_avoidance` | 장애물 회피 - 거리 기반 충돌 감지, 반응형 조향 |

### Part 4. 강화학습 환경 (20~28)

| # | 폴더 | 내용 |
|---|------|------|
| 20 | `20_base_env` | RL 환경 기초 - ManagerBasedRLEnv 구조, Action/Observation/Reward Manager |
| 21 | `21_observation_reward` | 관측 및 보상 설계 - 커스텀 관측 그룹, 보상 함수, 종료 조건 |
| 22 | `22_action_space` | 행동 공간 정의 - Effort/Position/Velocity 액션 타입 |
| 23 | `23_custom_env_complete` | 커스텀 환경 완성 - 전체 MDP 컴포넌트를 포함한 완전한 RL 환경 |
| 24 | `24_train_cartpole` | PPO 학습 - RSL-RL 통합, OnPolicyRunner, TensorBoard 로깅 |
| 25 | `25_evaluate_policy` | 정책 평가 - 학습된 모델 로드, 에피소드 실행, 성능 평가 |
| 26 | `26_train_locomotion` | 보행 학습 - 다리 로봇 보행 정책 학습 |
| 27 | `27_terrain_generation` | 지형 생성 - 절차적 랜덤 지형 생성 |
| 28 | `28_domain_randomization` | 도메인 랜덤화 - 로봇/환경 파라미터 랜덤화를 통한 Sim-to-Real 전이 |

### Part 5. 커스텀 로봇 통합 - Pinky Pro (29~33)

> **사전 준비**: Part 5 예제를 실행하려면 Pinky Pro 에셋이 필요합니다. 아래 안내를 참고하세요.

| # | 폴더 | 내용 |
|---|------|------|
| 30 | `30_urdf_preparation` | URDF 준비 - Pinky Pro URDF 에셋 구성 (메시, 충돌 지오메트리) |
| 31 | `31_urdf_to_usd` | URDF->USD 변환 - UrdfConverter를 사용한 에셋 변환 파이프라인 |
| 32 | `32_pinky_control` | Pinky 제어 - 차동 구동 제어, 웨이포인트 내비게이션 |
| 33 | `33_pinky_sensors` | Pinky 센서 - 커스텀 로봇 센서 통합 (IMU, 인코더 등) |

#### Pinky Pro 에셋 준비

Part 5 (29~33번) 예제에서 사용하는 Pinky Pro 로봇 에셋은 별도 저장소에서 클론해야 합니다.

```bash
# 30_urdf_preparation 폴더에 Pinky Pro ROS 2 패키지 클론
cd 30_urdf_preparation
git clone https://github.com/pinklab-art/pinky_pro.git
```

31번의 변환 결과물은 30번(URDF 준비) → 31번(URDF->USD 변환) 순서로 실행하면 자동 생성됩니다.

## 폴더 구조

```
.
├── README.md
├── 01_launch_sim/          # 시뮬레이션 기초
│   └── 01_launch_sim.py
├── ...
├── 30_urdf_preparation/    # URDF 준비 (git clone 필요)
│   └── 30_urdf_preparation.py
├── 31_urdf_to_usd/         # URDF->USD 변환
│   └── 31_urdf_to_usd.py
├── 32_pinky_control/
│   └── 32_pinky_control.py
└── 33_pinky_sensors/
    └── 33_pinky_sensors.py
```

## 문의

본 강의 자료는 **PinkLAB**에서 제작하였습니다.

- **홈페이지**: [pinklab.art](https://pinklab.art)
- **이메일**: [contact@pinklab.art](mailto:contact@pinklab.art)

기업 협업, 교육 문의는 위 연락처로 편하게 문의해 주세요.

## 라이선스

본 코드는 교육 목적으로 작성되었습니다. 
