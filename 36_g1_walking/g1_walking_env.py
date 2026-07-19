"""
g1_walking_env.py - G1 Walking(모션 모방) 공유 환경 모듈 (실행 파일 아님)

36강 시리즈(36_1 ~ 36_3)가 공통으로 사용하는 환경 정의를 모아둔 모듈입니다:
  - MotionReference: 리타게팅된 사람 보행 모션(.pkl)을 로드/관리하는 클래스
  - G1SceneCfg: 평지 + Unitree G1 휴머노이드 + 조명 + 발 접촉 센서
  - ObservationsCfg: 관측 8항목 (35강의 6항목 + 레퍼런스 관절 + 보행 위상)
  - ActionsCfg: 몸체 23관절 위치 명령 (손가락 제외 — 보행에 불필요)
  - RewardsCfg: "레퍼런스 모션 따라 걷기" 보상 설계 (관절 추종이 핵심)
  - EventsCfg: RSI(Reference State Initialization) — 모방 학습의 핵심 기법
  - TerminationsCfg: 에피소드 종료 3조건 (시간초과 / 기울어짐 / 추락)
  - G1WalkingEnvCfg: 위 요소를 조립한 완성 환경 설정
  - make_ppo_runner_cfg(): RSL-RL PPO 하이퍼파라미터 설정

모션 모방(motion imitation) 학습이란?
    사람의 보행 모션 캡처 데이터를 G1 관절로 리타게팅한 "레퍼런스 모션"을
    정답지로 삼아, 매 순간 로봇이 레퍼런스와 같은 자세를 취하도록 보상을
    주는 방식입니다. 순수 보상 설계(velocity tracking)보다 자연스러운
    걸음걸이를 얻기 쉽습니다.

주의:
    이 모듈은 isaaclab 패키지를 import하므로, 반드시 AppLauncher로
    시뮬레이터를 먼저 실행한 뒤에 import해야 합니다. (각 실행 스크립트 참고)
"""

import glob
import os
import pickle
import re
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

# feet_slide 등 보행 전용 보상 함수는 isaaclab_tasks의 velocity 태스크에서 재사용
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as velocity_mdp

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  레퍼런스 모션 데이터
# ══════════════════════════════════════════════════════════════════════════

# 보행 레퍼런스 모션(.pkl)은 용량 문제로 저장소에 포함하지 않습니다.
# 이 폴더의 데이터를 우선 사용하고, 없으면 g1/ 폴더에서 찾습니다.
# 직접 내려받기:
#   git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions g1_retargeted_motions
_THIS_DIR = Path(__file__).resolve().parent
_MOTION_CANDIDATES = [
    _THIS_DIR / "g1_retargeted_motions",
    _THIS_DIR.parent / "g1" / "g1_retargeted_motions",
]
DEFAULT_MOTION_NAME = "B3_-_walk1_stageii.pkl"

# B3_-_walk1 클립(229프레임 @30fps)은 전체가 순환하지 않지만,
# 프레임 42~155 구간(113프레임, 3.77초)은 시작/끝 관절각 차이가 최대
# 0.03rad로 깔끔하게 이어지는 "한 걸음 주기 × 3"입니다.
# 학습에는 이 구간만 잘라 무한 반복(modulo)하는 순환 레퍼런스로 씁니다.
CYCLE_START, CYCLE_END = 42, 155

# pkl의 23개 관절 순서 (0~12: 다리+허리, 13~22: 팔)
PKL_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_roll_joint",
]

# pkl 관절명 → G1_MINIMAL 자산 관절명 변환표.
# G1_MINIMAL에는 손목(wrist)이 없어 팔꿈치 roll로 근사합니다.
ALIAS = {
    "waist_yaw_joint": "torso_joint",
    "left_wrist_roll_joint": "left_elbow_roll_joint",
    "right_wrist_roll_joint": "right_elbow_roll_joint",
}

PKL_ARM_RANGE = range(13, 23)  # pkl 인덱스 중 팔에 해당하는 구간

# 행동/관측에 사용할 몸체 23관절 (손가락 14관절 제외 — 보행과 무관)
BODY_JOINT_EXPR = [
    ".*_hip_yaw_joint", ".*_hip_roll_joint", ".*_hip_pitch_joint",
    ".*_knee_joint", ".*_ankle_pitch_joint", ".*_ankle_roll_joint",
    "torso_joint",
    ".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint", ".*_elbow_roll_joint",
]


class MotionReference:
    """리타게팅된 보행 모션을 로드하고 학습/재생에 필요한 형태로 제공합니다.

    - load_file():   pkl 로드 + 순환 구간 트림 + 속도/높이 사전 계산 (CPU)
    - initialize():  env 생성 후 1회 — 텐서를 GPU로 옮기고 관절 매핑/버퍼 준비
    - frame_idx():   각 env의 현재 레퍼런스 프레임 번호 (RSI 위상 phase0 반영)
    """

    def __init__(self):
        self._loaded = False
        self._ready = False

    # ── 모션 이름/경로 → 실제 파일 경로 완성 ─────────────────────────────
    @staticmethod
    def resolve_motion_path(motion: str | Path | None = None) -> Path:
        """모션 지정을 실제 pkl 경로로 완성합니다.

        지원하는 지정 방식 (--motion_pkl 인자):
          - 생략               → 기본 보행 클립 (B3_-_walk1_stageii)
          - 전체/상대 경로      → 그대로 사용
          - 이름만 (확장자 생략 가능) → 데이터셋 하위 폴더에서 자동 검색
            예: "run1_subject2", "walk_turn_left" (부분 일치도 허용)
        """
        # (1) 실제 존재하는 경로면 그대로 사용
        if motion is not None and Path(motion).exists():
            return Path(motion)

        name = str(motion) if motion is not None else DEFAULT_MOTION_NAME
        if not name.endswith(".pkl"):
            name += ".pkl"

        for base in _MOTION_CANDIDATES:
            if not base.exists():
                continue
            # (2) 정확한 이름으로 하위 폴더 검색 (ACCAD_retargeted 등)
            hits = sorted(base.glob(f"**/{name}"))
            # (3) 없으면 부분 일치 검색 (예: "walk1" → B3_-_walk1_stageii.pkl)
            if not hits:
                hits = sorted(base.glob(f"**/*{name[:-4]}*.pkl"))
            if hits:
                if len(hits) > 1:
                    print(f"[MotionRef] '{motion}'에 해당하는 파일 {len(hits)}개 중 첫 번째 사용:")
                    for h in hits[:5]:
                        print(f"    {h.relative_to(base)}")
                return hits[0]

        raise FileNotFoundError(
            f"레퍼런스 모션 pkl을 찾을 수 없습니다: {motion}\n"
            f"검색 위치: {[str(p) for p in _MOTION_CANDIDATES]}\n"
            "데이터가 없다면 아래 명령으로 내려받으세요:\n"
            "  git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions "
            f"{_MOTION_CANDIDATES[0]}"
        )

    # ── 1단계: 파일 로드 (시뮬레이터/env 불필요) ─────────────────────────
    def load_file(self, motion: str | Path | None = None) -> None:
        pkl_path = self.resolve_motion_path(motion)
        # 데이터셋에 두 저장 형식이 혼재: ACCAD/dance_db는 일반 pickle,
        # lafan1/kungfu는 joblib. pickle 실패 시 joblib으로 재시도한다.
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
        except pickle.UnpicklingError:
            import joblib
            data = joblib.load(pkl_path)
        motion_data = data[list(data.keys())[0]]

        self.fps = int(motion_data["fps"])
        # 재생(36_1)용: 트림하지 않은 전체 클립
        self.dof_full = torch.tensor(motion_data["dof"], dtype=torch.float32)
        self.root_trans_full = torch.tensor(motion_data["root_trans_offset"], dtype=torch.float32)
        self.root_rot_full = torch.tensor(motion_data["root_rot"], dtype=torch.float32)  # xyzw!
        self.contact_full = torch.tensor(motion_data["contact_mask"], dtype=torch.float32)

        # 학습용: 순환 구간만 트림
        # (CYCLE_START/END는 기본 보행 클립 전용 — 다른 모션은 전체 클립 사용)
        if pkl_path.name == DEFAULT_MOTION_NAME:
            s, e = CYCLE_START, CYCLE_END
        else:
            s, e = 0, self.dof_full.shape[0]
        self.dof = self.dof_full[s:e].clone()
        root_trans = self.root_trans_full[s:e]
        self.T = self.dof.shape[0]

        # 프레임별 평면 이동 속도 (마지막 프레임은 순환이므로 첫 프레임과 연결)
        nxt = torch.roll(root_trans[:, :2], shifts=-1, dims=0)
        self.ref_speed = torch.linalg.norm(nxt - root_trans[:, :2], dim=1) * self.fps
        self.ref_speed[-1] = self.ref_speed[-2]  # wrap 지점의 좌표 점프 제거
        # 프레임별 루트 높이
        self.ref_root_z = root_trans[:, 2].clone()
        # 프레임별 관절 속도 (유한 차분, 순환 wrap)
        dof_nxt = torch.roll(self.dof, shifts=-1, dims=0)
        self.ref_dof_vel = (dof_nxt - self.dof) * self.fps
        self.ref_dof_vel[-1] = self.ref_dof_vel[-2]

        self.pkl_name = pkl_path.name
        self._loaded = True
        self._ready = False
        print(f"[MotionRef] {self.pkl_name}: 전체 {self.dof_full.shape[0]}프레임 @ {self.fps}fps, "
              f"순환 구간 [{s}:{e}] → {self.T}프레임 ({self.T / self.fps:.2f}초), "
              f"평균 속도 {self.ref_speed.mean():.2f} m/s")

    # ── 관절 매핑 (pkl 23관절 → 로봇 관절 인덱스) ────────────────────────
    def build_joint_map(self, joint_names: list[str], device: str) -> torch.Tensor:
        indices = []
        for pkl_name in PKL_JOINT_NAMES:
            robot_name = ALIAS.get(pkl_name, pkl_name)
            if robot_name not in joint_names:
                raise ValueError(
                    f"로봇에서 관절을 찾을 수 없습니다: {pkl_name} → {robot_name}\n"
                    f"로봇 관절 목록: {joint_names}"
                )
            indices.append(joint_names.index(robot_name))
        return torch.tensor(indices, dtype=torch.long, device=device)

    # ── 2단계: env 준비 (관측/보상 함수에서 lazy 호출) ───────────────────
    def initialize(self, env: ManagerBasedRLEnv) -> None:
        if self._ready:
            return
        if not self._loaded:
            self.load_file()
        device = env.device
        for name in ("dof", "ref_speed", "ref_root_z", "ref_dof_vel"):
            setattr(self, name, getattr(self, name).to(device))

        robot = env.scene["robot"]
        self.joint_map = self.build_joint_map(robot.data.joint_names, device)
        self.arm_map = self.joint_map[PKL_ARM_RANGE.start:PKL_ARM_RANGE.stop]
        # RSI가 정해주는 env별 시작 위상(프레임 오프셋)
        self.phase0 = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        self._ready = True
        print(f"[MotionRef] 관절 매핑 완료: pkl 23관절 → 로봇 인덱스 {self.joint_map.tolist()}")

    # ── env별 현재 레퍼런스 프레임 번호 ──────────────────────────────────
    def frame_idx(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        t = env.episode_length_buf.float() * env.step_dt
        return (self.phase0 + (t * self.fps).long()) % self.T


# 모듈 싱글턴: 36_1/36_2/36_3 이 하나의 인스턴스를 공유
MOTION = MotionReference()


# ══════════════════════════════════════════════════════════════════════════
#  씬 구성: 평지 + G1 휴머노이드 + 발 접촉 센서
# ══════════════════════════════════════════════════════════════════════════


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """G1 Walking 학습 씬: 바닥 + G1 휴머노이드 + 조명 + 발 접촉 센서."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )
    # 발(ankle_roll 링크) 접촉 센서 — feet_slide 보상 계산에 사용
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link",
        history_length=3,
        track_air_time=True,
    )


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 관측
# ══════════════════════════════════════════════════════════════════════════


def ref_joint_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 위상의 레퍼런스 관절각 (기본자세 대비 상대값, 23차원).

    정책이 "지금 어떤 자세를 취해야 하는지"를 직접 볼 수 있게 합니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    default = env.scene["robot"].data.default_joint_pos[:, MOTION.joint_map]
    return MOTION.dof[idx] - default


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """보행 주기 위상을 sin/cos 2차원으로 표현 (주기의 시작=끝 연속성 보장)."""
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    phase = idx.float() / MOTION.T * 2.0 * torch.pi
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 보상
# ══════════════════════════════════════════════════════════════════════════


def joint_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """전신 23관절이 레퍼런스 관절각에 가까울수록 1에 가까운 보상 (핵심!)."""
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    cur = env.scene["robot"].data.joint_pos[:, MOTION.joint_map]
    err = torch.sum(torch.square(cur - MOTION.dof[idx]), dim=1)
    return torch.exp(-1.0 * err)


def arm_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """팔 10관절 추가 추종 보상 — 자연스러운 팔 흔들기를 유도합니다."""
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    arm_slice = slice(PKL_ARM_RANGE.start, PKL_ARM_RANGE.stop)
    cur = env.scene["robot"].data.joint_pos[:, MOTION.arm_map]
    err = torch.sum(torch.square(cur - MOTION.dof[idx][:, arm_slice]), dim=1)
    return torch.exp(-3.0 * err)


def speed_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """전진 속도가 레퍼런스의 프레임별 속도(평균 ~1.07m/s)에 가까울수록 보상.

    단순히 '빠를수록 보상'(unbounded)으로 하면 몸을 던지는 돌진을 배우므로,
    반드시 목표 속도 추종(exp 커널) 형태로 설계합니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    vx = env.scene["robot"].data.root_lin_vel_b[:, 0]
    return torch.exp(-4.0 * torch.square(vx - MOTION.ref_speed[idx]))


def ref_height_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """몸체 높이가 레퍼런스의 프레임별 높이(0.75~0.84m)에 가까울수록 보상.

    보행 중에는 몸이 리드미컬하게 오르내리므로 고정 높이(35강의 0.74m)가
    아니라 레퍼런스 높이를 따라가게 합니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(z - MOTION.ref_root_z[idx]) / 0.01)


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 종료 / 이벤트(RSI)
# ══════════════════════════════════════════════════════════════════════════


def base_height_termination(env: ManagerBasedRLEnv, limit: float = 0.4) -> torch.Tensor:
    """몸체 높이가 limit 아래로 떨어지면(=넘어지면) 에피소드 종료."""
    return env.scene["robot"].data.root_pos_w[:, 2] < limit


def reset_to_motion_frame(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """RSI(Reference State Initialization): 리셋 시 레퍼런스의 임의 프레임 자세로 시작.

    모방 학습의 핵심 기법입니다. 항상 '차렷 자세 + 프레임 0'에서 시작하면
    에이전트가 보행 주기의 뒷부분 상태를 경험하기 어렵고, 시작 자세와
    레퍼런스 위상이 어긋난 채 학습이 진행됩니다. RSI는 임의 프레임 f를 뽑아
      (1) 그 프레임의 관절각/관절속도를 로봇에 직접 기록하고
      (2) 루트 높이/전진 속도도 레퍼런스에 맞추고
      (3) phase0[env] = f 로 위상을 동기화합니다.
    """
    MOTION.initialize(env)
    robot = env.scene["robot"]
    device = env.device

    # (0) 임의 프레임 샘플 + 위상 동기화
    f = torch.randint(0, MOTION.T, (len(env_ids),), device=device)
    MOTION.phase0[env_ids] = f

    # (1) 관절 상태: 기본자세에서 매핑된 23관절만 레퍼런스로 덮어쓰기
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_vel = robot.data.default_joint_vel[env_ids].clone()
    joint_pos[:, MOTION.joint_map] = MOTION.dof[f]
    joint_vel[:, MOTION.joint_map] = MOTION.ref_dof_vel[f]
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # (2) 루트 상태: env 원점 + 레퍼런스 높이, 진행 방향(+x)은 그대로(yaw=0),
    #     전진 속도는 레퍼런스 속도로 설정 (넘어지지 않고 걷던 중처럼 시작)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root_state[:, 2] = MOTION.ref_root_z[f]
    root_state[:, 7] = MOTION.ref_speed[f]   # 몸 기준 전진(+x) = 월드 +x (yaw=0)
    root_state[:, 8:13] = 0.0
    robot.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════
#  MDP 구성: 행동 / 관측 / 보상 / 이벤트 / 종료
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """행동: 몸체 23관절의 목표 위치. 손가락 14관절은 보행과 무관해 제외.

    scale=0.25로 한 번에 크게 못 움직이게 제한 (35강과 동일).
    """

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=BODY_JOINT_EXPR, scale=0.25
    )


@configclass
class ObservationsCfg:
    """관측: 35강의 6항목 + 모방 학습용 2항목 (레퍼런스 관절각, 보행 위상)."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)              # 몸체 선속도 (3)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)              # 몸체 각속도 (3)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)    # 중력 방향 투영 (3)
        joint_pos = ObsTerm(                                       # 몸체 관절 각도 (23)
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_EXPR)},
        )
        joint_vel = ObsTerm(                                       # 몸체 관절 속도 (23)
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_EXPR)},
        )
        last_action = ObsTerm(func=mdp.last_action)                # 직전 행동 (23)
        ref_joint = ObsTerm(func=ref_joint_obs)                    # 레퍼런스 관절각 (23)
        phase = ObsTerm(func=gait_phase_obs)                       # 보행 위상 sin/cos (2)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True                          # 총 103차원

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """보상: '레퍼런스 모션을 따라 걷기'를 점수로 정의."""

    alive_bonus = RewTerm(func=mdp.is_alive, weight=1.0)                    # 살아있으면 점수
    joint_tracking = RewTerm(func=joint_tracking_reward, weight=5.0)        # 전신 자세 추종 (핵심!)
    arm_tracking = RewTerm(func=arm_tracking_reward, weight=2.0)            # 팔 흔들기 추종
    speed_tracking = RewTerm(func=speed_tracking_reward, weight=2.0)        # 전진 속도 추종
    height_reward = RewTerm(func=ref_height_reward, weight=2.0)             # 레퍼런스 높이 추종
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)   # 과도한 기울어짐 감점
    feet_slide = RewTerm(                                                   # 발이 땅에 닿은 채 미끄러지면 감점
        func=velocity_mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)                # 관절을 마구 흔들면 감점
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)            # 행동을 급격히 바꾸면 감점
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)          # 관절 한계를 넘으면 감점


@configclass
class EventsCfg:
    """이벤트: 리셋 시 RSI 적용 (레퍼런스의 임의 프레임 자세로 시작)."""

    reset_motion = EventTerm(func=reset_to_motion_frame, mode="reset")


@configclass
class TerminationsCfg:
    """종료: 넘어진 채로 시간을 낭비하지 않도록 에피소드를 빨리 끊는다."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)                                # 10초 경과
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.8})    # ~46도 이상 기울어짐
    base_height = DoneTerm(func=base_height_termination, params={"limit": 0.35})         # 0.35m 아래로 추락


# ══════════════════════════════════════════════════════════════════════════
#  완성 환경 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class G1WalkingEnvCfg(ManagerBasedRLEnvCfg):
    """G1 Walking(모션 모방) 학습용 완성 환경 설정."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=2048, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventsCfg = EventsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4              # 물리 4스텝마다 정책 1회 실행 (제어 주기 50Hz)
        self.episode_length_s = 10.0     # 에피소드 최대 10초 (순환 클립이라 길이 자유)
        self.sim.dt = 0.005              # 물리 스텝 200Hz
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
#  RSL-RL PPO 하이퍼파라미터 설정 (36_2 학습 / 36_3 평가 공용)
# ══════════════════════════════════════════════════════════════════════════


def make_ppo_runner_cfg(max_iterations: int = 2000) -> RslRlOnPolicyRunnerCfg:
    """G1 Walking PPO 설정. 평가 시에도 동일 설정으로 체크포인트를 로드합니다.

    35강(standing) 대비 달라진 점:
      - 네트워크 [256,128,64]: 관측 103차원 + 보행이라는 더 어려운 과제
      - entropy_coef 0.008: 레퍼런스가 답을 알려주므로 탐색 필요성이 낮음
      - num_mini_batches=4: 배치가 큼 (2048 env × 24 스텝 = 49152 샘플)
    """
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=max_iterations,
        save_interval=50,                     # 체크포인트 저장 주기
        experiment_name="g1_walking_tutorial",
        obs_groups={"policy": ["policy"]},
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,              # 가치 함수 손실 가중치
            use_clipped_value_loss=True,      # 가치 손실 클리핑
            clip_param=0.2,                   # PPO 클리핑 범위 ε
            entropy_coef=0.008,               # 엔트로피 보너스 (탐색 장려)
            learning_rate=1.0e-3,             # 학습률
            num_learning_epochs=5,            # 수집 데이터로 학습 반복 횟수
            num_mini_batches=4,               # 미니배치 수
            schedule="adaptive",              # 적응적 학습률 (KL 기반)
            gamma=0.99,                       # 할인율
            lam=0.95,                         # GAE lambda
            desired_kl=0.01,                  # 목표 KL divergence
            max_grad_norm=1.0,                # gradient 클리핑
        ),
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,               # 초기 탐색 노이즈
            actor_hidden_dims=[256, 128, 64],  # 정책(액터) 네트워크
            critic_hidden_dims=[256, 128, 64], # 가치(크리틱) 네트워크
            activation="elu",                 # 활성화 함수
            actor_obs_normalization=True,     # 관측 정규화 (학습 안정화)
            critic_obs_normalization=True,
        ),
    )


# ══════════════════════════════════════════════════════════════════════════
#  체크포인트 자동 탐색 (36_1 비교 재생 / 36_3 평가 공용)
# ══════════════════════════════════════════════════════════════════════════


def find_latest_checkpoint() -> str | None:
    """가장 최근 학습 체크포인트를 자동으로 찾습니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 탐색할 경로 후보 (우선순위 순)
    search_paths = [
        # 36_2와 같은 위치에서 실행한 경우
        os.path.join(os.getcwd(), "logs", "rsl_rl", "g1_walking_tutorial"),
        # 이 스크립트 폴더 기준
        os.path.join(script_dir, "logs", "rsl_rl", "g1_walking_tutorial"),
        # 워크스페이스 루트 기준
        os.path.join(script_dir, "..", "..", "logs", "rsl_rl", "g1_walking_tutorial"),
    ]

    def iteration_number(path: str) -> int:
        m = re.search(r"model_(\d+)\.pt$", path)
        return int(m.group(1)) if m else -1

    for log_dir in search_paths:
        log_dir = os.path.abspath(log_dir)
        if not os.path.exists(log_dir):
            continue
        ckpts = glob.glob(os.path.join(log_dir, "**", "model_*.pt"), recursive=True)
        if ckpts:
            # model_2000 > model_950이 되도록 iteration 번호 기준 정렬 (사전순 정렬은 오답)
            return max(ckpts, key=iteration_number)

    return None
