"""
g1_brucelee_env.py - G1 Bruce Lee(무술 모션 모방) 공유 환경 모듈 (실행 파일 아님)

36강의 보행 환경(g1_walking_env.py)을 상속하여, 제자리 무술 동작
(kungfu_retargeted/Bruce_Lee_pose.pkl, 330프레임/11초)을 학습할 수 있게
네 가지를 고칩니다. 모두 "보행 전용 가정"이 무술 모션과 충돌하는 지점입니다:

  1. 종료 조건: 고정 높이(0.35m) 추락 판정 → 레퍼런스 대비 편차 판정
     - 클립의 루트 최저 높이가 0.306m라서, 낮은 자세를 정확히 따라할수록
       고정 한계에 걸려 에피소드가 죽는다 (로그에서 리셋의 6.6%가 이 원인).
     - |z - ref_z| > 0.3m 이면 종료 (DeepMimic식 조기 종료의 간단판).

  2. 속도 보상: 전진 vx 추종 → 월드 평면 "속도 벡터" 추종
     - 이 클립은 heading 기준 전진 속도가 -0.85 ~ +0.90 m/s로 후진/횡이동이
       절반인데, 기존 speed_tracking은 vx와 |v|(항상 양수)를 비교해서
       후진 구간마다 모방을 방해한다.

  3. 회전 추종: 루트 yaw를 레퍼런스 yaw에 맞추는 보상 추가
     - 클립의 누적 yaw 변화 100°, 최대 247°/s. 기존 환경은 회전을 완전히
       무시(RSI가 항상 yaw=0)하므로 몸통 회전이 학습될 수 없다.
     - RSI도 레퍼런스의 루트 자세(쿼터니언)/속도 벡터/yaw 각속도로 초기화.

  4. 감점 완화: flat_orientation -2.0 → -0.2, joint_vel -0.01 → -0.005
     - 레퍼런스 자체가 최대 20° 기울고(발차기 준비 자세) 빠른 관절 속도
       (최대 4.7 rad/s)를 요구하므로, 감점이 모방과 싸우지 않게 줄인다.

클립이 거의 순환(시작↔끝 관절각 차이 0.04 rad, yaw 차이 10°)이라
보행 클립처럼 modulo 무한 반복 방식을 그대로 씁니다.

주의:
    g1_walking_env와 마찬가지로 AppLauncher 이후에 import해야 합니다.
"""

import os
import re
import glob

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

# 보행 환경 모듈을 통째로 가져와 필요한 부분만 덮어쓴다
import g1_walking_env as base
from g1_walking_env import (
    CYCLE_END,
    CYCLE_START,
    DEFAULT_MOTION_NAME,
    G1WalkingEnvCfg,
    MotionReference,
    make_ppo_runner_cfg,
)

DEFAULT_BRUCE_MOTION = "Bruce_Lee_pose"


# ══════════════════════════════════════════════════════════════════════════
#  레퍼런스 모션: 루트 회전(yaw)까지 관리하는 확장판
# ══════════════════════════════════════════════════════════════════════════


def _quat_yaw_wxyz(q: torch.Tensor) -> torch.Tensor:
    """쿼터니언(wxyz)에서 yaw(z축 회전각)만 추출합니다. (ZYX 오일러 기준)"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class MotionReferenceYaw(MotionReference):
    """MotionReference 확장: 루트 자세(쿼터니언)·평면 속도 벡터·yaw를 추가 제공.

    보행 환경은 '항상 +x로 전진'을 가정해서 루트 회전 정보를 버렸지만,
    무술 모션은 몸통이 도는 것 자체가 동작이므로 아래를 추가 계산합니다:
      - ref_quat_wxyz: 프레임별 루트 쿼터니언 (RSI 초기화용, Isaac은 wxyz 순서)
      - ref_yaw:       프레임별 yaw (unwrap 처리 → 보상에서 차이만 wrap)
      - ref_yaw_rate:  프레임별 yaw 각속도 (RSI 초기 각속도용)
      - ref_vel_xy:    프레임별 월드 평면 속도 벡터 (속도 벡터 추종 보상용)
      - ref_vel_z:     프레임별 수직 속도 (RSI 초기 속도용 — 앉았다 일어서기)
    """

    def load_file(self, motion=None) -> None:
        super().load_file(motion)

        # 부모가 쓴 것과 같은 트림 구간을 재계산 (기본 보행 클립만 트림)
        if self.pkl_name == DEFAULT_MOTION_NAME:
            s, e = CYCLE_START, CYCLE_END
        else:
            s, e = 0, self.dof_full.shape[0]

        # pkl의 root_rot은 xyzw 순서 → Isaac/IsaacLab 표준 wxyz로 변환
        rot_xyzw = self.root_rot_full[s:e]
        self.ref_quat_wxyz = rot_xyzw[:, [3, 0, 1, 2]].clone()

        # yaw는 ±π에서 점프하므로 unwrap해 연속 함수로 만든다
        yaw = _quat_yaw_wxyz(self.ref_quat_wxyz)
        self.ref_yaw = torch.from_numpy(
            np.unwrap(yaw.numpy().astype(np.float64))
        ).float()
        yaw_nxt = torch.roll(self.ref_yaw, shifts=-1, dims=0)
        self.ref_yaw_rate = (yaw_nxt - self.ref_yaw) * self.fps
        self.ref_yaw_rate[-1] = self.ref_yaw_rate[-2]  # wrap 지점 점프 제거

        # 월드 평면 속도 "벡터" (기존 ref_speed는 크기만 있어 방향을 잃는다)
        root_trans = self.root_trans_full[s:e]
        nxt = torch.roll(root_trans, shifts=-1, dims=0)
        vel = (nxt - root_trans) * self.fps
        self.ref_vel_xy = vel[:, :2].clone()
        self.ref_vel_z = vel[:, 2].clone()
        self.ref_vel_xy[-1] = self.ref_vel_xy[-2]
        self.ref_vel_z[-1] = self.ref_vel_z[-2]

        print(f"[MotionRefYaw] yaw 범위 {torch.rad2deg(self.ref_yaw.min()):.0f}°"
              f" ~ {torch.rad2deg(self.ref_yaw.max()):.0f}°, "
              f"최대 yaw 속도 {torch.rad2deg(self.ref_yaw_rate.abs().max()):.0f}°/s, "
              f"평면 속도 최대 {self.ref_vel_xy.norm(dim=1).max():.2f} m/s")

    def initialize(self, env: ManagerBasedRLEnv) -> None:
        if self._ready:
            return
        super().initialize(env)
        for name in ("ref_quat_wxyz", "ref_yaw", "ref_yaw_rate", "ref_vel_xy", "ref_vel_z"):
            setattr(self, name, getattr(self, name).to(env.device))


# 모듈 싱글턴. 중요: 보행 모듈의 관측/보상 함수(joint_tracking_reward 등)는
# g1_walking_env.MOTION 전역을 참조하므로, 그 전역도 이 인스턴스로 교체해서
# 상속받은 모든 항이 회전 인식 버전 하나를 공유하게 만든다.
MOTION = MotionReferenceYaw()
base.MOTION = MOTION


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 관측 (보행 103차원 + 4차원 = 107차원)
# ══════════════════════════════════════════════════════════════════════════


def yaw_err_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """로봇 yaw − 레퍼런스 yaw 오차를 sin/cos 2차원으로 관측.

    정책 관측에는 절대 yaw가 없으므로(각속도·중력투영뿐), 회전을 추종하려면
    "지금 얼마나 돌아야 하는지"를 직접 보여줘야 합니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    yaw = _quat_yaw_wxyz(env.scene["robot"].data.root_quat_w)
    err = yaw - MOTION.ref_yaw[idx]
    return torch.stack([torch.sin(err), torch.cos(err)], dim=1)


def ref_vel_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    """현재 위상의 레퍼런스 평면 속도 벡터를 로봇 yaw 기준으로 회전해 관측 (2차원).

    "지금 몸 기준 어느 방향으로 얼마나 움직여야 하는지"를 알려줍니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    v = MOTION.ref_vel_xy[idx]
    yaw = _quat_yaw_wxyz(env.scene["robot"].data.root_quat_w)
    c, s = torch.cos(yaw), torch.sin(yaw)
    vx_b = c * v[:, 0] + s * v[:, 1]
    vy_b = -s * v[:, 0] + c * v[:, 1]
    return torch.stack([vx_b, vy_b], dim=1)


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 보상
# ══════════════════════════════════════════════════════════════════════════


def root_vel_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """월드 평면 속도 '벡터'가 레퍼런스와 가까울수록 보상.

    RSI가 루트 yaw를 레퍼런스 절대 yaw로 맞춰 시작하므로, 월드 좌표계에서
    벡터끼리 직접 비교할 수 있습니다. 전진/후진/횡이동이 모두 공정하게
    평가됩니다 (기존 speed_tracking의 vx-vs-|v| 문제 해결).
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    v = env.scene["robot"].data.root_lin_vel_w[:, :2]
    err = torch.sum(torch.square(v - MOTION.ref_vel_xy[idx]), dim=1)
    return torch.exp(-4.0 * err)


def yaw_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """루트 yaw가 레퍼런스 yaw에 가까울수록 보상 (몸통 회전 모방의 핵심)."""
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    yaw = _quat_yaw_wxyz(env.scene["robot"].data.root_quat_w)
    err = yaw - MOTION.ref_yaw[idx]
    err = torch.atan2(torch.sin(err), torch.cos(err))  # ±π로 wrap
    return torch.exp(-4.0 * torch.square(err))


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수: 종료 / 이벤트(RSI)
# ══════════════════════════════════════════════════════════════════════════


def ref_height_deviation_termination(env: ManagerBasedRLEnv, limit: float = 0.3) -> torch.Tensor:
    """루트 높이가 레퍼런스 높이에서 limit 이상 벗어나면 종료.

    고정 한계(0.35m)와 달리 낮은 자세(클립 최저 0.306m)를 허용하면서도,
    "서 있어야 할 때 넘어져 있음"은 즉시 걸러냅니다.
    """
    MOTION.initialize(env)
    idx = MOTION.frame_idx(env)
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return (z - MOTION.ref_root_z[idx]).abs() > limit


def reset_to_motion_frame_yaw(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """RSI 확장판: 관절 상태에 더해 루트 회전/속도 벡터/각속도까지 레퍼런스로 초기화.

    보행판 RSI는 yaw=0 + 전진 속도만 맞췄지만, 회전하는 모션은
    (1) 루트 쿼터니언을 그 프레임의 레퍼런스 자세로,
    (2) 선속도를 월드 속도 벡터(수직 포함)로,
    (3) z축 각속도를 레퍼런스 yaw 각속도로
    맞춰야 "동작 중간부터 자연스럽게 이어서" 시작할 수 있습니다.
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

    # (2) 루트 상태: env 원점 + 레퍼런스 높이/회전/속도
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:2] = env.scene.env_origins[env_ids, 0:2]
    root_state[:, 2] = MOTION.ref_root_z[f]
    root_state[:, 3:7] = MOTION.ref_quat_wxyz[f]
    root_state[:, 7:9] = MOTION.ref_vel_xy[f]
    root_state[:, 9] = MOTION.ref_vel_z[f]
    root_state[:, 10:12] = 0.0
    root_state[:, 12] = MOTION.ref_yaw_rate[f]
    robot.write_root_pose_to_sim(root_state[:, 0:7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(root_state[:, 7:13], env_ids=env_ids)


# ══════════════════════════════════════════════════════════════════════════
#  MDP 구성: 보행판을 상속하고 충돌 지점만 교체
# ══════════════════════════════════════════════════════════════════════════


@configclass
class BruceObservationsCfg(base.ObservationsCfg):
    """관측: 보행 8항목(103차원) + 회전 모방용 2항목(4차원) = 107차원."""

    @configclass
    class PolicyCfg(base.ObservationsCfg.PolicyCfg):
        yaw_err = ObsTerm(func=yaw_err_obs)      # yaw 오차 sin/cos (2)
        ref_vel = ObsTerm(func=ref_vel_obs)      # 레퍼런스 속도 벡터, 몸 기준 (2)

    policy: PolicyCfg = PolicyCfg()


@configclass
class BruceRewardsCfg(base.RewardsCfg):
    """보상: 전진 속도 추종을 속도 벡터 추종으로 교체 + yaw 추종 추가."""

    speed_tracking = None                                                    # vx 추종은 후진/횡이동과 상충 → 제거
    root_vel_tracking = RewTerm(func=root_vel_tracking_reward, weight=2.0)   # 평면 속도 벡터 추종
    yaw_tracking = RewTerm(func=yaw_tracking_reward, weight=2.0)             # 몸통 회전 추종


@configclass
class BruceEventsCfg:
    """이벤트: 리셋 시 회전 인식 RSI 적용."""

    reset_motion = EventTerm(func=reset_to_motion_frame_yaw, mode="reset")


@configclass
class BruceTerminationsCfg(base.TerminationsCfg):
    """종료: 고정 높이 추락 판정을 레퍼런스 대비 편차 판정으로 교체."""

    base_height = DoneTerm(func=ref_height_deviation_termination, params={"limit": 0.3})


@configclass
class G1BruceLeeEnvCfg(G1WalkingEnvCfg):
    """G1 Bruce Lee(무술 모션 모방) 학습용 완성 환경 설정."""

    observations: BruceObservationsCfg = BruceObservationsCfg()
    rewards: BruceRewardsCfg = BruceRewardsCfg()
    events: BruceEventsCfg = BruceEventsCfg()
    terminations: BruceTerminationsCfg = BruceTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # 클립이 11초라 한 에피소드가 전체 동작 + 순환 연결부까지 경험하게 12초
        self.episode_length_s = 12.0
        # 레퍼런스 자체가 최대 20° 기울고(발차기) 빠른 관절 속도를 쓰므로
        # 모방과 싸우는 감점을 완화한다
        self.rewards.flat_orientation.weight = -0.2
        self.rewards.joint_vel.weight = -0.005


# ══════════════════════════════════════════════════════════════════════════
#  RSL-RL PPO 설정 / 체크포인트 탐색 (36_4 학습 / 36_5 평가 공용)
# ══════════════════════════════════════════════════════════════════════════

BRUCE_EXPERIMENT = "g1_brucelee_tutorial"


def make_bruce_ppo_cfg(max_iterations: int = 3000):
    """보행과 같은 PPO 설정에 실험 이름만 분리 (체크포인트 혼입 방지).

    기본 3000 iter: 무술 모션이 보행보다 자세 다양성이 커서 여유를 둔다.
    """
    cfg = make_ppo_runner_cfg(max_iterations)
    cfg.experiment_name = BRUCE_EXPERIMENT
    return cfg


def find_latest_bruce_checkpoint() -> str | None:
    """가장 최근 Bruce Lee 학습 체크포인트를 자동으로 찾습니다."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths = [
        os.path.join(os.getcwd(), "logs", "rsl_rl", BRUCE_EXPERIMENT),
        os.path.join(script_dir, "logs", "rsl_rl", BRUCE_EXPERIMENT),
    ]
    for log_dir in search_paths:
        ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
        if ckpts:
            # model_999.pt 같은 파일명에서 iteration 번호로 정렬
            ckpts.sort(key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))
            return ckpts[-1]
    return None
