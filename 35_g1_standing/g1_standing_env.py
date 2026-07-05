"""
g1_standing_env.py - G1 Standing 공유 환경 모듈 (실행 파일 아님)

35강 시리즈(35_1 ~ 35_3)가 공통으로 사용하는 환경 정의를 모아둔 모듈입니다:
  - G1SceneCfg: 평지 + Unitree G1 휴머노이드 + 조명
  - ObservationsCfg: 관측 6항목 (몸체 속도, 중력 방향, 관절 상태, 직전 행동)
  - ActionsCfg: 관절 위치 명령 (scale=0.25로 과격한 움직임 방지)
  - RewardsCfg: "서 있기" 보상 설계 (높이 0.74m 유지가 핵심)
  - TerminationsCfg: 에피소드 종료 3조건 (시간초과 / 기울어짐 / 추락)
  - G1StandingEnvCfg: 위 요소를 조립한 완성 환경 설정
  - make_ppo_runner_cfg(): RSL-RL PPO 하이퍼파라미터 설정

학습(35_2)과 평가(35_3)에서 관측/보상 정의나 네트워크 형상이 조금이라도
다르면 체크포인트를 로드해도 정상 동작하지 않으므로, 반드시 이 모듈
하나만 수정합니다.

주의:
    이 모듈은 isaaclab 패키지를 import하므로, 반드시 AppLauncher로
    시뮬레이터를 먼저 실행한 뒤에 import해야 합니다. (각 실행 스크립트 참고)
"""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.envs.mdp as mdp

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip


# ══════════════════════════════════════════════════════════════════════════
#  씬 구성: 평지 + G1 휴머노이드
# ══════════════════════════════════════════════════════════════════════════


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """G1 Standing 학습 씬: 바닥 + G1 휴머노이드 + 조명."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


# ══════════════════════════════════════════════════════════════════════════
#  커스텀 MDP 함수
# ══════════════════════════════════════════════════════════════════════════


def base_pos_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """몸체 높이가 0.74m(G1 기본 서있는 키)에 가까울수록 1에 가까운 보상.

    가우시안 형태라서 목표에서 멀어질수록 보상이 급격히 0으로 떨어집니다.
    """
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(z - 0.74) / 0.01)


def base_height_termination(env: ManagerBasedRLEnv, limit: float = 0.4) -> torch.Tensor:
    """몸체 높이가 limit 아래로 떨어지면(=넘어지면) 에피소드 종료."""
    return env.scene["robot"].data.root_pos_w[:, 2] < limit


# ══════════════════════════════════════════════════════════════════════════
#  MDP 구성: 행동 / 관측 / 보상 / 종료
# ══════════════════════════════════════════════════════════════════════════


@configclass
class ActionsCfg:
    """행동: 전체 관절의 목표 위치. scale=0.25로 한 번에 크게 못 움직이게 제한."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25)


@configclass
class ObservationsCfg:
    """관측: 에이전트가 매 스텝 보게 되는 상태(State) 항목들."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)              # 몸체 선속도 (x,y,z)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)              # 몸체 각속도
        projected_gravity = ObsTerm(func=mdp.projected_gravity)    # 중력 방향 투영 (기울기 감지)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)                # 관절 각도 (기본자세 대비 상대값)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)                # 관절 속도
        last_action = ObsTerm(func=mdp.last_action)                # 직전 행동

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """보상: '서 있기'를 점수로 정의. 가중치로 각 항목의 중요도를 조절."""

    alive_bonus = RewTerm(func=mdp.is_alive, weight=1.0)                    # 살아있으면 점수
    height_reward = RewTerm(func=base_pos_z_reward, weight=15.0)            # 키 0.74m 유지 (핵심!)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)   # 기울어지면 감점
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.01)                # 관절을 마구 흔들면 감점
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)             # 행동을 급격히 바꾸면 감점
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)          # 관절 한계를 넘으면 감점


@configclass
class TerminationsCfg:
    """종료: 넘어진 채로 시간을 낭비하지 않도록 에피소드를 빨리 끊는다."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)                                   # 5초 경과
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7854})    # 45도 이상 기울어짐
    base_height = DoneTerm(func=base_height_termination, params={"limit": 0.35})            # 0.35m 아래로 추락


# ══════════════════════════════════════════════════════════════════════════
#  완성 환경 설정
# ══════════════════════════════════════════════════════════════════════════


@configclass
class G1StandingEnvCfg(ManagerBasedRLEnvCfg):
    """G1 Standing 학습용 완성 환경 설정."""

    scene: G1SceneCfg = G1SceneCfg(num_envs=512, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4              # 물리 4스텝마다 정책 1회 실행 (제어 주기 50Hz)
        self.episode_length_s = 5.0      # 에피소드 최대 5초
        self.sim.dt = 0.005              # 물리 스텝 200Hz
        self.sim.render_interval = self.decimation


# ══════════════════════════════════════════════════════════════════════════
#  RSL-RL PPO 하이퍼파라미터 설정 (35_2 학습 / 35_3 평가 공용)
# ══════════════════════════════════════════════════════════════════════════


def make_ppo_runner_cfg(max_iterations: int = 1000) -> RslRlOnPolicyRunnerCfg:
    """G1 Standing PPO 설정. 평가 시에도 동일 설정으로 체크포인트를 로드합니다.

    하이퍼파라미터 설명:
      - num_steps_per_env=24: 24스텝의 경험을 모은 후 학습 1회
      - actor/critic_hidden_dims=[128,64,32]: 3층 신경망 (관측 → 128 → 64 → 32 → 출력)
      - learning_rate=1e-3: 한 번에 얼마나 크게 배울지
      - gamma=0.99: 미래 보상을 얼마나 중요하게 볼지 (1에 가까울수록 장기적)
      - clip_param=0.2: PPO의 핵심 — 정책이 한 번에 너무 크게 바뀌지 않도록 제한
    """
    return RslRlOnPolicyRunnerCfg(
        num_steps_per_env=24,
        max_iterations=max_iterations,
        save_interval=50,                     # 체크포인트 저장 주기
        experiment_name="g1_standing_tutorial",
        obs_groups={"policy": ["policy"]},
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,              # 가치 함수 손실 가중치
            use_clipped_value_loss=True,      # 가치 손실 클리핑
            clip_param=0.2,                   # PPO 클리핑 범위 ε
            entropy_coef=0.01,                # 엔트로피 보너스 (탐색 장려)
            learning_rate=1.0e-3,             # 학습률
            num_learning_epochs=5,            # 수집 데이터로 학습 반복 횟수
            num_mini_batches=8,               # 미니배치 수
            schedule="adaptive",              # 적응적 학습률 (KL 기반)
            gamma=0.99,                       # 할인율
            lam=0.95,                         # GAE lambda
            desired_kl=0.01,                  # 목표 KL divergence
            max_grad_norm=1.0,                # gradient 클리핑
        ),
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.5,               # 초기 탐색 노이즈
            actor_hidden_dims=[128, 64, 32],  # 정책(액터) 네트워크
            critic_hidden_dims=[128, 64, 32], # 가치(크리틱) 네트워크
            activation="elu",                 # 활성화 함수
            actor_obs_normalization=True,     # 관측 정규화 (학습 안정화)
            critic_obs_normalization=True,
        ),
    )
