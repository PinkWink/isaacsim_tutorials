import argparse
import pickle
import random
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="g1_walking_2: Train G1 walking with motion reference")
parser.add_argument("--motion_pkl", type=str, default=None,
                    help="Walking motion pkl (default: ACCAD B3_walk1)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
MOTION_PKL = args_cli.motion_pkl
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
import isaaclab.envs.mdp as mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from isaaclab_assets.robots.unitree import G1_MINIMAL_CFG  # isort:skip


@configclass
class G1SceneCfg(InteractiveSceneCfg):
    """G1 보행 학습 씬: 바닥 + G1 휴머노이드 + 조명."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    )
    robot: ArticulationCfg = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )


def base_pos_z_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    z = env.scene["robot"].data.root_pos_w[:, 2]
    return torch.exp(-torch.square(z - 0.74) / 0.01)


def base_height_termination(env: ManagerBasedRLEnv, limit: float = 0.4) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < limit


def forward_velocity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.scene["robot"].data.root_lin_vel_b[:, 0]


def base_vertical_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    vz = env.scene["robot"].data.root_lin_vel_w[:, 2]
    return torch.square(vz)


# 보행 레퍼런스 모션 데이터는 용량 문제로 저장소에 포함하지 않습니다.
# 아래 명령으로 이 스크립트와 같은 폴더에 직접 내려받으세요:
#   git clone https://huggingface.co/datasets/openhe/g1-retargeted-motions g1_retargeted_motions
MOTIONS_DIR = Path(__file__).resolve().parent / "g1_retargeted_motions"

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
ALIAS = {
    "waist_yaw_joint": "torso_joint",
    "left_elbow_pitch_joint": "left_elbow_joint",
    "right_elbow_pitch_joint": "right_elbow_joint",
    "left_wrist_roll_joint": "left_elbow_roll_joint",
    "right_wrist_roll_joint": "right_elbow_roll_joint",
}
PKL_ARM_RANGE = range(13, 23)

_mref = {}


def _ensure_motion_init(env):
    if _mref.get("ready"):
        return
    device = env.scene["robot"].device
    pkl_path = (
        Path(MOTION_PKL) if MOTION_PKL
        else MOTIONS_DIR / "ACCAD_retargeted" / "B3_-_walk1_stageii.pkl"
    )
    if not pkl_path.exists():
        raise FileNotFoundError(f"Motion file not found: {pkl_path}\n ")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    motion = data[list(data.keys())[0]]
    _mref["dof"] = torch.tensor(motion["dof"], dtype=torch.float32, device=device)
    _mref["fps"] = int(motion["fps"])
    _mref["T"] = len(motion["dof"])

    names = env.scene["robot"].data.joint_names
    mapping = []
    for pkl_name in PKL_JOINT_NAMES:
        cand = [pkl_name, ALIAS.get(pkl_name, "")]
        found = -1
        for c in cand:
            if not c:
                continue
            key = c.replace("_joint", "")
            for i, n in enumerate(names):
                if key in n and (c in n or n.endswith(key + "_joint")):
                    found = i
                    break
            if found >= 0:
                break
        mapping.append(found)

    _mref["map"] = mapping
    _mref["arm_idx"] = [mapping[i] for i in PKL_ARM_RANGE if mapping[i] >= 0]
    _mref["all_idx"] = [ri for ri in mapping if ri >= 0]
    _mref["ready"] = True
    print(f"[MotionRef] {pkl_path.name}: {_mref['T']} frames @ {_mref['fps']} fps")


def _ref_frame_idx(env):
    step_dt = env.cfg.decimation * env.cfg.sim.dt
    t = env.episode_length_buf.float() * step_dt
    return (t * _mref["fps"]).long() % _mref["T"]


def _ref_joint_pos(env):
    _ensure_motion_init(env)
    idx = _ref_frame_idx(env)
    ref_dof = _mref["dof"][idx]
    out = env.scene["robot"].data.default_joint_pos.clone()
    for pi, ri in enumerate(_mref["map"]):
        if ri >= 0:
            out[:, ri] = ref_dof[:, pi]
    return out


def ref_joint_pos_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    ref = _ref_joint_pos(env)
    return ref - env.scene["robot"].data.default_joint_pos


def gait_phase_obs(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_motion_init(env)
    idx = _ref_frame_idx(env)
    phase = idx.float() / _mref["T"] * 2.0 * torch.pi
    return torch.stack([torch.sin(phase), torch.cos(phase)], dim=1)


def joint_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    ref = _ref_joint_pos(env)
    cur = env.scene["robot"].data.joint_pos
    idx = _mref["all_idx"]
    err = torch.sum(torch.square(cur[:, idx] - ref[:, idx]), dim=1)
    return torch.exp(-2.0 * err)


def arm_tracking_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    _ensure_motion_init(env)
    ref = _ref_joint_pos(env)
    cur = env.scene["robot"].data.joint_pos
    idx = _mref["arm_idx"]
    if not idx:
        return torch.zeros(env.num_envs, device=env.device)
    err = torch.sum(torch.square(cur[:, idx] - ref[:, idx]), dim=1)
    return torch.exp(-5.0 * err)


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        last_action = ObsTerm(func=mdp.last_action)
        ref_joint = ObsTerm(func=ref_joint_pos_obs)
        phase = ObsTerm(func=gait_phase_obs)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    alive_bonus = RewTerm(func=mdp.is_alive, weight=1.0)
    forward_velocity = RewTerm(func=forward_velocity_reward, weight=1.0)
    height_reward = RewTerm(func=base_pos_z_reward, weight=3.0)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    joint_tracking = RewTerm(func=joint_tracking_reward, weight=5.0)
    arm_tracking = RewTerm(func=arm_tracking_reward, weight=3.0)
    vertical_vel = RewTerm(func=base_vertical_velocity_penalty, weight=-1.5)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.02)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.08)
    joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7854})
    base_height = DoneTerm(func=base_height_termination, params={"limit": 0.35})


@configclass
class G1EnvCfg(ManagerBasedRLEnvCfg):
    scene: G1SceneCfg = G1SceneCfg(num_envs=512, env_spacing=3.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation


def main():
    random.seed(7)
    env_cfg = G1EnvCfg()
    env_cfg.sim.device = args_cli.device
    env = ManagerBasedRLEnv(cfg=env_cfg)

    train_cfg = RslRlOnPolicyRunnerCfg(
        num_steps_per_env=32,
        max_iterations=3000,
        save_interval=100,
        experiment_name="g1_walking_2",
        obs_groups={"policy": ["policy"]},
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.008,
            learning_rate=5e-4,
            num_learning_epochs=5,
            num_mini_batches=8,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        policy=RslRlPpoActorCriticCfg(
            init_noise_std=0.4,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
            actor_obs_normalization=True,
            critic_obs_normalization=True,
        ),
    )

    wrapped_env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(wrapped_env, train_cfg.to_dict(), log_dir="logs/g1_walking_2")
    print("[INFO] g1_walking_2: Walking + motion mimic. Logs: logs/g1_walking_2")
    runner.learn(num_learning_iterations=train_cfg.max_iterations, init_at_random_ep_len=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
