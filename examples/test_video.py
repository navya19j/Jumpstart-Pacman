import shutil
from stable_baselines3 import PPO
from collections import OrderedDict
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from gymnasium import spaces
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)

from huggingface_sb3 import EnvironmentName
import gymnasium as gym
import shutil
import argparse, sys
import torch

parser=argparse.ArgumentParser()

parser.add_argument("--path", help="path name", type=str)
args=parser.parse_args()

env_name = EnvironmentName("ALE/MsPacman-v5")
env_kwargs = {}
n_timesteps = 10
normalise = False
seed = 42
frame_stack = 4
vec_env_type = "dummy"
vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]

# Get environment spec
spec = gym.spec(env_name.gym_id)

def make_env(**kwargs) -> gym.Env:
    return spec.make(**kwargs)

# Make the environment
env = make_atari_env(
    make_env,
    n_envs=1,
    seed=seed,
    vec_env_cls=vec_env_class,
    vec_env_kwargs=env_kwargs,
)

if frame_stack:
    env = VecFrameStack(env, n_stack=frame_stack)

if not is_vecenv_wrapped(env, VecTransposeImage):
    wrap_with_vectranspose = False
    if isinstance(env.observation_space, spaces.Dict):
        # If even one of the keys is an image-space in need of transpose, apply transpose
        # If the image spaces are not consistent (for instance, one is channel first,
        # the other channel last); VecTransposeImage will throw an error
        for space in env.observation_space.spaces.values():
            wrap_with_vectranspose = wrap_with_vectranspose or (
                is_image_space(space) and not is_image_space_channels_first(space)  # type: ignore[arg-type]
            )
    else:
        wrap_with_vectranspose = is_image_space(env.observation_space) and not is_image_space_channels_first(
            env.observation_space  # type: ignore[arg-type]
        )

    if wrap_with_vectranspose:
        print("Wrapping the env in a VecTransposeImage.")
        env = VecTransposeImage(env)

model = PPO(
        "CnnPolicy",
        env,
        learning_rate = 2.5e-4,
        batch_size = 256,
        clip_range = 0.1,
        ent_coef = 0.01,
        vf_coef = 0.5,
        n_epochs = 4,
        n_steps = 128,
        verbose=1,
    )

# folder = args.folder
BASE_PATH   = args.path
folder      = BASE_PATH.split("/")[-1]
DIR_PATH    = "/".join(BASE_PATH.split("/")[:-1])
POLICY_PATH = f"{BASE_PATH}/policy.pth"

# unzip the folder
shutil.unpack_archive(f"{BASE_PATH}.zip", {BASE_PATH})

# create copy of folder
shutil.copytree(f"{BASE_PATH}", f"{DIR_PATH}/{folder}_copy")

# Remove guide policy from model
policy = torch.load(POLICY_PATH)
keys = list(model.keys())
for key in keys:
    if 'guide' in key:
        del model[key]

# Save it back
torch.save(model, POLICY_PATH)

# zip the folder
shutil.make_archive(f"{BASE_PATH}", 'zip', f"{BASE_PATH}")

model = PPO.load(f"{BASE_PATH}")
model_2 = PPO(
        "CnnPolicy",
        env,
        learning_rate = 2.5e-4,
        batch_size = 256,
        clip_range = 0.1,
        ent_coef = 0.01,
        vf_coef = 0.5,
        n_epochs = 4,
        n_steps = 128,
        verbose=1,
    )
# share the weights
model_2.policy.load_state_dict(model.policy.state_dict())

# SAVE_PATH
# make directories if not exist
shutil.rmtree(f"{BASE_PATH}/ppo")
# also copy /home/nj2513/ppo/ALE-MsPacman-v5 folder to f"BASE_PATH}/ppo/
shutil.copytree("/home/nj2513/ppo/ALE-MsPacman-v5", f"{BASE_PATH}/ppo/ALE-MsPacman-v5")
model_2.save(f"{BASE_PATH}/ppo/ALE-MsPacman-v5")