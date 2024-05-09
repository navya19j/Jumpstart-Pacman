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
import os

from huggingface_sb3 import EnvironmentName
import gymnasium as gym
import shutil
import argparse, sys
import torch
import zipfile

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

# folder = args.folder
BASE_PATH   = args.path
folder      = BASE_PATH.split("/")[-1]
DIR_PATH    = "/".join(BASE_PATH.split("/")[:-1])
print("BASE_PATH", BASE_PATH)
print("DIR_PATH", DIR_PATH)
POLICY_PATH = f"{BASE_PATH}/policy.pth"

print("Copying the file")
# create copy of file BASE_PATH
shutil.copy(f"{BASE_PATH}.zip", f"{BASE_PATH}_copy.zip")

print("Base path is", BASE_PATH)
print("Unzipping the folder")
# unzip the folder
# create a folder with the same name as the zip file
os.makedirs(BASE_PATH, exist_ok=True)
shutil.unpack_archive(f"{BASE_PATH}.zip", f"{BASE_PATH}", 'zip')

print("Loading the model")
# Remove guide policy from model
policy = torch.load(POLICY_PATH)
keys = list(policy.keys())
print("Keys are", keys)
for key in keys:
    if 'guide' in key:
        del policy[key]
        # print(f"Deleting {key}")

print("Saving the model")
# Save it back
torch.save(policy, POLICY_PATH)

print("Zipping the folder")
# zip the folder
shutil.make_archive(f"{BASE_PATH}_2", 'zip', f"{BASE_PATH}")

print("Making it a normal policy")
print("Loading model from", f"{BASE_PATH}_2")
model = PPO.load(f"{BASE_PATH}_2")
model_2 = PPO(
        "CnnPolicy",
        env)

print("Sharing the weights")
# share the weights
print("State dict of model is", model.policy.state_dict())
model_2.policy.load_state_dict(model.policy.state_dict().keys())

print("Saving the model")
# SAVE_PATH
# make directories if not exist
os.makedirs(f"{BASE_PATH}/ppo", exist_ok=True)
# also copy /home/nj2513/ppo/ALE-MsPacman-v5 folder to f"BASE_PATH}/ppo/
shutil.copytree("/home/nj2513/ppo/ALE-MsPacman-v5", f"{BASE_PATH}/ppo/ALE-MsPacman-v5")
model_2.save(f"{BASE_PATH}/ppo/ALE-MsPacman-v5")