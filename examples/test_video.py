import shutil
from stable_baselines3 import DQN
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
print("State dict of model is", policy.keys())
print("Saving the model")
# RuntimeError: Error(s) in loading state_dict for CnnPolicy:
#         Missing key(s) in state_dict: "q_net.features_extractor.cnn.0.weight", "q_net.features_extractor.cnn.0.bias", "q_net.features_extractor.cnn.2.weight", "q_net.features_extractor.cnn.2.bias", "q_net.features_extractor.cnn.4.weight", "q_net.features_extractor.cnn.4.bias", "q_net.features_extractor.linear.0.weight", "q_net.features_extractor.linear.0.bias", "q_net.q_net.0.weight", "q_net.q_net.0.bias", "q_net_target.features_extractor.cnn.0.weight", "q_net_target.features_extractor.cnn.0.bias", "q_net_target.features_extractor.cnn.2.weight", "q_net_target.features_extractor.cnn.2.bias", "q_net_target.features_extractor.cnn.4.weight", "q_net_target.features_extractor.cnn.4.bias", "q_net_target.features_extractor.linear.0.weight", "q_net_target.features_extractor.linear.0.bias", "q_net_target.q_net.0.weight", "q_net_target.q_net.0.bias". 
#         Unexpected key(s) in state_dict: "features_extractor.cnn.0.weight", "features_extractor.cnn.0.bias", "features_extractor.cnn.2.weight", "features_extractor.cnn.2.bias", "features_extractor.cnn.4.weight", "features_extractor.cnn.4.bias", "features_extractor.linear.0.weight", "features_extractor.linear.0.bias", "pi_features_extractor.cnn.0.weight", "pi_features_extractor.cnn.0.bias", "pi_features_extractor.cnn.2.weight", "pi_features_extractor.cnn.2.bias", "pi_features_extractor.cnn.4.weight", "pi_features_extractor.cnn.4.bias", "pi_features_extractor.linear.0.weight", "pi_features_extractor.linear.0.bias", "vf_features_extractor.cnn.0.weight", "vf_features_extractor.cnn.0.bias", "vf_features_extractor.cnn.2.weight", "vf_features_extractor.cnn.2.bias", "vf_features_extractor.cnn.4.weight", "vf_features_extractor.cnn.4.bias", "vf_features_extractor.linear.0.weight", "vf_features_extractor.linear.0.bias", "action_net.weight", "action_net.bias", "value_net.weight", "value_net.bias". 
# # change keys features_extractor to q_net.features_extractor

# Change the keys
new_policy = OrderedDict()
for key in policy.keys():
    new_key = key.replace("features_extractor", "q_net.features_extractor")
    new_policy[new_key] = policy[key]

# Save it back
# torch.save(policy, POLICY_PATH)
torch.save(new_policy, POLICY_PATH)


# print("Zipping the folder")
# # zip the folder
# shutil.make_archive(f"{BASE_PATH}_2", 'zip', f"{BASE_PATH}")

# print("Making it a normal policy")
# print("Loading model from", f"{BASE_PATH}_2")
# model = PPO.load(f"{BASE_PATH}_2", exact_match=True)
# print("State dict of model is", model.policy.state_dict().keys())
policy = torch.load(POLICY_PATH)
model_2 = DQN(
        "CnnPolicy",
        env)

print("Sharing the weights")
# share the weights
model_2.policy.load_state_dict(policy)

print("Saving the model")
# SAVE_PATH
# make directories if not exist
os.makedirs(f"{DIR_PATH}/dqn/ALE-MsPacman-v5", exist_ok=True)
# also copy /home/nj2513/dqn/ALE-MsPacman-v5 folder to f"DIR_PATH}/dqn/

shutil.copytree("/home/nj2513/dqn/ALE-MsPacman-v5", f"{DIR_PATH}/dqn/ALE-MsPacman-v5", dirs_exist_ok=True)
model_2.save(f"{DIR_PATH}/dqn/ALE-MsPacman-v5")