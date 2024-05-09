import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env, make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecFrameStack,
    VecNormalize,
    VecTransposeImage,
    is_vecenv_wrapped,
)
from collections import OrderedDict
from stable_baselines3.common.preprocessing import is_image_space, is_image_space_channels_first
from gymnasium import spaces
from huggingface_sb3 import EnvironmentName
from stable_baselines3.common.evaluation import evaluate_policy

import warnings
warnings.filterwarnings("ignore")

def main():

    env_name = EnvironmentName("ALE/MsPacman-v5")
    env_kwargs = {}
    n_timesteps = 1e7
    normalize = False
    seed = 0
    frame_stack = 4
    vec_env_type = "dummy"
    vec_env_class = {"dummy": DummyVecEnv, "subproc": SubprocVecEnv}[vec_env_type]

    spec = gym.spec(env_name.gym_id)

    def make_env(**kwargs) -> gym.Env:
        return spec.make(**kwargs)

    # Make the environment
    env = make_atari_env(
        make_env,
        n_envs=8,
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
        batch_size = 512,
        clip_range = 0.1,
        ent_coef = 0.01,
        vf_coef = 0.5,
        n_epochs = 4,
        n_steps = 128,
        verbose=1,
        tensorboard_log="logs/pacmanv5_guide",
    )
    model.learn(
        total_timesteps=n_timesteps,
        log_interval=10,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacmanv5_guide_PPO"
        ),
    )

    # save the model
    model.save("examples/models/pacmanv5_guide_PPO")

if __name__ == "__main__":
    main()
