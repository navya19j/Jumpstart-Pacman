import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_frame_stack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.contrib.common.wrappers import wrap_deepmind
from stable_baselines3.common.wrappers import NoopResetEnv, MaxAndSkipEnv
# from stable_baselines3.bench import Monitor
from stable_baselines3.common.vec_env import VecEnvWrapper, DummyVecEnv, VecNormalize, VecFrameStack, SubprocVecEnv

import warnings
warnings.filterwarnings("ignore")

def main():
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=200)
    env = NoopResetEnv(env, noop_max=2)
    env = MaxAndSkipEnv(env, skip=4)
    env = wrap_deepmind(env, episode_life=True, clip_rewards=False, frame_stack=True)
    # env = Monitor(env, log_dir, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])
    # stack 4 frames

    # model = TD3(
    #     "MultiInputPolicy",
    #     env,
    #     verbose=1,
    #     tensorboard_log="logs/pacman_guide"
    # )
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate = 2.5e-4,
        batch_size = 256,
        clip_range = 0.1,
        ent_coef = 0.01,
        n_epochs = 4,
        n_steps = 128,
        verbose=1,
    )
    model.learn(
        total_timesteps=1e5,
        log_interval=100,
        progress_bar=True,
        callback=EvalCallback(
            env,
            n_eval_episodes=100,
            best_model_save_path="examples/models/pacman_guide_DQN"
        ),
    )

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

if __name__ == "__main__":
    main()
