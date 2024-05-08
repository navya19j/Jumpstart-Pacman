import gymnasium as gym
from stable_baselines3 import TD3
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import vec_frame_stack
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.evaluation import evaluate_policy

import warnings
warnings.filterwarnings("ignore")

def main():
    env = gym.make("ALE/MsPacman-v5", max_episode_steps=200)
    # stack 4 frames
    env = vec_frame_stack(env, n_stack=4)
    
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
